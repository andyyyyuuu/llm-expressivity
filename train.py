from nnsight import LanguageModel
from tqdm.auto import tqdm
import torch
from dotenv import load_dotenv
import os
from utils import set_seed
from module import DownstreamModule, LayerIntervention

load_dotenv()

DO_WANDB = os.getenv("DO_WANDB_TRAINING", "0") == "1"

if DO_WANDB:
    import wandb
    wandb.login(key=os.getenv("WANDB_API_KEY"))


def tune_soft_prompt(module: DownstreamModule, target_entropy: float, target_logits: torch.Tensor, lr: float=0.1, max_epochs: int=500, early_stop_patience: int=20, log_losses: bool=True) -> tuple[torch.nn.Parameter, float, bool]:
    
    if DO_WANDB:
        wandb.init(project="expressivity-of-llms-training", 
                   config={
                       "target_entropy": target_entropy,
                       "prefix_length": module.prefix_length,
                       "intervention": module.__repr__(),
                       "lr": lr,
                   })

    target_logits.requires_grad = False
    target_log_probs = torch.log_softmax(target_logits, dim=-1)
    
    soft_prompt = torch.nn.Parameter(
        torch.randn(module.input_shape, device=module.device, dtype=torch.float32),
        requires_grad=True,
    )

    optimizer = torch.optim.Adam([soft_prompt], lr=lr)
    loss_fn = torch.nn.KLDivLoss(log_target=True, reduction="batchmean")

    no_improvement_count = 0
    best_loss = float('inf')
    best_prompt = None

    early_stopped = False

    for epoch in tqdm(range(max_epochs), desc=f"training H={target_entropy:.2f}", leave=False):
        log_probs = module.forward(soft_prompt)
        loss = loss_fn(log_probs.unsqueeze(0), target_log_probs.unsqueeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if log_losses and epoch % 10 == 0:
            tqdm.write(f"Epoch {epoch}, Loss: {loss.item()}")
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_prompt = soft_prompt.detach().clone()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        if DO_WANDB:
            wandb.log({
                "loss": loss.item(),
            })
        if no_improvement_count >= early_stop_patience:
            if log_losses:
                tqdm.write(f"Early stopping at epoch {epoch}")
            early_stopped = True
            break
    return best_prompt, best_loss, early_stopped


if __name__ == "__main__":
    from dists import optimize_vanilla

    set_seed(int(os.getenv("TRAINING_SEED", "216")))
    module = LayerIntervention(layer=5, prefix_length=5)
    target_entropy = 0.5
    lr = 0.1
    max_epochs = 500
    early_stop_patience = 20
    target_logits = optimize_vanilla(target_entropy, module.vocab_size, 1e-4, device=module.device)
    best_prompt, best_loss, early_stopped = tune_soft_prompt(module, target_entropy, target_logits, lr, max_epochs, early_stop_patience)
    print(f"Best prompt: {best_prompt}")
    print(f"Best loss: {best_loss}")
    print(f"Early stopped: {early_stopped}")
