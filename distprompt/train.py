from nnsight import LanguageModel
from tqdm.auto import tqdm
import torch
from dotenv import load_dotenv
import os
from .utils import set_seed, log_entropy
from .modules import DownstreamModule, LayerIntervention
from typing import Callable

load_dotenv()

DO_WANDB = os.getenv("DO_WANDB_TRAINING", "0") == "1"

if DO_WANDB:
    import wandb
    wandb.login(key=os.getenv("WANDB_API_KEY"))


def entropy_only_loss(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_entropy = target if target.ndim == 0 else log_entropy(target, dim=-1)
    return (log_entropy(x, dim=-1) - target_entropy) ** 2


def tune_soft_prompt(module: DownstreamModule, target_entropy: float, target_logits: torch.Tensor | None, loss_fn: Callable = None, lr: float=0.1, max_epochs: int=500, early_stop_patience: int=20, log_losses: bool=True) -> tuple[torch.nn.Parameter, float, bool]:
    
    if DO_WANDB:
        wandb.init(project="expressivity-of-llms-training", 
                   config={
                       "target_entropy": target_entropy,
                       "prefix_length": module.prefix_length,
                       "intervention": module.__repr__(),
                       "lr": lr,
                   })

    if target_logits is None:
        if loss_fn is None:
            raise ValueError("target_logits must be provided when using the default KL loss")
        target_for_loss = torch.tensor(target_entropy, device=module.device, dtype=torch.float32)
    else:
        target_logits.requires_grad = False
        target_for_loss = torch.log_softmax(target_logits.float(), dim=-1).unsqueeze(0)
    
    soft_prompt = torch.nn.Parameter(
        torch.randn(module.input_shape, device=module.device, dtype=torch.float32),
        requires_grad=True,
    )

    optimizer = torch.optim.Adam([soft_prompt], lr=lr)
    loss_fn = loss_fn or torch.nn.KLDivLoss(log_target=True, reduction="batchmean")

    no_improvement_count = 0
    best_loss = float('inf')
    best_prompt = None

    early_stopped = False

    for epoch in tqdm(range(max_epochs), desc=f"training H={target_entropy:.2f}", leave=False):
        log_probs = module.forward(soft_prompt).float()
        loss = loss_fn(log_probs.unsqueeze(0), target_for_loss)
        current_prompt = soft_prompt.detach().clone()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = max(loss.item(), 0.0)
        if log_losses and epoch % 10 == 0:
            tqdm.write(f"Epoch {epoch}, Loss: {loss_value}")
        
        if loss_value < best_loss:
            best_loss = loss_value
            best_prompt = current_prompt
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        if DO_WANDB:
            wandb.log({
                "loss": loss_value,
            })
        if no_improvement_count >= early_stop_patience:
            if log_losses:
                tqdm.write(f"Early stopping at epoch {epoch}")
            early_stopped = True
            break
    return best_prompt, best_loss, early_stopped


if __name__ == "__main__":
    from .dists import optimize_vanilla

    set_seed(int(os.getenv("TRAINING_SEED", "216")))
    module = LayerIntervention(layer=5, prefix_length=5)
    target_entropy = 0.5
    lr = 0.1
    max_epochs = 500
    early_stop_patience = 20
    target_logits = optimize_vanilla(target_entropy, module.vocab_size, 1e-4, device=module.device)
    best_prompt, best_loss, early_stopped = tune_soft_prompt(
        module,
        target_entropy,
        target_logits,
        lr=lr,
        max_epochs=max_epochs,
        early_stop_patience=early_stop_patience,
    )
    print(f"Best prompt: {best_prompt}")
    print(f"Best loss: {best_loss}")
    print(f"Early stopped: {early_stopped}")
