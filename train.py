from nnsight import LanguageModel
from tqdm.auto import tqdm
import torch
from dotenv import load_dotenv
import os
from config import InterventionConfig
from utils import set_seed

load_dotenv()

DO_WANDB = os.getenv("DO_WANDB_TRAINING", "0") == "1"

if DO_WANDB:
    import wandb
    wandb.login(key=os.getenv("WANDB_API_KEY"))



def intervene(model, patch: torch.Tensor, config: InterventionConfig) -> None: 
    assert patch.shape == (1, config.prefix_length, model.config.hidden_size)
    if config.type == "embed":
        model.model.embed_tokens.output = patch
    elif config.type == "layer":
        model.model.layers[config.layer].output[0] = patch


def tune_soft_prompt(model: LanguageModel, target_entropy: float, target_logits: torch.Tensor, config: InterventionConfig, lr: float=0.1, max_epochs: int=500, early_stop_patience: int=20, log_losses: bool=True) -> tuple[torch.nn.Parameter, float, bool]:
    
    config.check_valid(model)

    if DO_WANDB:
        wandb.init(project="expressivity-of-llms-training", 
                   config={
                       "target_entropy": target_entropy,
                       "prefix_length": config.prefix_length,
                       "intervention_type": config.type,
                       "intervention_layer": config.layer,
                       "lr": lr,
                   })

    for param in model.model.parameters():
        param.requires_grad = False
    
    model_dtype = model.model.embed_tokens.weight.dtype
    target_logits.requires_grad = False
    target_log_probs = torch.log_softmax(target_logits, dim=-1)
    
    soft_prompt = torch.nn.Parameter(
        torch.randn(config.prefix_length, model.config.hidden_size, device=model.device, dtype=model_dtype),
        requires_grad=True,
    )

    optimizer = torch.optim.Adam([soft_prompt], lr=lr)
    loss_fn = torch.nn.KLDivLoss(log_target=True, reduction="batchmean")

    no_improvement_count = 0
    best_loss = float('inf')
    best_prompt = None

    early_stopped = False

    for epoch in tqdm(range(max_epochs), desc=f"training H={target_entropy:.2f}", leave=False):
        input_ids = torch.zeros((1, config.prefix_length), dtype=torch.long, device=model.device)
        with model.trace(input_ids):
            intervene(model, soft_prompt.unsqueeze(0), config)
            logits = model.output.save() # (1, L, V)
        final_logits = logits.logits.squeeze(0)[-1, :] # (V)
        log_probs = torch.log_softmax(final_logits, dim=-1)
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
    model = LanguageModel("meta-llama/Llama-3.2-1B", device_map="auto", dispatch=True)
    target_entropy = 0.5
    prefix_length = 5
    lr = 0.1
    max_epochs = 500
    early_stop_patience = 20
    target_logits = optimize_vanilla(target_entropy, model.config.vocab_size, 1e-4, device=model.device)
    best_prompt, best_loss, early_stopped = tune_soft_prompt(model, target_entropy, target_logits, InterventionConfig(type="layer", layer=5, prefix_length=prefix_length), lr, max_epochs, early_stop_patience)
    print(f"Best prompt: {best_prompt}")
    print(f"Best loss: {best_loss}")
    print(f"Early stopped: {early_stopped}")
