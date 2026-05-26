from nnsight import LanguageModel
from dists import optimize_vanilla, set_seed
from tqdm import tqdm
import torch
from dotenv import load_dotenv
import os

load_dotenv()

DO_WANDB = os.getenv("DO_WANDB_TRAINING", "0") == "1"

if DO_WANDB:
    import wandb
    wandb.login(key=os.getenv("WANDB_API_KEY"))


def tune_soft_prompt(model: LanguageModel, target_entropy: float, prefix_length: int=5, lr: float=0.1, max_epochs: int=500, early_stop_patience: int=20, log_losses: bool=True) -> tuple[torch.nn.Parameter, float]:

    if DO_WANDB:
        wandb.init(project="expressivity-of-llms-training", 
                   config={
                       "target_entropy": target_entropy,
                       "prefix_length": prefix_length,
                       "lr": lr,
                   })

    for param in model.model.parameters():
        param.requires_grad = False
    
    vocab_size = model.config.vocab_size
    embed_size = model.config.hidden_size
    target_logits = optimize_vanilla(target_entropy, dist_size=vocab_size, epsilon=1e-4)
    target_logits.requires_grad = False
    target_log_probs = torch.log_softmax(target_logits, dim=-1)
    
    soft_prompt = torch.nn.Parameter(torch.randn(prefix_length, embed_size, device=model.device), requires_grad=True)
    target_log_probs = target_log_probs.to(model.device)

    optimizer = torch.optim.Adam([soft_prompt], lr=lr)
    loss_fn = torch.nn.KLDivLoss(log_target=True, reduction="batchmean")

    no_improvement_count = 0
    best_loss = float('inf')
    best_prompt = None

    for epoch in tqdm(range(max_epochs), desc=f"training H={target_entropy:.2f}"):
        with model.trace(torch.tensor([[0] * prefix_length])): 
            model.model.embed_tokens.output = soft_prompt.unsqueeze(0) # (1, L, H)
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
            tqdm.write(f"Early stopping at epoch {epoch}")
            break
    return best_prompt, best_loss


if __name__ == "__main__":
    set_seed(42)
    model = LanguageModel("meta-llama/Llama-3.2-1B", device_map="auto", dispatch=True)
    target_entropy = 10
    prefix_length = 5
    lr = 0.1
    max_epochs = 500
    early_stop_patience = 20
    best_prompt, best_loss = tune_soft_prompt(model, target_entropy, prefix_length, lr, max_epochs, early_stop_patience)
    print(f"Best prompt: {best_prompt}")
    print(f"Best loss: {best_loss}")
