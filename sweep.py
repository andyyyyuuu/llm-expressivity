import wandb
from train import tune_soft_prompt
from dotenv import load_dotenv
import os
from math import log
from tqdm import tqdm
from nnsight import LanguageModel

load_dotenv()

wandb.login(key=os.getenv("WANDB_API_KEY"))

wandb.init(project="expressivity-of-llms")

wandb.define_metric("target_entropy")
wandb.define_metric("best_loss", step_metric="target_entropy")

def get_range(start: float, end: float, step: float, round_to: int=12) -> list[float]:
    while start < end:
        yield round(start, round_to)
        start += step
    return
def run_experiment(prefix_length: int=5) -> None: 
    model = LanguageModel("meta-llama/Llama-3.2-1B", device_map="auto", dispatch=True)
    for i in tqdm(list(get_range(0, log(model.config.vocab_size), 0.05)), desc=f"sweeping up to H<={log(model.config.vocab_size):.2f}"):
        best_prompt, best_loss = tune_soft_prompt(model, i, prefix_length, lr=0.1, max_epochs=500, early_stop_patience=20, log_losses=False)
        wandb.log({
            "target_entropy": i,
            "best_loss": best_loss,
        })

if __name__ == "__main__":
    run_experiment()