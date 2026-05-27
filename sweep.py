import csv
from train import tune_soft_prompt
from dotenv import load_dotenv
import os
from datetime import datetime
from math import log
from pathlib import Path
from tqdm import tqdm
from nnsight import LanguageModel

load_dotenv()

def get_range(start: float, end: float, step: float, round_to: int=12) -> list[float]:
    while start < end:
        yield round(start, round_to)
        start += step
    return

def run_experiment(save_path: str, prefix_length: int=5) -> None: 
    model = LanguageModel("meta-llama/Llama-3.2-1B", device_map="auto", dispatch=True)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_entropy", "best_loss"])
        f.flush()
        for i in tqdm(list(get_range(0, log(model.config.vocab_size), 0.05)), desc=f"sweeping up to H<={log(model.config.vocab_size):.2f}"):
            best_prompt, best_loss = tune_soft_prompt(model, i, prefix_length, lr=0.1, max_epochs=500, early_stop_patience=20, log_losses=False)
            writer.writerow([i, best_loss])
            f.flush()

if __name__ == "__main__":
    save_path = Path("saves") / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"Writing results to {save_path}")
    run_experiment(save_path=str(save_path))