import csv
from .train import tune_soft_prompt
from dotenv import load_dotenv
import os
from datetime import datetime
from math import log
from pathlib import Path
from tqdm.auto import tqdm
import torch
from .targets import Targets
from .utils import set_seed
from .modules import DownstreamModule, LayerIntervention

load_dotenv()

from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"))


def run_experiment(save_path: str, module: DownstreamModule, targets: Targets) -> None: 

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_entropy", "best_loss", "early_stopped", "prompt"])
        f.flush()
        for i, (H, target) in enumerate(tqdm(targets, desc=f"sweeping targets")):
            set_seed(int(os.getenv("TRAINING_SEED", "216")) + i)
            best_prompt, best_loss, early_stopped = tune_soft_prompt(module, H, target, lr=0.1, max_epochs=500, early_stop_patience=20, log_losses=False)
            writer.writerow([H, best_loss, int(early_stopped)])
            f.flush()



if __name__ == "__main__":
    from .targets import OptimEntropyGrid
    seed = int(os.getenv("TARGETS_SEED", "216"))
    module = LayerIntervention(layer=5, prefix_length=5)
    targets = OptimEntropyGrid(module=module, seed=seed, is_outlier=False)
    save_path = Path("saves") / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M')}_l5.csv"
    tqdm.write(f"writing results to {save_path}")
    run_experiment(str(save_path), module, targets)
