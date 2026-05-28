import csv
from train import tune_soft_prompt
from dotenv import load_dotenv
import os
from datetime import datetime
from math import log
from pathlib import Path
from tqdm.auto import tqdm
from nnsight import LanguageModel
from config import InterventionConfig
import torch
from dists import optimize_vanilla
from utils import set_seed

load_dotenv()

def get_range(start: float, end: float, step: float, round_to: int=12) -> list[float]:
    while start < end:
        yield round(start, round_to)
        start += step
    return



def load_targets(entropies: list[float], vocab_size: int, cache_path: str, device: torch.device | str, epsilon: float=1e-4) -> list[tuple[float, torch.Tensor]]:
    seed = int(os.getenv("TARGETS_SEED", "216"))
    if Path(cache_path).exists():
        loaded = torch.load(cache_path, map_location=device)
        if loaded['seed'] != seed:
            raise ValueError(f"cached targets have seed {loaded['seed']}, but requested {seed}")
        if len(loaded['targets']) != len(entropies):
            raise ValueError(f"cached targets have {len(loaded['targets'])} entries, but requested {len(entropies)}")
        
        print(f"loaded {len(loaded['targets'])} targets from {cache_path} with seed {loaded['seed']}")
        return loaded["targets"]
    
    targets = []
    print(f"synthesizing {len(entropies)} targets with seed {seed} and epsilon {epsilon} to {cache_path}")
    for i, H in enumerate(tqdm(entropies, desc="synthesizing target logits")):
        set_seed(seed + i)
        targets.append((H, optimize_vanilla(H, vocab_size, epsilon, device=device)))
    
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"targets": targets, "seed": seed, "epsilon": epsilon}, cache_path)

    return targets



model = LanguageModel("meta-llama/Llama-3.2-1B", device_map="auto", dispatch=True)


def run_experiment(save_path: str, intervention_config: InterventionConfig=InterventionConfig(type="layer", layer=5, prefix_length=5)) -> None: 
    target_dists = load_targets(list(get_range(0, log(model.config.vocab_size), 0.05)), model.config.vocab_size, f"saves/targets_{model.config.vocab_size}.pt", model.device)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_entropy", "best_loss", "early_stopped"])
        f.flush()
        for i in tqdm(range(len(target_dists)), desc=f"sweeping up to H<={target_dists[-1][0]:.2f}"):
            set_seed(int(os.getenv("TRAINING_SEED", "216")) + i)
            best_prompt, best_loss, early_stopped = tune_soft_prompt(model, target_dists[i][0], target_dists[i][1], intervention_config, lr=0.1, max_epochs=500, early_stop_patience=20, log_losses=False)
            writer.writerow([target_dists[i][0], best_loss, int(early_stopped)])
            f.flush()


if __name__ == "__main__":
    layer = 5
    intervention_config = InterventionConfig(type="layer", layer=layer, prefix_length=5)
    save_path = Path("saves") / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M')}_l{layer}.csv"
    print(f"writing results to {save_path}")
    run_experiment(str(save_path), intervention_config)
