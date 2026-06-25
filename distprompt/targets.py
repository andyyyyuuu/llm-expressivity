from abc import abstractmethod, ABC
import torch
from tqdm.auto import tqdm
from pathlib import Path
from math import log
from typing import Iterator
import os

from .utils import set_seed, find_device, entropy, log_entropy
from .modules import DownstreamModule
from .train import tune_soft_prompt, entropy_only_loss


def L(logits: torch.Tensor, targ_entropy: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    actual_entropy = entropy(probs, dim=-1)
    return (actual_entropy - targ_entropy) ** 2

def L_prime(alpha: torch.nn.Parameter, beta: torch.nn.Parameter) -> torch.Tensor:
    raise NotImplementedError("Not implemented")

def get_range(start: float, end: float, step: float, round_to: int=12) -> list[float]:
    while start < end:
        yield round(start, round_to)
        start += step
    return


CACHE_PATH = "saves/{target_type}_{vocab_size}_{seed}.pt"

class Targets(ABC):

    def __init__(self, target_type: str, module: DownstreamModule, seed: int | None = None) -> None:
        """
        Note: this constructor eagerly-loads targets from cache if it exists.
        """
        self.seed = int(os.getenv("TARGETS_SEED", "216")) if seed is None else seed
        self.target_type = target_type
        self.vocab_size = module.vocab_size
        self.device = module.device
        self._targets = self.get()

    @abstractmethod
    def _generate(self) -> list[tuple[float, torch.Tensor]]:
        raise NotImplementedError("Subclass must implement _generate")
    

    def _get_path(self) -> str:
        return CACHE_PATH.format(target_type=self.target_type, vocab_size=self.vocab_size, seed=self.seed)


    def get(self) -> list[tuple[float, torch.Tensor]]:

        cache_path = self._get_path()

        if Path(cache_path).exists():
            loaded = torch.load(cache_path, map_location=self.device)
            tqdm.write(f"loaded {len(loaded['targets'])} targets from {cache_path}")

            if loaded['seed'] != self.seed:
                raise ValueError(f"cached targets have seed {loaded['seed']}, but requested {self.seed}")
            if loaded['vocab_size'] != self.vocab_size:
                raise ValueError(f"cached targets have vocab size {loaded['vocab_size']}, but requested {self.vocab_size}")
            if loaded['targets'][0][1].shape[0] != self.vocab_size:
                raise ValueError(f"cached target has shape {loaded['targets'][0][1].shape} mismatching saved vocab size {self.vocab_size}")
            loaded_target_type = loaded.get('target_type', loaded.get('type'))
            if loaded_target_type != self.target_type:
                raise ValueError(f"cached targets have type {loaded_target_type}, but requested {self.target_type}")
            
            return loaded['targets']
        
        set_seed(self.seed)
        generated = self._generate()
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"targets": generated, "seed": self.seed, "target_type": self.target_type, "vocab_size": self.vocab_size}, cache_path)
        tqdm.write(f"saved {len(generated)} targets to {cache_path}")
        return generated
    
    def __len__(self) -> int:
        return len(self._targets)
    
    def __getitem__(self, index: int) -> tuple[float, torch.Tensor]:
        return self._targets[index]
    
    def __iter__(self) -> Iterator[tuple[float, torch.Tensor]]:
        return iter(self._targets)



class OptimEntropyGrid(Targets):
    """
    Generates targets by gradient based optimization for a grid of target entropies. 
    """
    def __init__(self, module: DownstreamModule, is_outlier: bool = False, seed: int | None = None) -> None:
        self.epsilon = 1e-4
        self.max_iters = 1e5
        self.step = 0.05
        self.is_outlier = is_outlier
        self.entropies = list(get_range(0, log(module.vocab_size), 0.05))
        target_type = "optimentropygrid-outlier" if is_outlier else "optimentropygrid-vanilla"
        target_type += f"-st{self.step}"
        super().__init__(target_type, module, seed)
    
    def _optimize_vanilla(self, target_entropy: float) -> torch.Tensor:
        param = torch.nn.Parameter(torch.randn(self.vocab_size, device=self.device, dtype=torch.float32))
        optimizer = torch.optim.Adam([param], lr=0.1)
        for i in range(int(self.max_iters)):
            optimizer.zero_grad()
            loss = L(param, target_entropy)
            loss.backward()
            optimizer.step()
            actual_entropy = entropy(torch.softmax(param, dim=-1), dim=-1).item()
            err = abs(actual_entropy - target_entropy)
            if err <= self.epsilon:
                break
        else:
            raise RuntimeError(f"Optimization failed to converge within {self.max_iters} iterations.")
        return param
    

    def _optimize_outlier(self, entropy: float) -> torch.Tensor:
        raise NotImplementedError("Sorry! Outlier distributions might be implemented in the future!")


    def _generate(self) -> list[tuple[float, torch.Tensor]]:
        targets = []
        for i, H in enumerate(tqdm(self.entropies, total=len(self.entropies), desc="Generating targets")):
            set_seed(self.seed + i)
            if self.is_outlier:
                target = self._optimize_outlier(H)
            else:
                target = self._optimize_vanilla(H)
            targets.append((H, target))
        return targets


class GaussianLogits(Targets):
    """
    Generates targets by sampling from a Gaussian distribution.
    """
    def __init__(self, module: DownstreamModule, samples: int, scale: float = 1.0, seed: int | None = None) -> None:
        self.samples = samples
        self.scale = scale
        target_type = f"gaussianlogits-sc{scale:.2f}-n{samples}"
        super().__init__(target_type, module, seed)
    
    def _generate(self) -> list[tuple[float, torch.Tensor]]:
        targets = []
        for i in range(self.samples):
            set_seed(self.seed + i)
            target_logits = self.scale * torch.randn(self.vocab_size, device=self.device, dtype=torch.float32)
            H = entropy(torch.softmax(target_logits, dim=-1), dim=-1).item()
            targets.append((H, target_logits))
        return targets


class ReachableEntropyGrid(Targets):
    """
    Generates targets by end-to-end optimization on LLM for easy-to-reach distributions
    taking a particular set of fixed entropy values. 
    """

    def __init__(self, module: DownstreamModule, seed: int | None = None) -> None:
        self.entropies = list(get_range(0, log(module.vocab_size), 0.05))
        self.module = module
        super().__init__("reachableentropygrid", module, seed)

    def _generate(self) -> list[tuple[float, torch.Tensor]]:
        targets = []
        for i, H in tqdm(enumerate(self.entropies), total=len(self.entropies), desc="Tuning targets"):
            set_seed(self.seed + i)
            best_prompt, best_loss, early_stopped = tune_soft_prompt(self.module, H, None, loss_fn=entropy_only_loss, lr=0.1, max_epochs=500, early_stop_patience=20, log_losses=False)
            target_log_probs = self.module.forward(best_prompt).detach()
            actual_H = log_entropy(target_log_probs, dim=-1).item()
            targets.append((actual_H, target_log_probs))
        return targets