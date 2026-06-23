from abc import abstractmethod, ABC
import torch
from tqdm import tqdm
from pathlib import Path
from .utils import set_seed, find_device, entropy
from math import log


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

    def __init__(self, target_type: str, vocab_size: int, seed: int) -> None:
        self.seed = seed
        self.target_type = target_type
        self.vocab_size = vocab_size
        self.device = find_device()

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



class OptimEntropyGrid(Targets):
    def __init__(self, vocab_size: int, seed: int, is_outlier: bool = False) -> None:
        self.epsilon = 1e-4
        self.max_iters = 1e5
        self.is_outlier = is_outlier
        self.entropies = list(get_range(0, log(vocab_size), 0.05))
        target_type = "optimentropygrid_outlier" if is_outlier else "optimentropygrid_vanilla"
        super().__init__(target_type, vocab_size, seed)
    
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
        for i, H in enumerate(tqdm(self.entropies, desc="Generating targets")):
            set_seed(self.seed + i)
            if self.is_outlier:
                target = self._optimize_outlier(H)
            else:
                target = self._optimize_vanilla(H)
            targets.append((H, target))
        return targets


