import torch

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def entropy(x: torch.Tensor, dim: int) -> torch.Tensor:
    return -torch.sum(x * torch.log(x), dim=dim)