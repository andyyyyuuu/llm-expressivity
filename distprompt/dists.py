import torch
from tqdm import tqdm
from .utils import entropy, set_seed


def L(logits: torch.Tensor, targ_entropy: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    actual_entropy = entropy(probs, dim=-1)
    return (actual_entropy - targ_entropy) ** 2

def L_prime(alpha: torch.nn.Parameter, beta: torch.nn.Parameter) -> torch.Tensor:
    raise NotImplementedError("Not implemented")


def optimize_vanilla(
    targ_entropy: float,
    dist_size: int,
    epsilon: float,
    max_iters: int = 1e5,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.nn.Parameter:
    device = torch.device(device)
    param = torch.nn.Parameter(torch.randn(dist_size, device=device, dtype=dtype))
    optimizer = torch.optim.Adam([param], lr=0.1)
    with tqdm(total=1, bar_format="{desc}", leave=False, dynamic_ncols=True) as status:
        for i in range(int(max_iters)):
            optimizer.zero_grad()
            loss = L(param, targ_entropy)
            loss.backward()
            optimizer.step()
            actual_entropy = entropy(torch.softmax(param, dim=-1), dim=-1).item()
            err = abs(actual_entropy - targ_entropy)
            status.set_description_str(
                f"optimize H*={targ_entropy:.4f}: H={actual_entropy:.4f}, |err|={err:.6e}, ||grad||={param.grad.norm().item():.2e}"
            )
            if err <= epsilon:
                break
        else:
            raise RuntimeError(f"Optimization failed to converge within {max_iters} iterations.")
    return param



if __name__ == "__main__":
    set_seed(216)
    param = optimize_vanilla(targ_entropy=0, dist_size=10, epsilon=1e-6)
    print(param)
    print(entropy(torch.softmax(param, dim=-1), dim=-1))
