import torch

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def entropy(x: torch.Tensor, dim: int) -> torch.Tensor:
    return -torch.sum(x * torch.log(x), dim=dim)

def L(logits: torch.Tensor, targ_entropy: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    actual_entropy = entropy(probs, dim=-1)
    return (actual_entropy - targ_entropy) ** 2

def L_prime(alpha: torch.nn.Parameter, beta: torch.nn.Parameter) -> torch.Tensor:
    raise NotImplementedError("Not implemented")


def optimize_vanilla(targ_entropy: float, dist_size: int, epsilon: float, max_iters: int = 1e5) -> torch.nn.Parameter:
    param = torch.nn.Parameter(torch.randn(dist_size))
    optimizer = torch.optim.Adam([param], lr=0.1)
    for i in range(int(max_iters)):
        optimizer.zero_grad()
        loss = L(param, targ_entropy)
        loss.backward()
        optimizer.step()
        if abs(entropy(torch.softmax(param, dim=-1), dim=-1) - targ_entropy) <= epsilon:
            break
    else:  # did not know you could do this until very recently
        raise RuntimeError(f"Optimization failed to converge within {max_iters} iterations.")
    return param


if __name__ == "__main__":
    set_seed(42)
    param = optimize_vanilla(targ_entropy=1.0, dist_size=10, epsilon=1e-6)
    print(param)
    print(entropy(torch.softmax(param, dim=-1), dim=-1))

