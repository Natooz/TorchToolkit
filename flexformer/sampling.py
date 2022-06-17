import torch


def top_k(x: torch.Tensor, k: int, temperature: float = None) -> torch.Tensor:
    r"""Top K sampling

    :param x: input tensor of shape (N,C) or (T,N,C)
    :param k: k factor
    :param temperature: temperature for softmax
    :return: sampling results as (N)
    """
    x_copy = x.clone() / temperature if temperature is not None else x.clone()
    indices_to_inf = x < torch.topk(x, k)[0][..., -1, None]
    x_copy[indices_to_inf] = float('-inf')
    if x.dim() == 2:  # (N,C)
        return torch.multinomial(torch.softmax(x_copy, -1), 1).squeeze(-1)
    elif x.dim() == 3:  # (T,N,C)
        return torch.stack([torch.multinomial(torch.softmax(xi, -1), 1).squeeze(-1) for xi in x_copy])


def nucleus(x: torch.Tensor, p: float, temperature: float = None) -> torch.Tensor:
    r"""Nucleus sampling (top p)

    :param x: input tensor of shape (C), (N,C)
    :param p: p factor
    :param temperature: temperature for softmax
    :return: sampling results as (N)
    """
    if temperature is not None:
        x = x / temperature
    x_sorted, sorted_indices = torch.sort(x, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(x_sorted, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    x_copy = x.clone()
    if x.dim() == 1:
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        x_copy[indices_to_remove] = float('-inf')
    else:
        for i, to_remove in enumerate(sorted_indices_to_remove):
            x_copy[i, sorted_indices[i, to_remove]] = float('-inf')

    return torch.multinomial(torch.softmax(x_copy, -1), 1).squeeze(-1)
