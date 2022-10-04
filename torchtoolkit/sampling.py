from typing import Tuple, List, Union

import torch


def top_k(x: torch.Tensor, k: int, temperature: float = None, nb_samples: int = 1) -> torch.Tensor:
    r"""Top K sampling

    :param x: input tensor of shape (N,C) or (T,N,C)
    :param k: k factor
    :param temperature: temperature for softmax. (default: None)
    :param nb_samples: number of samples to draw. (default: 1)
    :return: sampling results as (N) or (T,N)
    """
    assert nb_samples < k, 'nb_samples is >= to k, it should be inferior to k to draw enough samples.' \
                           'If you want nb_samples == k, you might as well use the torch.top_k method.'
    x_copy = x.clone() / temperature if temperature is not None else x.clone()
    indices_to_inf = x < torch.topk(x, k)[0][..., -1, None]
    x_copy[indices_to_inf] = float('-inf')
    if x.dim() == 2:  # (N,C) --> (N)
        return torch.multinomial(torch.softmax(x_copy, -1), nb_samples).squeeze(-1)
    elif x.dim() == 3:  # (T,N,C) --> (T,N)
        return torch.stack([torch.multinomial(torch.softmax(xi, -1), nb_samples).squeeze(-1) for xi in x_copy])


def nucleus(x: torch.Tensor, p: float, temperature: float = None, nb_samples: int = 1) -> torch.Tensor:
    r"""Nucleus sampling (top p)

    :param x: input tensor of shape (C), (N,C)
    :param p: top-p value
    :param temperature: temperature for softmax. (default: None)
    :param nb_samples: number of samples to draw. (default: 1)
    :return: sampling results as scalar tensor or (N)
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

    return torch.multinomial(torch.softmax(x_copy, -1), nb_samples).squeeze(-1)


def beam_search(logits: torch.Tensor, beam_probs: List[float], x: torch.Tensor = None, nb_beams: int = None,
                apply_softmax: bool = True, return_beam_probs: bool = False) \
        -> Union[Tuple[Union[torch.Tensor, List[Tuple[int, int]]], List[float]], torch.Tensor, List[Tuple[int, int]]]:
    r"""Beam search sampling / decoding method
    Returns either the indices as a list of tuples (beam_idx, token), or the actual beams (x)
    with the next token appended if x is given.
    beam_probs stores the cumulative probabilities of the all tokens of the current beams (x),
    and will be updated inplace with the new beams.
    The second dimension of (N) of the logits represents the beams, its length is the number of beams
    if nb_beams is None. Else nb_beams will update the number of beams within beam_probs (and x). It can
    be used to dynamically change the number of beams.
    Example:

    beam_probs = [1]  # init for the first step, here only 1 beam, will be updated in place
    for _ in range(nb_steps):
        logits = model(x)  # shape (T,N,C), x has shape (T,N)
        x = beam_search(logits, beam_probs, x, nb_beams=8)  # append new tokens for each beam, shape (T+1,N)

    :param logits: logit tensor of shape (T,N,C), T is sequence length, N beams (used as batch size), C vocab size
    :param beam_probs: list of cumulative probabilities of each beam (N)
    :param x: actual beams, as tokens (T,N). Give this argument to update x with the new beams
            and appended new tokens. If given, the function will return x updated. (default: None)
    :param nb_beams: number of beams, to be used to set the number of output beams. Give None to keep
            the same beam size (N) as in the logit tensor (default: None)
    :param apply_softmax: applies softmax on the last dim of the logits (default: True)
    :param return_beam_probs: will return the cumulative probabilities (beam_probs) of the selected beams.
           If given False, the list beam_probs will be updated inplace (default: False)
    :return: indices of the beams to keep and token to append as a list of tuples (beam_idx, token),
            or updated beams (x tensor) of shape (T+1,nb_beams), and updated beam_probs if return_beam_probs
    """
    beam_dim = nb_beams if nb_beams is not None else logits.shape[1]  # = N
    vocab_size = logits.shape[2]
    if apply_softmax:
        logits = torch.softmax(logits, -1)

    # Computes cumulative probs and pick the top ones as new beams
    cum_probs = [beam_prob + log for n, beam_prob in enumerate(beam_probs) for log in logits[-1, n].tolist()]
    indices = torch.topk(torch.Tensor(cum_probs), beam_dim).indices.tolist()  # (N)
    real_indices = [(idx // vocab_size, idx % vocab_size) for idx in indices]  # as tuples (beam_idx, token)

    # Updates beam probs list inplace
    for i, (n, l) in enumerate(real_indices):
        if i < len(beam_probs):
            beam_probs[n] = cum_probs[l]
        else:  # in case the number of beams got increased
            beam_probs.append(cum_probs[l])
    while len(beam_probs) > beam_dim:  # deletes extra beams if the number got reduced
        del beam_probs[-1]

    if x is not None:  # no inplace operation possible here to update x
        x = torch.stack([torch.cat([x[:, n], torch.Tensor([l])]) for n, l in real_indices]).t()
        return (x, beam_probs) if return_beam_probs else x
    return (real_indices, beam_probs) if return_beam_probs else real_indices
