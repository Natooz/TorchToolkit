from typing import Union
from math import prod

import torch


def calculate_accuracy(result: torch.Tensor, expected: torch.Tensor, mode: str = 'greedy',
                       top_kp: Union[int, float] = None, temperature: Union[int, float] = None) -> float:
    r"""Computes the accuracy of a result according to its expected target

    :param result: result to assess, shape (*,C), or (*) for likelihood
    :param expected: expected result, shape (*)
    :param mode: accuracy computation method:
                greedy: (default) will consider only the maximum logit (argmax) for each prediction
                top_k: will count a good prediction if the target index is in the top_kp logits
                top_p: will count a good prediction if the target index is in the top_kp logits
                softmax: apply softmax over the results and will consider the probability of the expected index / token
                likelihood: computes the average distance between results and expected: 1 - mean(abs(result - expected)

    :param top_kp: value to use with top_k or top_p modes
    :param temperature: temperature for softmax
    :return:
    """
    nb_dims = expected.dim()
    shape = expected.shape
    nb_of_logits = prod(shape)

    if mode == 'top_k':
        assert nb_dims in [1, 2], 'Top-k accuracy only works with target tensors of 1 or 2 dimensions'
        top_indices = torch.topk(result, top_kp).indices
        exp_in_res = 0
        for n in range(shape[0]):
            if nb_dims == 2:
                for t in range(shape[1]):
                    if expected[n, t] in top_indices[n, t]:
                        exp_in_res += 1
            else:  # 1 dim
                if expected[n] in top_indices[n]:
                    exp_in_res += 1
        return exp_in_res / nb_of_logits

    elif mode == 'top_p':
        assert nb_dims in [1, 2], 'Top-p accuracy only works with target tensors of 1 or 2 dimensions'
        res = result.clone() / temperature if temperature is not None else result.clone()
        res_sorted, sorted_indices = torch.sort(res, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(res_sorted, dim=-1), dim=-1)
        res_indices = cumulative_probs < top_kp
        # Shift the indices to the right to keep also the first token above the threshold
        res_indices[..., 1:] = res_indices[..., :-1].clone()
        exp_in_res = 0
        for n in range(shape[0]):
            if nb_dims == 2:
                for t in range(shape[1]):
                    if expected[n, t] in sorted_indices[n, t, res_indices[n, t]]:
                        exp_in_res += 1
            else:  # 1 dim
                indices = sorted_indices[res_indices]
                if expected[n] in indices[n]:
                    exp_in_res += 1
        return exp_in_res / nb_of_logits

    elif mode == 'softmax':
        res = result.clone() / temperature if temperature is not None else result.clone()
        res = torch.softmax(res, dim=-1)
        su = 0
        for n in range(shape[0]):
            if nb_dims == 2:
                for t in range(shape[1]):
                    su += res[n, t, expected[n, t]]
            else:  # 1 dim
                su += res[n, expected[n]]
        return su / nb_of_logits
    elif mode == 'likelihood':
        return 1 - torch.mean(torch.abs(result - expected))

    return (torch.argmax(result, dim=-1) == expected).sum().item() / nb_of_logits  # greedy, default
