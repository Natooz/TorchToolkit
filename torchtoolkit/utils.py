from typing import List, Tuple
from random import uniform

from torch import cuda, Tensor, randint, full, arange, manual_seed


def seed_everything(seed: int):
    r"""Set the seed for pytorch, random (and numpy if available) modules for reproducibility.

    :param seed: seed integer
    """
    from random import seed as rdm_seed
    import os
    from numpy.random import seed as np_seed

    rdm_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    manual_seed(seed)
    cuda.manual_seed_all(seed)
    np_seed(seed)


def mask_tensor(x: Tensor, mask_token: int, masking_ratio: float):
    r"""Mask a tensor, i.e. randomly replace tokens with a special masking token. It also allows to
    randomize a ratio of tokens.

    :param x: tensor to mask, of any shape, the last dim will be randomized.
    :param mask_token: the mask token
    :param masking_ratio: ratio of tokens to mask, has to be comprised within [O,1]
    :return: the masked tensor
    """
    mask_shape = (*x.shape[:-1], int(x.shape[-1] * masking_ratio))
    mask = full(mask_shape, mask_token)
    indices_to_mask = randint(1, x.shape[-1] - 1, mask_shape).to(x.device)
    return x.clone().scatter_(-1, indices_to_mask, mask)


def randomize_tensor(x: Tensor, random_range: Tuple[int, int], random_ratio: float = None,
                     random_ratio_range: Tuple[float, float] = (0.1, 1)):
    r"""Randomize a token sequence. It replaces randomly selected tokens (accordingly to random_ratio
    and random_ratio_range) by random values within random_range.

    :param x: tensor to randomize, of any shape, the last dim will be randomized.
    :param random_range: token range of random tokens
    :param random_ratio: ratio of randomized tokens (within the masking ratio), has to be comprised within [O,1]
    :param random_ratio_range: if random_ratio is None, a random ratio will randomly be chosen between. You
                            can set a range for this ratio, between 0 and 1 (default: (0, 1))
    :return: the randomized tensor
    """
    if random_ratio is None:  # randomly pick a ratio from the ratio range
        random_ratio = uniform(*random_ratio_range)
    mask_shape = (*x.shape[:-1], int(x.shape[-1] * random_ratio))
    indices_to_randomize = randint(0, x.shape[0], mask_shape).to(x.device)
    random_values = randint(random_range[0], random_range[1], mask_shape)
    return x.clone().scatter_(-1, indices_to_randomize, random_values)


def convert_idx_tensor(idx: Tensor) -> List[Tensor]:
    """Convert a tensor of indices into a list of tensors specifying the dim indices.
    idx is of any shape (*), and specifies the indices of another tensor x of the same shape + an extra dimension (*,T).
    idx specifies one element along the last dimension of x, this method will return a list of tensors of shape:
    [(prod(idx.shape))] * (idx.dim() + 1)
    It is useful if you want to index tensors.
    Examples: idx shape --> output shape
        (2,2) --> [(4), (4), (4)]
            [[0, 1], [2, 3]] --> [(0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 2, 3)]

        (2,3) --> [(6), (6), (6)]
            [[0, 1, 2], [3, 4, 5]] --> [(0, 0, 0, 1, 1, 1), (0, 1, 2, 0, 1, 2), (0, 1, 2, 3, 4, 5)]

        (2,2,2) --> [(8), (8), (8), (8)]
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
            --> [(0, 0, 0, 0, 1, 1, 1, 1),
                (0, 0, 1, 1, 0, 0, 1, 1),
                (0, 1, 0, 1, 0, 1, 0, 1),
                (0, 1, 2, 3, 4, 5, 6, 7)]

        (3,4,2) --> [(24), (24), (24), (24)]
            [[[0, 2], [1, 3], [1, 2], [4, 3]],
            [[0, 2], [1, 3], [1, 2], [4, 3]],
            [[0, 2], [1, 3], [1, 2], [4, 3]]]
            --> [(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2),
                (0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3),
                (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
                ()]

    :param idx: tensor of indices, of shape (*), specifying indexes of another tensor of shape (*,T),
            with T >= max(idx).
    :return: list of tensors of shape [(prod(idx.shape))] * (idx.dim() + 1), ready for indexing.
    """
    numel = idx.shape.numel()  # Total number of elements to index
    return [arange(0, idx.shape[d]).repeat_interleave(nb_rep_int := idx.shape[d+1:].numel()).
            repeat(numel // (idx.shape[d] * nb_rep_int)) for d in range(idx.dim())] + [idx.flatten()]
