from typing import List

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


def mask_tensor(x: Tensor, mask_token: int, masking_ratio: float, random_ratio: float = 0.,
                random_range: tuple = None):
    r"""Mask a tensor, i.e. randomly replace tokens with a special masking token. It also allows to
    randomize a ratio of tokens.

    :param x: tensor to mask
    :param mask_token: the mask token
    :param masking_ratio: ratio of tokens to mask, has to be comprised within [O,1]
    :param random_ratio: ratio of randomized tokens (within the masking ratio), has to be comprised within [O,1]
                        (default: 0.)
    :param random_range: token range of random tokens (default: None)
    :return: the masked tensor
    """
    indices_mask = randint(1, x.shape[1] - 1, (x.shape[0], int(masking_ratio * x.shape[1]))).to(x.device)
    mask = full(indices_mask.shape, mask_token)
    if random_ratio > 0:
        indices_random = randint(0, mask.shape[1], (x.shape[0], int(mask.shape[1] * random_ratio)))
        random_values = randint(*random_range, indices_random.shape)
        mask = mask.scatter_(-1, indices_random, random_values).to(x.device)
    return x.clone().scatter_(-1, indices_mask, mask)


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
