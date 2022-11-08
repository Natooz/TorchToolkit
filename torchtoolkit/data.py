from typing import List, Tuple

from torch import Tensor, stack
from torch.utils.data import Dataset, Subset, random_split


def create_subsets(dataset: Dataset, split_ratio: List[float]) -> List[Subset]:
    r"""Create subsets of a dataset following split ratios.
    if sum(split_ratio) != 1, the remaining portion will be inserted as the first subset

    :param dataset: Dataset object, must implement the __len__ magic method.
    :param split_ratio: split ratios as a list of float
    :return: the list of subsets
    """
    assert all(0 <= ratio <= 1. for ratio in split_ratio), 'The split ratios must be comprise within [0,1]'
    assert sum(split_ratio) <= 1., 'The sum of split ratios must be inferior or equal to 1'
    len_subsets = [int(len(dataset) * ratio) for ratio in split_ratio]
    if sum(split_ratio) != 1.:
        len_subsets.insert(0, len(dataset) - sum(len_subsets))
    subsets = random_split(dataset, len_subsets)
    return subsets


def collate_ar(batch: List[Tensor]) -> Tuple[Tensor, Tensor]:
    r"""A collate function for PyTorch DataLoaders, for auto-regressive tasks.
    This will create an input and expected output (target) sequences, the latter being the former shifted by one.

    :param batch: batch as a list of Tensors
    :return: the input and expected output (target) sequences, both of shape (N,T)
    """
    batch = stack(batch)
    return batch[..., :-1], batch[..., 1:]
