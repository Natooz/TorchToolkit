from logging import Logger
from typing import List

from torch import device, cuda, Tensor, randint, full
from torch.utils.data import Dataset, Subset, random_split


def select_device(use_cuda: bool = True, log: bool = False) -> device:
    """ Select the device on which PyTorch will run

    :param use_cuda: specify if you want to run it on the GPU if available (default: True)
    :param log: will log a warning message if a CUDA device is detected but not used (default: False)
    :return: cpu or cuda:0
    """
    if cuda.is_available():
        if use_cuda:
            return device('cuda:0')
        elif log:
            print('WARNING: You have a CUDA device, you should probably run with it')
    return device('cpu')


def log_cuda_info(dev: device, logger: Logger = None, memory_only: bool = False):
    """ Log the info of GPU
    :param dev: the device object
    :param logger: a logger object, if not given this function will print info (default: None)
    :param memory_only: choose to log only the memory state of GPU (default: False)
    """
    if dev.type == 'cuda':
        log_func = logger.debug if logger is not None else print
        if not memory_only:
            log_func('******** GPU info ********')
            log_func(f'GPU: {cuda.get_device_name(0)}')
        log_func(f'Total memory: {round(cuda.torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)}GB')
        log_func(f'Cached memory: {round(cuda.memory_reserved(0) / 1024 ** 3, 1)}GB')
        log_func(f'Allocated memory: {round(cuda.memory_allocated(0) / 1024 ** 3, 1)}GB')


def create_subsets(dataset: Dataset, split_ratio: List[float]) -> List[Subset]:
    """Create subsets of a dataset following split ratios.
    if sum(split_ratio) != 1, the remaining portion will be inserted as the first subset

    :param dataset: Dataset object, must implement the __len__ magic method.
    :param split_ratio: split ratios as a list of float
    :return: the list of subsets
    """
    assert all(0 <= ratio <= 1. for ratio in split_ratio), f'The split ratios must be comprise within [0,1]'
    assert sum(split_ratio) <= 1., f'The sum of split ratios must be inferior or equal to 1'
    len_subsets = [int(len(dataset) * ratio) for ratio in split_ratio]
    if sum(split_ratio) != 1.:
        len_subsets.insert(0, len(dataset) - sum(len_subsets))
    subsets = random_split(dataset, len_subsets)
    return subsets


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
    return x.scatter_(-1, indices_mask, mask)