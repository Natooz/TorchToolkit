from logging import Logger

from torch import device, cuda
from torch.backends.mps import is_available as mps_available
from torch.nn.modules import Module


def select_device(use_cuda: bool = True, use_mps: bool = True, log: bool = False) -> device:
    r"""Select the device on which PyTorch will run

    :param use_cuda: specify if you want to run it on the GPU if available. (default: True)
    :param use_mps: will run on MPS device if available. (default: True)
    :param log: will log a warning message if a CUDA device is detected but not used. (default: False)
    :return: 'cpu' or 'cuda:0' device object.
    """
    if cuda.is_available():
        if use_cuda:
            return device('cuda:0')
        elif log:
            print('WARNING: You have a CUDA device, you should probably run with it')
    if mps_available():
        if use_mps:
            return device("mps")
        elif log:
            print("WARNING: You have a MPS device, you should probably run with it")
    return device('cpu')


def log_cuda_info(logger: Logger = None, memory_only: bool = False):
    r"""Log the info of GPU

    :param logger: a Logger object. By default, this method will log at the INFO level.
            If no Logger is given, `print` (stdout) will be used. (default: None)
    :param memory_only: choose to log only the memory state of GPU. (default: False)
    """
    log_func = logger.info if logger is not None else print
    if cuda.is_available():
        if not memory_only:
            log_func('******** GPU INFO ********')
            log_func(f'GPU: {cuda.get_device_name(0)}')
        log_func(f'Total memory: {round(cuda.torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)}GB')
        log_func(f'Cached memory: {round(cuda.memory_reserved(0) / 1024 ** 3, 1)}GB')
        log_func(f'Allocated memory: {round(cuda.memory_allocated(0) / 1024 ** 3, 1)}GB')
    else:
        log_func('No cuda device detected')


def log_model_parameters(model: Module, logger: Logger = None, model_desc: bool = True):
    r"""Log the number of parameters of a model

    :param model: model to analyze.
    :param logger: a Logger object. By default, this method will log at the INFO level.
            If no Logger is given, `print` (stdout) will be used. (default: None)
    :param model_desc: also logs the description of the model, i.e. the modules. (default: True)
    """
    log_func = logger.info if logger is not None else print
    if not model_desc:
        log_func('******** MODEL INFO ********')
        log_func(model)
    log_func(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    log_func(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
