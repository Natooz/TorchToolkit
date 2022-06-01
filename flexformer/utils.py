from torch import device, cuda


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
