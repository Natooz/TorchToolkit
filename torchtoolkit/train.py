from logging import Logger
from pathlib import Path
from typing import List, Callable
from contextlib import contextmanager
from functools import partial

from torch import Tensor, device as device_, cuda, autocast, mean, no_grad, save
from torch.nn.modules import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import Metric, calculate_accuracy


def select_device(use_cuda: bool = True, log: bool = False) -> device_:
    r"""Select the device on which PyTorch will run

    :param use_cuda: specify if you want to run it on the GPU if available. (default: True)
    :param log: will log a warning message if a CUDA device is detected but not used. (default: False)
    :return: 'cpu' or 'cuda:0' device object.
    """
    if cuda.is_available():
        if use_cuda:
            return device_('cuda:0')
        elif log:
            print('WARNING: You have a CUDA device, you should probably run with it')
    return device_('cpu')


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


@contextmanager
def __null_context():
    yield


class ValidAccModeParameters:
    def __init__(self, valid_acc_target: float, nb_past_steps: int, min_nb_steps: int = 0,
                 max_nb_steps: int = float('inf')):
        """

        :param valid_acc_target: validation accuracy target
        :param nb_past_steps: the number of past valid accuracy values to use to compute the average validation
                        accuracy. If this average is > to the minimum valid acc to reach, the training can be stopped
                        if the number of steps is > to min_nb_steps given above.
        :param min_nb_steps: min number of training steps when working with min_valid_acc (default: 0)
        :param max_nb_steps: max number of training steps when working with min_valid_acc (default: +inf)
        """
        self.valid_acc_target = valid_acc_target
        self.nb_past_steps = nb_past_steps
        self.min_nb_steps = min_nb_steps
        self.max_nb_steps = max_nb_steps


class Iterator:
    def __init__(self, nb_steps: int = None, early_stop_steps: int = float('inf'),
                 valid_acc_mode_parameters: ValidAccModeParameters = None, pbar_desc: str = 'TRAINING'):
        """Training iterator class.
        Can work in two modes:
            1. Number of steps: will be iterated a fixed number of times
            2. Valid accuracy: will be iterated till the model reaches a target validation
                accuracy value, or if the number of training steps exceeds max_nb_steps.

        :param nb_steps: number of training steps. (default None)
        :param valid_acc_mode_parameters: parameters to use to train a model in "validation accuracy mode".
                        (default: None)
        :param early_stop_steps: will stop training if the validation accuracy did not increase in this last
                                number of training steps (default: inf)
        :param pbar_desc: progress bar description. (default: TRAINING)
        """
        assert nb_steps is not None or valid_acc_mode_parameters is not None, \
            'You must give at least nb_steps or min_valid_acc argument to construct the iterator'
        self.nb_steps = nb_steps
        self.valid_acc_params = valid_acc_mode_parameters
        self._past_valid_acc = []
        self.best_valid_step = 0  # for early stop
        self._early_stop_steps = early_stop_steps
        self.pbar = tqdm(total=nb_steps if self.valid_acc_params is None else self.valid_acc_params.max_nb_steps,
                         desc=pbar_desc)

    def __iter__(self):
        self.step = 0
        return self

    def __next__(self):
        # Early stop if valid acc did not increase
        if self._early_stop_steps is not None and self.step - self.best_valid_step >= self._early_stop_steps:
            raise StopIteration

        # Validation target mode
        elif self.valid_acc_params is not None:
            if self.__is_past_valid_acc_ok() or \
                    self.valid_acc_params.min_nb_steps < self.step < self.valid_acc_params.max_nb_steps:
                return self.__iter_update()
            raise StopIteration

        elif self.step < self.nb_steps:
            return self.__iter_update()

        raise StopIteration

    def __iter_update(self):
        self.step += 1
        self.pbar.update(1)
        return self.step

    def __is_past_valid_acc_ok(self) -> bool:
        if len(self._past_valid_acc) < self.valid_acc_params.nb_past_steps:
            return False
        return self.__mean_last_valid_acc(self.valid_acc_params.nb_past_steps) >= self.valid_acc_params.valid_acc_target

    def __mean_last_valid_acc(self, nb_val: int) -> float: return sum(self._past_valid_acc[-nb_val:]) / nb_val

    def update_valid_acc(self, valid_acc: float):
        """Stores the validation accuracy given in argument. Need to be called at each validation step in order
        to keep track, and compute the average value of the last steps to stop or keep training.

        :param valid_acc: validation accuracy to store.
        """
        self._past_valid_acc.append(valid_acc)


def train(model: Module, criterion: Module, optimizer: Optimizer, dataloader_train: DataLoader,
          dataloader_valid: DataLoader, nb_steps: int, valid_intvl: int, nb_valid_steps: int,
          tsb: SummaryWriter = None, pbar_desc: str = 'TRAINING', logger: Logger = None, log_intvl: int = 10,
          acc_func: Callable = calculate_accuracy, valid_metrics: List[Metric] = None, iterator_kwargs: dict = None,
          lr_scheduler=None, device: device_ = None, use_amp: bool = True, gradient_clip_norm: float = None,
          saving_dir: Path = None):
    """A generic training function.
    Every valid_intvl steps, it will run nb_valid_steps validation steps during which the model
    will be evaluated on the dataloader_valid data, retrieving the average loss and accuracy values.

    :param model: model Module to train. It musts implement a 'forward_train' method which takes as argument
                the input tensor, the target Tensor and the criterion.
                It returns y (output probabilities), loss (computed with the criterion), y_sampled (y sampled,
                to be used with metrics in validation, you can return None if you don't use metrics).
    :param criterion: the criterion.
    :param optimizer: the optimizer.
    :param dataloader_train: the DataLoader object which loads training samples.
    :param dataloader_valid:  the DataLoader object which loads validation samples.
    :param nb_steps: number of training steps (model updates).
    :param valid_intvl: number of training steps between each validation phase.
    :param nb_valid_steps: number of validation steps to perform per validation phase.
    :param tsb: tensorboard object, to project loss, accuracy and metrics. (default: None)
    :param pbar_desc: description of the tqdm progress bar. (default 'TRAINING')
    :param logger: Logger object, used to log the progress bar messages. If given,
                this method will `debug` the progress bar, at an interval of log_intvl training
                steps (see arg below). If you want to log the progress states in a file
                but without printing them, make sure that the logger has a StreamHandler at
                the INFO level and a FileHandler at DEBUG level. (default: None)
    :param log_intvl: number of training steps between update of the progress bar. (default: 10)
    :param acc_func: accuracy function. (default: torchtoolkit.metrics.calculate_accuracy in greedy mode)
    :param valid_metrics: custom metrics to run during validation phase, torchtoolkit.metrics.Metric. (default: None)
    :param iterator_kwargs: parameters for the training iterator, to be given as a dictionary as:
                - 'valid_acc_mode_parameters': parameters to train the model until it reaches a target
                        validation accuracy value, see ValidAccModeParameters (default: None)
                - 'early_stop_steps': will stop training if the validation accuracy did not increase in
                        this last number of training steps (default: inf)
                (default: None)
    :param lr_scheduler: learning rate scheduler. (default: None)
    :param device: device to run on (default: None --> select_device(use_cuda=True))
    :param use_amp: to use Automatic Mixed Precision (AMP) during training. (default: True)
    :param gradient_clip_norm: norm of gradient clipping. (default: None)
    :param saving_dir: output directory to save the model state_dict. (default: None)
    """
    if saving_dir is not None:
        saving_dir.mkdir(parents=True, exist_ok=True)
    if iterator_kwargs is None:
        iterator_kwargs = {}
    device = device if device is not None else select_device(use_cuda=True)
    valid_metrics = [] if valid_metrics is None else valid_metrics
    model = model.to(device)
    model.train()
    best_valid_loss = float('inf')
    best_valid_loss_step = 0
    last_loss_valid = last_acc_valid = 0  # use for pbar postfix
    train_iter = iter(dataloader_train)
    valid_iter = iter(dataloader_valid)
    amp_context = __null_context if not use_amp else partial(autocast, 'cuda')
    if device.type == 'cuda':
        cuda.empty_cache()  # clears GPU memory, may be required after running several trainings successively

    for training_step in (iterator := Iterator(nb_steps, **iterator_kwargs, pbar_desc=pbar_desc)):
        optimizer.zero_grad()  # Initialise gradients
        try:
            x, target = next(train_iter)  # (N,T)
        except StopIteration:
            train_iter = iter(dataloader_train)
            x, target = next(train_iter)
        with amp_context():
            x, target = x.to(device), target.to(device)
            y, loss, _ = model.forward_train(x, target, criterion)  # (N,T,C)
        acc = acc_func(y, target)
        last_loss_train, last_acc_train = loss.item(), acc

        loss.backward()  # stores / accumulate gradients in the graph
        if gradient_clip_norm is not None:
            clip_grad_norm_(model.parameters(), gradient_clip_norm)
        # torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.step()  # updates the weights

        if tsb is not None:
            tsb.add_scalar('Loss/train', loss.item(), training_step)
            tsb.add_scalar('Accuracy/train', acc, training_step)  # assuming all groups have the same LR
            tsb.add_scalar('Learning rate/training', optimizer.param_groups[0]['lr'], training_step)
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Validation
        if training_step % valid_intvl == 0:
            model.eval()
            valid_loss, valid_acc = [], []
            for valid_step in range(nb_valid_steps):
                try:
                    x, target = next(valid_iter)  # (N,T)
                except StopIteration:
                    valid_iter = iter(dataloader_valid)
                    x, target = next(valid_iter)
                with no_grad():
                    x, target = x.to(device), target.to(device)
                    y, loss, y_sampled = model.forward_train(x, target, criterion)  # (N,C,T)
                    valid_loss.append(loss.item())
                    valid_acc.append(acc_func(y, target))

                for m, metric in enumerate(valid_metrics):
                    metric(x, y_sampled)
            valid_loss = mean(Tensor(valid_loss))
            valid_acc = mean(Tensor(valid_acc))
            last_loss_valid, last_acc_valid = valid_loss, valid_acc
            iterator.update_valid_acc(float(valid_acc))
            if tsb is not None:
                tsb.add_scalar('Loss/valid', valid_loss, training_step)
                tsb.add_scalar('Accuracy/valid', valid_acc, training_step)
                for m, metric in enumerate(valid_metrics):
                    tsb.add_scalar(f'Metrics/{metric.name}', metric.results[-1], training_step)

            # Save model if loss as decreased
            if saving_dir is not None and valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_valid_loss_step = training_step
                iterator.best_valid_step = training_step
                save({'training_step': training_step,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'train_loss': last_loss_train, 'train_acc': last_acc_train,
                      'valid_loss': last_loss_valid, 'valid_acc': last_acc_valid},
                     saving_dir / 'checkpoint.pt.tar')
            model.train()

        iterator.pbar.set_postfix({'train_loss': f'{last_loss_train:.4f}', 'train_acc': f'{last_acc_train:.4f}',
                                   'valid_loss': f'{last_loss_valid:.4f}', 'valid_acc': f'{last_acc_valid:.4f}',
                                   'best_valid_acc_step': best_valid_loss_step},
                                  refresh=False)
        if logger is not None and training_step % log_intvl == 0:
            logger.debug(str(iterator.pbar).encode('utf-8'))

    model.eval()
