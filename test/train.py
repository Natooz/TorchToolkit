
from sys import stdout
from pathlib import Path, PurePath
from typing import Union, List, Any

import miditok
from miditoolkit import MidiFile
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split

from flexformer import Transformer
from flexformer.train import select_device, train


BATCH_SIZE = 4
VALID_SPLIT = 15
TEST_SPLIT = 5
MIDI_PATHS = PurePath('test', 'Maestro_MIDIs')
MAX_SEQ_LEN = 512
MIN_SEQ_LEN = 384
LEARNING_RATE = 3e-4
TRAINING_STEPS = 20000
WEIGHT_DECAY = 0.01
VALID_INTERVAL = 20
NB_VALID_STEPS = 10
LOG_INTERVAL = 10

NB_LAYERS = 4
D_MODEL = 256
NB_HEADS = 4
D_FFWD = 1024
DROPOUT = 0.1


def progress(count: int, total: int, bar_len: int = 30, status: str = '', beginning: str = None, printing: bool = True):
    """ Prints a progress state, can be used in a for loop for example

    :param count: current index of progress
    :param total: total number of calculation
    :param bar_len: length of the bar
    :param status: Short message
    :param beginning: Short beginning (default is 'count / total')
    :param printing: Print the result in stdout
    """
    if beginning is None:
        beginning = f'{count} / {total}'
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 2)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    prog = f'\r{beginning} [{bar}] {percents:.1f}% ...{status}' + ('\n' if count == total else '')
    if printing:
        stdout.write(prog)
        stdout.flush()
    return prog


def mean(list_: List[Any]) -> float: return sum(list_) / len(list_) if len(list_) > 0. else 0.


class MIDIDataset(Dataset):
    """ Normal Transformer Dataset
    This class serves to load converted MIDI files data and prepare them for training.

    :param data_path: path containing the data to load, ex: 'data/death_metal_dataset'
    :param max_seq_len: maximum sequence length (in nb of tokens)
    :param min_seq_len: minimum sequence length
    """

    def __init__(self, data_path: Union[Path, PurePath, str], max_seq_len: int, min_seq_len: int,
                 tokeniz: miditok.MIDITokenizer):
        self.root_path = data_path  # root directory of the dataset
        self.samples = []
        files_paths = list(Path(self.root_path).glob('**/*.mid'))

        for fi, file_path in enumerate(files_paths):
            progress(fi + 1, len(files_paths), bar_len=30, status=str(file_path), printing=True,
                     beginning=f'Preparing data for training {fi + 1} / {len(files_paths)}')
            midi = MidiFile(file_path)
            tokens = tokeniz.midi_to_tokens(midi)[0]

            i = 0
            while i < len(tokens):
                if i >= len(tokens) - min_seq_len:
                    break  # last sample is too short
                sample = tokens[i:i + max_seq_len]
                self.samples.append(torch.LongTensor(sample))
                i += max_seq_len
        self.samples = torch.nn.utils.rnn.pad_sequence(self.samples, batch_first=True, padding_value=0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __str__(self) -> str:
        return 'No data loaded' if len(self) == 0 else f'{len(self.samples)} samples'


if __name__ == '__main__':
    # (0) Init objects and variables
    device = select_device(True)
    if device.type == 'cuda':
        torch.cuda.empty_cache()  # clears GPU memory

    # (1) Prepare data loaders
    tokenizer = miditok.REMI()
    dataset = MIDIDataset(MIDI_PATHS, MAX_SEQ_LEN, MIN_SEQ_LEN, tokenizer)

    len_valid = int(len(dataset) * VALID_SPLIT / 100)
    len_train = len(dataset) - len_valid
    subset_train, subset_valid = random_split(dataset, [len_train, len_valid])

    data_loader_train = DataLoader(subset_train, BATCH_SIZE, shuffle=True)
    data_loader_valid = DataLoader(subset_valid, BATCH_SIZE, shuffle=True, drop_last=True)
    train_iter = iter(data_loader_train)
    valid_iter = iter(data_loader_valid)

    # (2) Creates the model, and logs info before training
    model = Transformer(NB_LAYERS, 0, len(tokenizer), D_MODEL, NB_HEADS, D_FFWD, DROPOUT, MAX_SEQ_LEN, device=device,
                        padding_token=tokenizer['PAD_None'], causal_enc=True)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.vocab['PAD_None'])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # (3) Train
    tensorboard = SummaryWriter(str(Path('test', 'test')))
    train(model, criterion, optimizer, data_loader_train, data_loader_valid, TRAINING_STEPS, VALID_INTERVAL,
          NB_VALID_STEPS, tensorboard, LOG_INTERVAL, 'TEST TRAINING')

    '''def log(time_step: int, loss_, acc_, pre_training: bool = False):
        mode = 'PRE-TRAINING' if pre_training else 'TRAINING' if model.training else 'VALIDATION'
        print(progress(time_step, TRAINING_STEPS, 30, f'Loss {loss_:.4f} | Accuracy {acc_:.4f}',
                       beginning=f'{time.time() - tt0:.2f}sec {mode} {time_step} / {TRAINING_STEPS}'))


    tt0 = time.time()
    model.train()
    for training_step in range(TRAINING_STEPS):
        # Training
        optimizer.zero_grad()  # Initialise gradients
        try:
            x = next(train_iter).to(device)  # (N,T)
        except StopIteration:
            train_iter = iter(data_loader_train)
            x = next(train_iter).to(device)
        key_mask = (x[:, :-1] == tokenizer.vocab['PAD_None']).to(device)  # (N,T)
        y = model(x.t()[:-1], auto_padding_mask=True)  # (T,N,C), below are # (N,C,T) & (N,T)
        loss = criterion(y.permute(1, 2, 0), x[:, 1:])
        acc = (torch.argmax(y.transpose(0, 1), dim=2) == x[:, 1:]).sum().item() / (y.size(0) * y.size(1))

        loss.backward()  # stores / accumulate gradients in the graph
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        # torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.step()  # updates the weights
        if training_step % LOG_INTERVAL == 0:
            log(training_step, loss, acc)

        # Validation
        if training_step % VALID_INTERVAL == 0:
            model.eval()
            valid_losses = []
            valid_accs = []
            for valid_step in range(NB_VALID_STEPS):
                try:
                    x = next(valid_iter).to(device)  # (N,T)
                except StopIteration:
                    valid_iter = iter(data_loader_valid)
                    x = next(valid_iter).to(device)
                with torch.no_grad():
                    y = model(x.t()[:-1], auto_padding_mask=True)  # (T,N,C), below are # (N,C,T) & (N,T)
                    valid_loss = criterion(y.permute(1, 2, 0), x[:, 1:])  # (N,C,T) & (N,T)
                    valid_acc = (torch.argmax(y.transpose(0, 1), dim=2) == x[:, 1:]).sum().item() / (y.size(0) *
                                                                                                     y.size(1))  # (T,N)

                valid_losses.append(valid_loss.item())
                valid_accs.append(valid_acc)
            valid_losses = mean(valid_losses)
            valid_accs = mean(valid_accs)
            log(training_step, valid_losses, valid_accs)
            model.train()'''
