from typing import List, Union
from pathlib import Path, PurePath
import json

from torch import LongTensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2LMHeadModel, GPT2Config
from tqdm import tqdm

from torchtoolkit.train import train, log_cuda_info, log_model_parameters
from torchtoolkit.data import create_subsets, collate_ar
from torchtoolkit.sampling import top_k


class MIDIDataset(Dataset):
    r"""Dataset class. Loads json files of tokenized MIDIs and prepare them for training.

    :param data_path: path containing the data to load, ex: 'data/death_metal_dataset'.
    :param max_seq_len: maximum sequence length (in nb of tokens).
    :param min_seq_len: minimum sequence length.
    :param padding_token: padding token, usually 0.
    """

    def __init__(self, data_path: Union[List[Union[Path, PurePath, str]], Union[Path, PurePath, str]],
                 max_seq_len: int, min_seq_len: int, padding_token: int):
        self.samples = []
        files_paths = data_path if isinstance(data_path, list) else list(Path(data_path).glob('**/*.json'))
        import os
        print(os.getcwd())
        print(os.path.realpath(__file__))
        print(data_path)
        print(files_paths)

        for file_path in tqdm(files_paths, desc='Preparing data'):
            with open(file_path) as json_file:
                try:  # in case of error of conversion, no track / empty json file
                    tokens = json.load(json_file)['tokens'][0]  # first track
                except IndexError:
                    continue

            i = 0
            while i < len(tokens):
                if i >= len(tokens) - min_seq_len:
                    break  # last sample is too short
                spl = tokens[i:i + max_seq_len]
                self.samples.append(LongTensor(spl))
                i += max_seq_len
        if len(self.samples) > 0:
            self.samples = pad_sequence(self.samples, batch_first=True, padding_value=padding_token)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class MyModel(GPT2LMHeadModel):

    def forward_train(self, x: LongTensor, target: LongTensor, criterion: Module, k: int = 5):
        # x and target: (N,T)
        logits = self.forward(x).logits  # (N,T,C) x as source and target here to mock
        loss = criterion(logits.transpose(2, 1), target)  # (N,C,T) and (N,T)
        y_sampled = top_k(logits.transpose(1, 0), k)  # (N,T)
        return logits, loss, y_sampled


def test_training():
    config = GPT2Config(vocab_size=300, n_positions=128, n_embd=48, n_layer=2, n_head=4, n_inner=96)
    model = MyModel(config)
    tensorboard = SummaryWriter()
    criterion = CrossEntropyLoss(ignore_index=0)
    optimizer = Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-2)
    lr_scheduler = OneCycleLR(optimizer, max_lr=1e-4, total_steps=100, pct_start=0.5)
    log_cuda_info()
    log_model_parameters(model)

    dataset = MIDIDataset(Path('test', 'test_files'), max_seq_len=128, min_seq_len=64, padding_token=0)
    subset_train, subset_valid = create_subsets(dataset, [0.4])
    dataloader_train = DataLoader(subset_train, batch_size=8, collate_fn=collate_ar)
    dataloader_valid = DataLoader(subset_valid, batch_size=8, collate_fn=collate_ar)

    train(model, criterion, optimizer, dataloader_train, dataloader_valid, nb_steps=50, valid_intvl=10,
          nb_valid_steps=5, log_intvl=10, tsb=tensorboard, lr_scheduler=lr_scheduler)
    assert True  # test passed with no error


if __name__ == '__main__':
    test_training()
