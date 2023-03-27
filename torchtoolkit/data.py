"""
Methods for PyTorch Datasets / Subsets, and DataLoader collators working with Hugging Face models.
"""


from typing import Any, Dict, List, Tuple, Union

import torch
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


class DataCollatorClassification:
    def __init__(self, pad_token: int):
        """Collator for classification.
        Input_ids will be padded with the pad token given.

        :param pad_token: pas token
        """
        self.pad_token = pad_token

    def __call__(self, examples: List[Dict[str, Any]], return_tensors=None) -> Dict[str, torch.LongTensor]:
        x, y = _collate_cla(examples, self.pad_token)
        attention_mask = (x != self.pad_token).int()
        return {"input_ids": x, "labels": y, "attention_mask": attention_mask}


class DataCollatorContrastive:
    def __init__(self, pad_token: int):
        """Collator for contrastive learning.
        The batch will be passed twice through the model.
        The labels are ranks (arange()).

        :param pad_token: pas token
        """
        self.pad_token = pad_token

    def __call__(self, batch: List[Dict[str, Any]], return_tensors=None) -> Dict[str, torch.LongTensor]:
        x = _pad_batch(batch, self.pad_token)  # (N,T)
        attention_mask = (x != self.pad_token).int()
        return {"input_ids": x, "labels": torch.arange(x.size(0)).long(), "attention_mask": attention_mask}  # rank


class DataCollatorBasic:
    def __init__(
            self,
            pad_token: int,
            bos_token: int = None,
            eos_token: int = None,
            pad_on_left: bool = False,
            shift_labels: bool = False
    ):
        """Multifunction data collator, that can pad the sequences (right or left), add BOS and EOS tokens.
        Input_ids will be padded with the pad token given, while labels will be padded with -100.

        :param pad_token: PAD token
        :param bos_token: BOS token
        :param eos_token: EOS token
        :param pad_on_left: will pad sequence on the left (default: False).
        :param shift_labels: will shift inputs and labels for autoregressive training / teacher forcing.
        """
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_on_left = pad_on_left
        self.shift_labels = shift_labels

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.LongTensor]:
        _add_bos_eos_tokens_to_batch(batch, bos_tok=self.bos_token, eos_tok=self.eos_token)
        pad_on_left = batch[0]["pad_on_left"] if "pad_on_left" in batch[0] else self.pad_on_left
        x, y = _pad_batch(batch, self.pad_token, pad_on_left), _pad_batch(batch, -100, pad_on_left)
        # attention mask handled in model
        if self.shift_labels:  # otherwise it's handled in models such as GPT2LMHead
            x = x[:-1]
            y = y[1:]

        return {"input_ids": x, "labels": y}


def _add_bos_eos_tokens_to_batch(
        batch: List[Dict[str, torch.LongTensor]],
        dict_key: str = "input_ids",
        bos_tok: int = None,
        eos_tok: int = None
):
    if bos_tok is None and eos_tok is None:
        return

    for i in range(len(batch)):
        if bos_tok is not None and eos_tok is not None:
            batch[i][dict_key] = torch.cat([torch.LongTensor([bos_tok]),
                                            batch[i][dict_key],
                                            torch.LongTensor([eos_tok])]).long()
        elif bos_tok is not None:
            batch[i][dict_key] = torch.cat([torch.LongTensor([bos_tok]), batch[i][dict_key]]).long()
        else:  # EOS not None
            batch[i][dict_key] = torch.cat([batch[i][dict_key], torch.LongTensor([eos_tok])]).long()


def _pad_batch(
        batch: List[Dict[str, torch.LongTensor]],
        pad_token: int,
        dict_key: str = "input_ids",
        pad_on_left: bool = False
) -> torch.LongTensor:
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""

    length_of_first = batch[0][dict_key].size(0)

    # Check if padding is necessary.
    are_tensors_same_length = all(x[dict_key].size(0) == length_of_first for x in batch)
    if are_tensors_same_length:
        return torch.stack([e[dict_key] for e in batch], dim=0).long()

    # Creating the full tensor and filling it with our data.
    if pad_on_left:
        return pad_left([e[dict_key] for e in batch], pad_token)
    else:
        return torch.nn.utils.rnn.pad_sequence(
            [e[dict_key] for e in batch],
            batch_first=True,
            padding_value=pad_token
        ).long()


def pad_left(batch: List[torch.LongTensor], pad_token: int) -> torch.LongTensor:
    # Here the sequences are padded to the left, so that the last token along the time dimension
    # is always the last token of each seq, allowing to efficiently generate by batch
    batch = [torch.flip(seq, dims=(0,)) for seq in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_token)  # (N,T)
    batch = torch.flip(batch, dims=(1,)).long()
    return batch


def _collate_cla(
        batch: List[Dict[str, Union[torch.LongTensor, int]]],
        pad_tok: int
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    x = _pad_batch(batch, pad_tok)
    y = torch.LongTensor([d["labels"] for d in batch])
    return x, y  # (N,T) and (N)
