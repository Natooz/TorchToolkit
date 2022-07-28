#!/usr/bin/python3 python

""" Positional Encodings methods

"""

from math import pi, log
from typing import Tuple

from torch import Tensor, sin, cos, arange, exp, zeros, ones, linspace, einsum, repeat_interleave, cat, stack, matmul, \
    float, randn
from torch.nn import Module, Parameter, Dropout
from torch.nn.functional import pad


class PositionalEncoding(Module):
    """ Positional Encoding base class
    Holds the type of positional encoding
    """

    def __init__(self, absolute: bool = False, rotary: bool = False, relative: bool = False):
        super().__init__()
        self.absolute = absolute
        self.rotary = rotary
        self.relative = relative


class AbsolutePositionalEncoding(PositionalEncoding):
    """ Module injecting positional information in the embeddings of a sequence.
    To be used at the beginning of a transformer network, before the first layers.

    :param d_model: embedding size
    :param max_len: max length of the sequences that will be treated
    :param dropout: dropout value
    """

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__(absolute=True)
        self.dropout = Dropout(p=dropout)

        pe = zeros(max_len, d_model)
        position = arange(0, max_len, dtype=float).unsqueeze(1)
        div_term = exp(arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
        pe[:, 0::2] = sin(position * div_term)
        pe[:, 1::2] = cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """ Adds positional encoding to a sequence

        :param x: input tensor, shape (sequence length, batch size, embedding size)
        :return the tensor with positional encoding
        """

        x = x + self.pe[:x.size()[0], :].to(x.device, dtype=x.dtype)
        return self.dropout(x)


class RotaryPositionalEncoding(PositionalEncoding):
    """ Rotary "Positional Encoding", from the Roformer paper: https://arxiv.org/abs/2104.09864
    Adapted from Ludidrains's implementation: https://github.com/lucidrains/rotary-embedding-torch
    This module is to be used inside the attention layers.
    Create an object of this class and pass it to every attention layers, and
    pass the keys and queries tensors when calling it before calculating attention
    (whether dot product, feature map or other linear att)

    :param dim: rotations dim (must be <= to d_model)
    :param positions: starting and ending positions of the rotations to compute
    :param custom_freqs: if provided, will use these frequencies
    :param freqs_for: frequencies type. Can be lang, pixel or constant
    :param theta: theta parameter for lang frequencies
    :param max_freq:
    :param num_freqs:
    """

    def __init__(self, dim: int, positions: Tuple[int, int] = (0, 1024), custom_freqs: Tensor = None,
                 freqs_for: str = 'lang', theta: int = 10000, max_freq: int = 10, num_freqs: int = 1):
        super().__init__(rotary=True)
        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        self.register_buffer('freqs', freqs)  # parameters, 𝚯
        self.register_buffer('rotations', self.generate_rotations(positions))  # rotary matrix

    def generate_rotations(self, positions: Tuple[int, int]) -> Tensor:
        """ Generate the rotations tensor

        :param positions: starting and ending positions of the rotations to compute
        :return: the rotations tensor
        """
        rotations = einsum('..., f -> ... f', arange(*positions, dtype=self.freqs.dtype), self.freqs)
        rotations = repeat_interleave(rotations, 2, -1)  # repeat(rotations, '... n -> ... (n r)', r = 2)
        return rotations

    @staticmethod
    def _rotate_half(x) -> Tensor:
        x = x.unflatten(-1, (x.size(-1) // 2, 2))
        x1, x2 = x.unbind(dim=-1)
        x = stack((-x2, x1), dim=-1)
        return x.flatten(-2, -1)

    def forward(self, x, start_index=0):
        """ Applies the rotations to the given tensor

        :param x: input tensor, with a shape (batch_size * num_head, seq_len, head_dim)
        :param start_index: starting index of the input tensor, i.e. its position in "time"
        :return: the input tensor with rotary positional encoding
        """
        rot_dim = self.rotations.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= x.shape[-1], \
            f'feature dimension {x.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

        x_left, x, x_right = x[..., :start_index], x[..., start_index:end_index], x[..., end_index:]
        x = (x * self.rotations[:x.shape[-2]].cos()) + (self._rotate_half(x) * self.rotations[:x.shape[-2]].sin())
        return cat((x_left, x, x_right), dim=-1)


class RelativePositionalEncoding(PositionalEncoding):
    """ Relative Positional Encoding, as presented in the MusicTransformer:
    https://arxiv.org/abs/1809.04281
    This module is to be used inside the attention layers (Softmax Dot Product attention).
    Create an object of this class and pass it to every attention layers, and pass the
    queries tensors when calling it to compute the S_rel matrix.

    :param head_dim: dimension of the heads (should be d_model / num_heads)
    :param max_len: max length of the sequences that will be treated
    """

    def __init__(self, head_dim: int, max_len: int):
        super().__init__(relative=True)
        self.Er = Parameter(randn(max_len, head_dim))
        self.max_len = max_len

    def forward(self, q: Tensor) -> Tensor:
        """ Computes the S_rel matrix, to be added to QK dot product before softmax

        :param q: the Q (queries) matrix (batch_size * num_head, seq_len, head_dim)
        :return: S_rel matrix, relative distances
        """
        seq_len = q.size()[1]  # q.shape: (batch_size * num_heads, seq_len, head_dim)

        start = self.max_len - seq_len
        er_t = self.Er[start:, :].transpose(0, 1)  # (head_dim, seq_len)
        q_er = matmul(q, er_t)  # (batch_size, num_heads, seq_len, seq_len)
        s_rel = self._skew(q_er)  # (batch_size, num_heads, seq_len, seq_len)
        return s_rel

    @staticmethod
    def _skew(q_er: Tensor) -> Tensor:
        """ Skewing procedure to compute S_rel

        :param q_er: Q.Er matrix (batch_size * num_heads, seq_len, seq_len)
        :return: the final S_rel matrix
        """
        padded = pad(q_er, (1, 0))  # (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size_x_num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size_x_num_heads, num_cols, num_rows)  # (b_size*n_heads, 1 + seq_len, seq_len)
        s_rel = reshaped[:, 1:, :]  # (batch_size * num_heads, seq_len, seq_len)
        return s_rel
