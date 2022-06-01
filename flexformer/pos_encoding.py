#!/usr/bin/python3 python

""" Positional Encodings methods

"""

from math import pi, log, sqrt
from typing import Tuple, Union

import torch
from torch.nn import Module, Parameter, Conv1d, Conv2d, Conv3d, Dropout
from torch.nn.functional import pad, softplus
from torch import Tensor


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

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
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
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        self.register_buffer('freqs', freqs)  # parameters, 𝚯
        self.register_buffer('rotations', self.generate_rotations(positions))  # rotary matrix

    def generate_rotations(self, positions: Tuple[int, int]) -> Tensor:
        """ Generate the rotations tensor

        :param positions: starting and ending positions of the rotations to compute
        :return: the rotations tensor
        """
        rotations = torch.einsum('..., f -> ... f', torch.arange(*positions, dtype=self.freqs.dtype), self.freqs)
        rotations = torch.repeat_interleave(rotations, 2, -1)  # repeat(rotations, '... n -> ... (n r)', r = 2)
        return rotations

    @staticmethod
    def _rotate_half(x) -> Tensor:
        x = x.unflatten(-1, (x.size(-1) // 2, 2))
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
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
        return torch.cat((x_left, x, x_right), dim=-1)


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
        self.Er = Parameter(torch.randn(max_len, head_dim))
        self.max_len = max_len

    def forward(self, q: Tensor) -> Tensor:
        """ Computes the S_rel matrix, to be added to QK dot product before softmax

        :param q: the Q (queries) matrix (batch_size * num_head, seq_len, head_dim)
        :return: S_rel matrix, relative distances
        """
        seq_len = q.size()[1]  # q.shape: (batch_size * num_heads, seq_len, head_dim)

        start = self.max_len - seq_len
        er_t = self.Er[start:, :].transpose(0, 1)  # (head_dim, seq_len)
        q_er = torch.matmul(q, er_t)  # (batch_size, num_heads, seq_len, seq_len)
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


# TODO adapt this heritate from Pos Enc class
class SinSPE(PositionalEncoding):
    """Code generator for sinusoidal stochastic positional encoding.

    :param num_heads: The number of attention heads.
    :param head_dim: The number of input features per attention head.
    :param num_realizations: The number of realizations of the stochastic process (R).
    :param num_sines: The number of sin and cos components (K).
    """

    def __init__(self, num_heads: int = 8, head_dim: int = 64, num_realizations: int = 256, num_sines: int = 1):
        super().__init__(sin_spe=True)

        # saving dimensions
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_sines = num_sines
        self.num_realizations = num_realizations

        # register the parameter
        for param in ['freqs', 'offsets', 'gains']:
            self.register_parameter(param, Parameter(torch.randn(num_heads, head_dim, num_sines)))

        # normalize the gains
        self.gains.data[...] /= torch.sqrt(self.gains.norm(dim=-1, keepdim=True)) / 2.

        # bias initial frequencies to low values for long term range
        self.freqs.data[...] -= 4.

        self.code_shape = (num_heads, head_dim)

    def forward(self, shape, num_realizations=None):
        """
        Generate the code, composed of a random QBar and Kbar,
        depending on the parameters, and return them for use with a
        SPE module to actually encode queries and keys.
        :param shape: The outer shape of the inputs: (batchsize, *size)
        :param num_realizations: if provided, overrides self.num_realizations
        """
        if len(shape) != 2:
            raise ValueError('Only 1D inputs are supported by SineSPE')

        # get shape of the queries. Here it's only 1d
        max_len = shape[1]

        # build omega_q and omega_k,
        # with shape (num_heads, keys_dim, length, 2*num_sines)
        indices = torch.linspace(0, max_len - 1, max_len, device=self.freqs.device)

        # making sure the frequencies are in [0, 0.5]
        freqs = torch.sigmoid(self.freqs[:, :, None, :]) / 2.

        phases_q = (2 * pi * freqs * indices[None, None, :, None] + self.offsets[:, :, None, :])
        omega_q = torch.stack([torch.cos(phases_q), torch.sin(phases_q)], dim=-1).view(
            1, self.num_heads, self.head_dim, max_len, 2 * self.num_sines
        )

        phases_k = (2 * pi * freqs * indices[None, None, :, None])
        omega_k = torch.stack([torch.cos(phases_k), torch.sin(phases_k)], dim=-1).view(
            1, self.num_heads, self.head_dim, max_len, 2 * self.num_sines
        )

        # gains is (num_heads, keys_dim, num_sines). Making then nonnegative with softplus
        gains = softplus(self.gains)

        # now upsample it
        gains = torch.stack(
            (gains, gains), dim=-1).view(
            self.num_heads, self.head_dim, 2 * self.num_sines)

        # the number of realizations is overrided by the function argument if provided
        if num_realizations is None:
            num_realizations = self.num_realizations

        # draw noise of appropriate shape on the right device
        z = torch.randn(
            1, self.num_heads, self.head_dim, 2 * self.num_sines,
            num_realizations,
            device=self.freqs.device) / sqrt(self.num_sines * 2)

        # scale each of the 2*num_sines by the appropriate gain
        # z is still (1, num_heads, keys_dim, 2*num_sines, num_realizations)
        z = z * gains[None, ..., None]

        # computing the sum over the sines.
        # gets (1, num_heads, keys_dim, length, num_realizations)
        qbar = torch.matmul(omega_q, z)
        kbar = torch.matmul(omega_k, z)

        # permuting them to be (1, length, num_heads, keys_dim, num_realizations)
        qbar = qbar.permute(0, 3, 1, 2, 4)
        kbar = kbar.permute(0, 3, 1, 2, 4)

        # final scaling
        scale = (num_realizations * self.head_dim) ** 0.25
        return qbar / scale, kbar / scale

    def get_posattn_matrix(self, max_len=2048):  # useless here ?
        indices = torch.linspace(0, max_len - 1, max_len, device=self.freqs.device)

        # making sure the frequencies are in [0, 0.5]
        freqs = torch.sigmoid(self.freqs[:, :, None, :]) / 2.

        phases_q = (2 * pi * freqs * indices[None, None, :, None] + self.offsets[:, :, None, :])
        omega_q = torch.stack([torch.cos(phases_q), torch.sin(phases_q)], dim=-1).view(
            1, self.num_heads, self.head_dim, max_len, 2 * self.num_sines
        )

        phases_k = (2 * pi * freqs * indices[None, None, :, None])
        omega_k = torch.stack([torch.cos(phases_k), torch.sin(phases_k)], dim=-1).view(
            1, self.num_heads, self.head_dim, max_len, 2 * self.num_sines
        )

        # gains is (num_heads, keys_dim, 2*num_sines). Making then nonnegative with softplus
        gains = softplus(self.gains)
        # gains = gains / torch.sqrt(gains.norm(dim=-1, keepdim=True))
        gains = torch.stack(
            (gains, gains), dim=-1).view(
            self.num_heads, self.head_dim, 2 * self.num_sines)

        gains_squared_diag = torch.diag_embed(gains ** 2)

        print('[get posattn matrix] Omega_q: {}, lambda: {}, Omega_k: {}'.format(
            omega_q.size(), gains_squared_diag.size(), omega_k.size()
        ))
        # print (gains_squared_diag[0, 0])

        # get (1, num_heads, keys_dim) attention maps, each of size (max_len, max_len)
        omega_q_mult_gains_squared_diag = torch.einsum(
            'ihdmk, hdku -> ihdmu',
            omega_q, gains_squared_diag
        )
        pos_attn_matrices = torch.einsum(
            'ihdmk, ihdnk -> ihdmn',
            omega_q_mult_gains_squared_diag, omega_k
        )
        print('[get posattn matrix] pos_attn: {}'.format(
            pos_attn_matrices.size()
        ))

        return pos_attn_matrices


class ConvSPE(PositionalEncoding):
    """Code generator for convolutive stochastic positional encoding.

    :param ndim: The number of attention dimensions (e.g. 1 = sequence, 2 = image).
    :param num_heads: The number of attention heads.
    :param in_features: The number of input features per attention head.
    :param num_realizations: The number of realizations of the stochastic process (R).
    :param kernel_size: The size of the convolution kernel.
    """

    def __init__(self, ndim: int = 1, num_heads: int = 8, in_features: int = 64, num_realizations: int = 256,
                 kernel_size: Union[int, Tuple[int, ...]] = 200):
        super().__init__(conv_spe=True)

        if ndim == 1:
            conv_class = Conv1d
        elif ndim == 2:
            conv_class = Conv2d
        elif ndim == 3:
            conv_class = Conv3d
        else:
            raise Exception('`ndim` must be 1, 2 or 3')

        # making kernel_size a list of length dimension in any case
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * ndim

        # saving dimensions
        self.ndim = ndim
        self.in_features = in_features
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.num_realizations = num_realizations

        # create the two convolution layers
        self.conv_q = conv_class(in_channels=num_heads * in_features, out_channels=num_heads * in_features, stride=1,
                                 kernel_size=kernel_size, padding=0, bias=False, groups=num_heads * in_features)
        self.conv_k = conv_class(in_channels=num_heads * in_features, out_channels=num_heads * in_features, stride=1,
                                 kernel_size=kernel_size, padding=0, bias=False, groups=num_heads * in_features)

        # random init
        self.conv_q.weight.data = torch.rand(self.conv_q.weight.shape)
        self.conv_k.weight.data = torch.rand(self.conv_k.weight.shape)

        scale = sqrt(torch.prod(torch.tensor(kernel_size).float()) / 2)
        self.conv_q.weight.data = self.conv_q.weight.data / scale
        self.conv_k.weight.data = self.conv_k.weight.data / scale

        self.code_shape = (num_heads, in_features)

    def forward(self, shape, num_realizations=None):
        """
        generate the random QBar and Kbar, depending on the parameters,
        Args:
            shape: The outer shape of the inputs: (batchsize, *size)
            num_realizations:
        """
        batchsize = 1
        original_shape = shape[1:]

        # decide on the size of the signal to generate
        # (larger than desired to avoid border effects)
        shape = [4 * k + s for (k, s) in zip(self.kernel_size, original_shape)]

        # the number of realizations is overrided by the function argument if provided
        if num_realizations is None:
            num_realizations = self.num_realizations

        # draw noise of appropriate shape on the right device
        z = torch.randn(
            batchsize * num_realizations,
            self.num_heads * self.in_features,
            *shape,
            device=self.conv_q.weight.device)

        # apply convolution, get (batchsize*num_realizations, num_heads*keys_dim, *shape)
        kbar = self.conv_q(z)
        qbar = self.conv_k(z)

        # truncate to desired shape (remove the start to avoid the border effects)
        for dim in range(len(shape)):
            k = self.kernel_size[dim]
            s = original_shape[dim]

            indices = [slice(batchsize * num_realizations),
                       slice(self.num_heads * self.in_features)] + [slice(k, k + s, 1), ]
            qbar = qbar[indices]
            kbar = kbar[indices]

        # making (batchsize, num_realizations, num_heads, keys_dim, *shape)
        kbar = kbar.view(batchsize, num_realizations,
                         self.num_heads, self.in_features, *original_shape)
        qbar = qbar.view(batchsize, num_realizations,
                         self.num_heads, self.in_features, *original_shape)

        # permuting to be
        # (batchsize, *shape, num_heads, keys_dim, num_realizations) as desired
        qbar = qbar.permute(0, *[x for x in range(4, self.ndim + 4)], 2, 3, 1)
        kbar = kbar.permute(0, *[x for x in range(4, self.ndim + 4)], 2, 3, 1)

        # final scaling
        scale = (num_realizations * self.in_features) ** 0.25
        return qbar / scale, kbar / scale

    def get_posattn_matrix(self, shape, num_realizations=None):
        batchsize = 1
        original_shape = shape[1:]

        # decide on the size of the signal to generate
        # (larger than desired to avoid border effects)
        shape = [4 * k + s for (k, s) in zip(self.kernel_size, original_shape)]

        # the number of realizations is overrided by the function argument if provided
        if num_realizations is None:
            num_realizations = self.num_realizations

        # draw noise of appropriate shape on the right device
        z = torch.randn(
            batchsize * num_realizations,
            self.num_heads * self.in_features,
            *shape,
            device=self.conv_q.weight.device)

        # apply convolution, get (batchsize*num_realizations, num_heads*keys_dim, *shape)
        kbar = self.conv_q(z)
        qbar = self.conv_k(z)

        for dim in range(len(shape)):
            k = self.kernel_size[dim]
            s = original_shape[dim]

            indices = [slice(batchsize * num_realizations),
                       slice(self.num_heads * self.in_features)] + [slice(k, k + s, 1), ]
            qbar = qbar[indices]
            kbar = kbar[indices]

        print('[get posattn matrix] Qbar: {}, Kbar: {}'.format(
            qbar.size(), kbar.size()
        ))

        # get (num_heads * keys_dim) attention maps, each of size (max_len, max_len)
        pos_attn_matrices = torch.einsum(
            'rdm, rdn -> dmn',
            qbar, kbar
        )
        print('[get posattn matrix] pos_attn: {}'.format(
            pos_attn_matrices.size()
        ))

        # reshape attention maps to the same shape as those of SineSPE
        pos_attn_matrices = pos_attn_matrices.view(
            batchsize, self.num_heads, self.in_features, original_shape[-1], original_shape[-1]
        )

        return pos_attn_matrices
