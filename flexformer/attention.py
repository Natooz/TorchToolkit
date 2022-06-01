""" Attention modules

Some of these implementations are inspired from:
https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/activation.py
https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py
https://github.com/lucidrains/performer-pytorch
https://github.com/idiap/fast-transformers

"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor, BoolTensor
from torch.nn.functional import relu, linear, softmax
from torch.nn.functional import dropout as dropout_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from fast_transformers.causal_product import causal_dot_product

from .pos_encoding import PositionalEncoding


class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0., bias: bool = True, kdim: int = None,
                 vdim: int = None, device: torch.device = None, dtype=None, merge_batch_head_dim: bool = True) -> None:
        r"""Basic Attention class, framework to scaled dot product attention and linear attentions (elu/favor+)

        :param embed_dim: embedding / model dimension
        :param num_heads: number of attention heads
        :param dropout: dropout value (default: 0.)
        :param bias: use biases in linear layers (default: True)
        :param kdim: dimension of keys, give None for embed_dim (default: None)
        :param vdim: dimension of values, give None for embed_dim (default: None)
        :param device: device to run (default: None)
        :param dtype: dtype (default None)
        :param merge_batch_head_dim: will merge head and batch dimensions for faster computation,
                                    incompatible with linear attention (will be set to False) (default: True)
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.merge_batch_head_dim = merge_batch_head_dim

        # pos_enc set to False
        self.pos_enc = PositionalEncoding()  # will be replaced when calling TransformerEncoder / Decoder layers

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def _compute_qkv(self, target: Tensor, source: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Computes Queries, Keys and Values (Q K anD V) from target and source.
        Target and source can have different embedding sizes.

        :param target: target sequence of shape (T,N,E)
        :param source: source sequence of shape (S,N,E)
        :return: Queries, Keys and Values tensor, of shape (N*nH, S, H) if merge_batch_head_dim else (N, nH, S, H)
        """
        # set up shape vars
        tgt_len, bsz, embed_dim = target.shape
        src_len, _, _ = source.shape
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(self.num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // self.num_heads

        # compute in-projection: queries, keys and values
        if self._qkv_same_embed_dim:  # (T, N, E) and (S, N, E)
            q, k, v = self._in_projection_packed(target, source, source)
        else:
            if self.in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = self.in_proj_bias.chunk(3)
            q = linear(target, self.q_proj_weight, b_q)  # (T, N, E)
            k = linear(source, self.k_proj_weight, b_k)  # (S, N, Es)
            v = linear(source, self.v_proj_weight, b_v)  # (S, N, Es)

        # reshape q, k, v for multihead attention and make em batch first (N*nH, S, H)
        if self.merge_batch_head_dim:
            q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
            k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
            v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        else:  # reshape to (N, nH, S, H)
            q = q.contiguous().view(tgt_len, bsz,  self.num_heads, head_dim).permute(1, 2, 0, 3)
            k = k.contiguous().view(-1, bsz, self.num_heads, head_dim).permute(1, 2, 0, 3)
            v = v.contiguous().view(-1, bsz, self.num_heads, head_dim).permute(1, 2, 0, 3)

        if self.pos_enc.rotary:  # Rotary pos encoding, applied to q and k before dot prod.
            q = self.pos_enc(q)
            k = self.pos_enc(k)

        return q, k, v  # (N*nH, S, H) or (N, nH, S, H)

    def _in_projection_packed(self, q: Tensor, k: Tensor, v: Tensor) \
            -> Tuple[Tensor, Tensor, Tensor]:
        r"""Project source and/or target sequences to get queries, keys and values.

        :param q: queries input tensor (target)
        :param k: queries input tensor (source)
        :param v: queries input tensor (source)
        :return: a tuple with respectively the queries, keys and values tensors
        """
        embed_dim = q.shape[-1]  # E
        if k is v:
            if q is k:
                # self-attention
                return linear(q, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
            else:
                # encoder-decoder attention
                w_q, w_kv = self.in_proj_weight.split([embed_dim, embed_dim * 2])
                if self.in_proj_bias is None:
                    b_q = b_kv = None
                else:
                    b_q, b_kv = self.in_proj_bias.split([embed_dim, embed_dim * 2])
                return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
        else:
            w_q, w_k, w_v = self.in_proj_weight.chunk(3)
            if self.in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = self.in_proj_bias.chunk(3)
            return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

    def forward(self, target: Tensor, source: Tensor, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:
        r"""To implement in inheriting classes

        :param target: target sequence of shape (T,N,E)
        :param source: source sequence of shape (S,N,E)
        :param key_padding_mask: key padding mask of shape (N,S)
        :param attn_mask: attention mask of shape (T,S)
        """
        raise NotImplementedError


class ScaledDotProductAttention(Attention):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0., bias: bool = True, kdim: int = None,
                 vdim: int = None, device: torch.device = None, dtype=None, merge_batch_head_dim: bool = True) -> None:
        """ Scaled dot product attention, as in the original Transformer paper "Attention is all you need"

        :param embed_dim: embedding / model dimension
        :param num_heads: number of attention heads
        :param dropout: dropout value (default: 0.)
        :param bias: use biases in linear layers (default: True)
        :param kdim: dimension of keys, give None for embed_dim (default: None)
        :param vdim: dimension of values, give None for embed_dim (default: None)
        :param device: device to run (default: None)
        :param dtype: dtype (default None)
        :param merge_batch_head_dim: will merge head and batch dimensions for faster computation,
                                    incompatible with linear attention (will be set to False) (default: True)
        """
        super().__init__(embed_dim, num_heads, dropout, bias, kdim, vdim, device, dtype, merge_batch_head_dim)

    def forward(self, target: Tensor, source: Tensor, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:
        tgt_len, bsz, _ = target.shape
        src_len, _, _ = source.shape

        q, k, v = self._compute_qkv(target, source)  # shape (bsz * num_heads, seq_len, head_dim)

        # prep attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                print("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)
            else:
                assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                    f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        # prep key padding mask
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            print("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
                expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                # attn_mask = attn_mask.logical_or(key_padding_mask)
                attn_mask = torch.logical_or(attn_mask, key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        # (deep breath) calculate attention and out projection
        q = q / math.sqrt(self.kdim)
        # (N*nH, T, dH) x (N*nH, S, dH) -> (N*nH, T, S)
        attn = torch.bmm(q, k.transpose(-2, -1))
        if self.pos_enc.relative:  # relative pos encoding
            attn += self.pos_enc(q)
        if attn_mask is not None:
            attn += attn_mask
        attn = softmax(attn, dim=-1)  # if nan: output = output.masked_fill(torch.isnan(output), 0)
        if not self.training and self.dropout > 0.0:
            attn = dropout_(attn, p=self.dropout)
        # (N*nH, T, S) x (N*nH, S, dH) -> (N*nH, T, dH)
        attn_output = torch.bmm(attn, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn_output = torch.nn.functional.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        return attn_output


class LinearAttention(Attention):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0., bias: bool = True, kdim: int = None,
                 vdim: int = None, device: torch.device = None, dtype=None, causal: bool = False):
        r"""Linear attention framework class

        Implementation inspired by:
        https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py
        https://github.com/lucidrains/performer-pytorch
        https://github.com/idiap/fast-transformers

        :param embed_dim: embedding / model dimension
        :param num_heads: number of attention heads
        :param dropout: dropout value (default: 0.)
        :param bias: use biases in linear layers (default: True)
        :param kdim: dimension of keys, give None for embed_dim (default: None)
        :param vdim: dimension of values, give None for embed_dim (default: None)
        :param device: device to run (default: None)
        :param dtype: dtype (default None)
        :param causal: causal attention, will quickly compute attention with causality (default: False)
        """
        self.causal = causal
        super().__init__(embed_dim, num_heads, dropout, bias, kdim, vdim, device, dtype, merge_batch_head_dim=False)

    def feature_map(self, x: Tensor, is_query: bool) -> Tensor:
        r"""Feature map function, projects input before linear attention approximation

        :param x: input tensor to project
        :param is_query: specify if input are queries
        """
        raise NotImplementedError

    def forward(self, target: Tensor, source: Tensor, key_padding_mask: Optional[BoolTensor] = None,
                attn_mask: Optional[BoolTensor] = None, eps: Optional[float] = 1e-6) -> Tensor:
        r"""Forward pass, computes attention linearly wrt the input sequence length.

        :param target: target sequence, from which queries are derived - shape: (T,N,E)
        :param source: source sequence, from which keys and values are derived - shape: (S,N,E)
        :param key_padding_mask: padding mask of keys - shape: (N,S)
        :param attn_mask: UNUSED HERE attention mask
        :param eps: epsilon value
        :return: attention output of shape (T,N,E)
        """
        tgt_len, bsz, _ = target.shape
        q, k, v = self._compute_qkv(target, source)  # (N, nH, S, dH), nH number of heads, dH=E/nH

        # project queries and keys with positive orthogonal vectors
        q = self.feature_map(q, is_query=True)  # (N, nH, S, dH)
        k = self.feature_map(k, is_query=False)  # (N, nH, S, dH)

        # Applies the key padding mask
        if key_padding_mask is not None:
            k = k * key_padding_mask.float()[:, None, :, None]

        # Compute the dot product of keys and values, then with queries
        if self.causal:
            k_cumsum = k.cumsum(dim=-2) + eps
            d_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))
            if isinstance(q, torch.cuda.HalfTensor):
                q, k, v = map(lambda t: t.float(), (q, k, v))  # HalfTensors converted to FloatTensors
                with torch.cuda.amp.autocast(enabled=False):
                    out = torch.cuda.amp.float_function(causal_dot_product(q, k, v))  # (N, nH, S, E/nH)
            else:
                out = causal_dot_product(q, k, v)

            out = torch.einsum('...nd,...n->...nd', out, d_inv)
        else:
            # fast transformers (N, nH, S, dH)
            # kv = torch.einsum("nsd,nsm->nmd", k, v)
            kv = torch.einsum("nhsd,nhsm->nhmd", k, v)
            z = 1 / (torch.einsum("nhtd,nhd->nht", q, k.sum(dim=-2)) + eps)  # Compute the normalizer
            out = torch.einsum("nhtd,nhmd,nht->nhtm", q, kv, z)  # Finally, compute and return the new values

        if self.training and self.dropout > 0.0:
            out = nn.functional.dropout(out, p=self.dropout)

        # reshape (N, nH, T, E/nH) --> (T, N, nH, E/nH) --> (T, N, E)
        out = out.permute(1, 2, 0, 3).contiguous().view(tgt_len, bsz, self.embed_dim)
        out = linear(out, self.out_proj.weight, self.out_proj.bias)

        return out


class EluAttention(LinearAttention):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0., bias: bool = True, kdim: int = None,
                 vdim: int = None, device: torch.device = None, dtype=None, causal: bool = False):
        r"""Linear attention with Elu feature maps
        Transformers are RNNs: https://arxiv.org/abs/2006.16236

        :param embed_dim: embedding / model dimension
        :param num_heads: number of attention heads
        :param dropout: dropout value (default: 0.)
        :param bias: use biases in linear layers (default: True)
        :param kdim: dimension of keys, give None for embed_dim (default: None)
        :param vdim: dimension of values, give None for embed_dim (default: None)
        :param device: device to run (default: None)
        :param dtype: dtype (default None)
        :param causal: causal attention, will quickly compute attention with causality
        """
        super().__init__(embed_dim, num_heads, dropout, bias, kdim, vdim, device, dtype, causal)

    def feature_map(self, x: Tensor, is_query: bool) -> Tensor:
        return nn.functional.elu(x) + 1


class FavorPlusAttention(LinearAttention):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0., bias: bool = True, kdim: int = None,
                 vdim: int = None, device: torch.device = None, dtype=None, nb_features: int = None,
                 ortho_scaling: int = 0, causal: bool = False):
        r"""Fast Attention Via positive Orthogonal Random vectors
        Linear attention mechanism introduced in Performers: https://arxiv.org/abs/2009.14794

        Implementation inspired by:
        https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py
        https://github.com/lucidrains/performer-pytorch

        :param embed_dim: embedding / model dimension
        :param num_heads: number of attention heads
        :param dropout: dropout value (default: 0.)
        :param bias: use biases in linear layers (default: True)
        :param kdim: dimension of keys, give None for embed_dim (default: None)
        :param vdim: dimension of values, give None for embed_dim (default: None)
        :param device: device to run (default: None)
        :param dtype: dtype (default None)
        :param nb_features: number of favor+ features
        :param ortho_scaling:
        :param causal: causal attention, will quickly compute attention with causality
        """
        super().__init__(embed_dim, num_heads, dropout, bias, kdim, vdim, device, dtype, causal)

        self.dim_heads = embed_dim // num_heads
        self.nb_features = nb_features if nb_features is not None else int(self.dim_heads * math.log(self.dim_heads))
        self.ortho_scaling = ortho_scaling

        features = self._generate_gaussian_orthogonal_features(device)
        self.register_buffer('features', features)

    def _generate_gaussian_orthogonal_features(self, device=None) -> Tensor:
        r"""Generates gaussian orthogonal features

        :param device: device run on
        :return: gaussian orthogonal features of shape (num_features, dim_head)
        """
        nb_full_blocks, remaining_rows = divmod(self.nb_features, self.dim_heads)
        block_list = []

        for _ in range(nb_full_blocks):
            q = self.__generate_orthogonal_matrix(self.dim_heads, device=device)
            block_list.append(q)

        if remaining_rows > 0:
            q = self.__generate_orthogonal_matrix(self.dim_heads, device=device)
            block_list.append(q[:remaining_rows])

        final_matrix = torch.cat(block_list)

        if self.ortho_scaling == 0:
            multiplier = torch.randn((self.nb_features, self.dim_heads), device=device).norm(dim=1)
        elif self.ortho_scaling == 1:
            multiplier = math.sqrt((float(self.dim_heads))) * torch.ones((self.nb_features,), device=device)
        else:
            raise ValueError(f'Invalid scaling {self.ortho_scaling}')

        return torch.diag(multiplier) @ final_matrix

    @staticmethod
    def __generate_orthogonal_matrix(size: int, device: torch.device = None) -> Tensor:
        r""" Generates a random orthogonal matrix

        :param size: size of the matrix
        :param device: device on which to run (default None, cpu)
        :return: a random orthogonal matrix
        """
        random_matrix = torch.randn((size, size), device=device)
        q, _ = torch.linalg.qr(random_matrix, mode="complete")
        return q.t()

    def feature_map(self, x: Tensor, is_query: bool, normalize_data: bool = True, eps: float = 1e-4) -> Tensor:
        r"""Projects an input with positive random features (PRF)

        :param x: tensor to approximate
        :param is_query: set True if data are queries
        :param normalize_data: normalize data (default True)
        :param eps: epsilon value (default 1e-4)
        :return: the approximated softmax kernel
        """
        n, nh, *_ = x.shape  # (N, nH, l, dH) <=> batch_size, n_head, seq_len, head_dim

        data_normalizer = (x.shape[-1] ** -0.25) if normalize_data else 1.
        ratio = self.features.shape[0] ** -0.5

        projection = self.features.repeat(n, nh, 1, 1)  # (N, nH, nf, dH)
        projection = projection.type_as(x)

        data_dash = torch.einsum('...id,...fd->...if', (data_normalizer * x), projection)  # (N, nH, l, nf)

        diag_data = x ** 2
        diag_data = torch.sum(diag_data, dim=-1)
        diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
        diag_data = diag_data.unsqueeze(dim=-1)

        if is_query:
            data_dash = ratio * (torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=-1, keepdim=True)) + eps)
        else:
            data_dash = ratio * (
                    torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True)) + eps)

        return data_dash.type_as(x)
