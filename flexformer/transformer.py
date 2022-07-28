""" Transformer modules

"""

from typing import Union, Optional
from copy import deepcopy
from pathlib import Path, PurePath

from torch.nn import Module, LayerNorm, Linear, Dropout, Embedding, ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.functional import relu, gelu
from torch import Tensor, cuda, load, save, triu, full
from torch import device as device_

from .attention import Attention, ScaledDotProductAttention
from .pos_encoding import RotaryPositionalEncoding
from .train import select_device


class Transformer(Module):

    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.1, num_classes: int = None, nb_features_rotary_pos_enc: int = None, dtype=None,
                 activation: str = 'gelu', layer_norm_eps: float = 1e-5, device: device_ = None,
                 padding_token: int = None, causal_enc: bool = False, causal_dec: bool = False,
                 custom_encoder: Module = None, custom_decoder: Module = None, custom_input_module: Module = None,
                 custom_output_module: Module = None):
        r"""A complete Transformer module, with embedding and logits layers.
        TODO custom predict func ?

        :param num_encoder_layers: number of encoder layers, set 0 for no encoder
        :param num_decoder_layers: number of decoder layers, set 0 for no decoder
        :param d_model: model dimension (embedding dim)
        :param nhead: number of attention heads
        :param dim_feedforward: dimension of the feedforward layers (default: 2048)
        :param dropout: dropout value (default: 0.1)
        :param num_classes: number of classes, vocabulary size (default: None, is required if no custom_input_module and
                            custom_output_module are given)
        :param nb_features_rotary_pos_enc: number of features for Rotary positional encoding. This has to be >= to the
                            maximum sequence length that will be passed to the model. The PE module will be passed to
                            every encoder and decoder layers. Give None for no Rotary PE. (default: None)
        :param dtype:
        :param layer_norm_eps: the eps value in layer normalization components (default=1e-5)
        :param device: device onto the model must be put (default: cuda:0 if cuda.is_available() else cpu)
        :param padding_token: specifies a padding token idx, si that it doesn't contribute to the gradients.
                            The Embedding layer at index padding_token will therefore not be updated during training.
                            (default: None)
        :param causal_enc: if set True, a causal attention mask will automatically be applied in the encoder
                            (default: False)
        :param causal_dec: if set True, a causal attention mask will automatically be applied in the decoder
                            (default: False)
        :param custom_encoder: custom encoder module, will override other params
        :param custom_decoder: custom decoder module, will override other params
        :param custom_input_module: custom input module, which creates embeddings from token sequences
                            (default: None)
        :param custom_output_module: custom output module, is intended to compute logits from the last hidden states
                            (default: None)
        """
        super().__init__()

        # ATTRIBUTES
        self.device = device if device is not None else select_device()
        self.causal_enc = causal_enc
        self.causal_dec = causal_dec
        self.padding_token = padding_token

        # ASSERTIONS
        head_dim, rest = divmod(d_model, nhead)
        assert rest == 0, f'Non valid combination of model dimension ({d_model}) and number of heads ({nhead})'
        assert head_dim % 2 == 0, f'Non valid combination of model dimension ({d_model}) and number of heads ({nhead})'

        # MODULES
        self.embedder = Embedding(num_classes, d_model, padding_token) \
            if custom_input_module is None else custom_input_module
        pos_enc = RotaryPositionalEncoding(head_dim // 2, (0, nb_features_rotary_pos_enc)) \
            if nb_features_rotary_pos_enc is not None else None

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            self.encoder = None
            if num_encoder_layers > 0:
                encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                        layer_norm_eps, device, dtype)
                encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)
                self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
                if pos_enc is not None:
                    for layer in self.encoder.layers:
                        layer.self_attn.pos_enc = pos_enc

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            self.decoder = None
            if num_decoder_layers > 0:
                decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                        layer_norm_eps, device, dtype)
                decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)
                self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
                if pos_enc is not None:
                    for layer in self.decoder.layers:
                        layer.self_attn.pos_enc = pos_enc
                        layer.cross_attn.pos_enc = pos_enc

        self.to_logits = Linear(in_features=d_model, out_features=num_classes) \
            if custom_output_module is None else custom_output_module

        # INITIALIZATION
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        self.to(self.device)

    def forward(self, src: Optional[Tensor] = None, tgt: Optional[Tensor] = None,
                src_attn_mask: Optional[Tensor] = None, tgt_attn_mask: Optional[Tensor] = None,
                mem_attn_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                mem_key_padding_mask: Optional[Tensor] = None, auto_padding_mask: bool = False):
        r"""Following the model architecture, it will pass source and target sequences through the encoder
        and decoder layers. In the case of:
            - encoder + decoder (seq2seq) --> you need to pass src and tgt arguments
            - encoder only --> just pass the src
            - decoder only --> pass tgt, and a src which will serve as memory / hidden state
            + all the associated masks in each case if needed

        :param src: source sequence of the encoder, shape (S,N) (default: None)
        :param tgt: target sequence of the decoder, shape (T,N) (default: None)
        :param src_attn_mask: attention mask for the source sequence, shape (S,S) (default: None)
        :param tgt_attn_mask: attention mask for the target sequence, shape (T,T) (default: None)
        :param mem_attn_mask: attention mask for the encoder-decoder attention, shape (T,S) (default: None)
        :param src_key_padding_mask: padding mask of the source sequence, shape (N,S) (default: None)
        :param tgt_key_padding_mask: padding mask of the target sequence, shape (N,T) (default: None)
        :param mem_key_padding_mask: padding mask of the encoder-decoder attention, shape (N,S) (default: None)
        :param auto_padding_mask: if True and if no padding mask where given and if self.padding_token is not None,
                                it will automatically create padding masks for source and target (not memory) from
                                the padding token given when creating the model. (default: False)
        :return: output tensor, from the encoder is src is given and not tgt, else from the decoder, shape (T,N,C)
        """
        # Causal mask
        if self.causal_enc and src_attn_mask is None:
            src_attn_mask = self.create_causal_mask(src.shape[0])  # (S,S)
        if self.causal_dec and tgt_attn_mask is None:
            tgt_attn_mask = self.create_causal_mask(tgt.shape[0])  # (T,T)
        # Padding mask
        if auto_padding_mask and src_key_padding_mask is None:
            src_key_padding_mask = (src == self.padding_token).to(self.device).t()  # (N,T)
        if auto_padding_mask and tgt_key_padding_mask is None:
            tgt_key_padding_mask = (src == self.padding_token).to(self.device).t()  # (N,T)

        if self.encoder is not None and src is not None:
            src = self.embedder(src)  # (T,N,E)
            src = self.encoder(src, src_attn_mask, src_key_padding_mask=src_key_padding_mask)

        if self.decoder is not None:
            tgt = self.embedder(tgt)  # (T,N,E)
            tgt = self.decoder(tgt, src, tgt_attn_mask, mem_attn_mask, tgt_key_padding_mask, mem_key_padding_mask)

        return self.to_logits(src) if tgt is None else self.to_logits(tgt)  # (T,N,C)

    def train_forward(self, src: Optional[Tensor] = None, tgt: Optional[Tensor] = None,
                      src_attn_mask: Optional[Tensor] = None, tgt_attn_mask: Optional[Tensor] = None,
                      mem_attn_mask: Optional[Tensor] = None,
                      src_key_padding_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                      mem_key_padding_mask: Optional[Tensor] = None, auto_padding_mask: bool = True,
                      batch_first: bool = True):
        r"""Training function, which takes the same arguments as model.forward but apply the permutations
        so that the returned tensor has shape (N,C,T) to compute the cross-entropy loss

        :param src: source sequence of the encoder, shape (N,S) if batch_first else (S,N) (default: None)
        :param tgt: target sequence of the decoder, shape (N) if batch_first else (T,N) (default: None)
        :param src_attn_mask: attention mask for the source sequence, shape (S,S) (default: None)
        :param tgt_attn_mask: attention mask for the target sequence, shape (T,T) (default: None)
        :param mem_attn_mask: attention mask for the encoder-decoder attention, shape (T,S) (default: None)
        :param src_key_padding_mask: padding mask of the source sequence, shape (N,S) (default: None)
        :param tgt_key_padding_mask: padding mask of the target sequence, shape (N,T) (default: None)
        :param mem_key_padding_mask: padding mask of the encoder-decoder attention, shape (N,S) (default: None)
        :param auto_padding_mask: if True and if no padding mask where given and if self.padding_token is not None,
                                it will automatically create padding masks for source and target (not memory) from
                                the padding token given when creating the model. (default: True)
        :param batch_first: set True if the first dimension of your inputs is batch (default: True)
        :return: output tensor of the decoder, shape (N,C,T)
        """
        if src is not None and batch_first:
            src = src.t()
        if tgt is not None and batch_first:
            tgt = tgt.t()
        out = self.forward(src, tgt, src_attn_mask, tgt_attn_mask, mem_attn_mask,
                           src_key_padding_mask, tgt_key_padding_mask, mem_key_padding_mask,
                           auto_padding_mask)  # (T,N,C)
        return out.permute(1, 2, 0)  # (N,C,T)

    def create_causal_mask(self, size: int) -> Tensor:
        r"""Generates a causal attention mask, to be used with left-to-right predictions.

        :param size: size of the mask (S)
        :return: the attention mask tensor, of shape (S,S)
        """
        return triu(full((size, size), float('-inf')), diagonal=1).to(self.device)

    def save_checkpoint(self, path: Union[str, Path, PurePath], optimizer_state=0):
        r"""Saves a checkpoint containing the model's parameters and attributes
        Is typically used during training, or to save different baselines

        :param path: path to save checkpoint
        :param optimizer_state: (optional) optimizer.dict_state() if saved during training
        """
        save({'model_state_dict': self.state_dict(), 'optimizer_state_dict': optimizer_state}, path)

    def load_checkpoint(self, path: Union[str, Path, PurePath]):
        r"""Loads a checkpoint containing the model's parameters and attributes

        :param path: path to the checkpoint to load
        :return the optimizer's state_dict, if taking back training
        """
        if not cuda.is_available():
            checkpoint = load(path, map_location=device_('cpu'))
            self.device = device_('cpu')
        else:
            checkpoint = load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['optimizer_state_dict']


class TransformerEncoderLayer(Module):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "gelu", layer_norm_eps: float = 1e-5, device: device_ = None, dtype=None,
                 custom_self_attn_layer: Attention = None) -> None:
        r"""A Transformer encoder layer, using self-attention on input (source) sequence.

        :param d_model: model dimension (embedding dim)
        :param nhead: number of attention heads
        :param dim_feedforward: dimension of the feedforward layers (default: 2048)
        :param dropout: dropout value (default: 0.1)
        :param activation: activation function, must be gelu or relu (default: gelu)
        :param layer_norm_eps: the eps value in layer normalization components (default=1e-5)
        :param device: device onto the model must be put (default: None)
        :param dtype:
        :param custom_self_attn_layer: custom self-attention layer (default: None)
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        # Attention
        self.self_attn = custom_self_attn_layer if custom_self_attn_layer is not None \
            else ScaledDotProductAttention(d_model, nhead, dropout, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_attn_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        :param src: source sequence of the decoder, shape (S,N,E)
        :param src_attn_mask: attention mask, shape (S,S) (default: None)
        :param src_key_padding_mask: padding mask of the hidden state, shape (N,S) (default: None)
        :return: output tensor of the decoder, shape (S,N,E)
        """
        src2 = self.self_attn(src, src, attn_mask=src_attn_mask, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


class TransformerDecoderLayer(Module):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "gelu", layer_norm_eps: float = 1e-5, device: device_ = None, dtype=None,
                 custom_self_attn_layer: Attention = None, custom_cross_attn_layer: Attention = None) -> None:
        r"""A Transformer decoder layer, with both self-attention and cross-attention.
        A decoder layer receives in input a target sequence, and a hidden state (memory) sequence from
        an Encoder.

        :param d_model: model dimension (embedding dim)
        :param nhead: number of attention heads
        :param dim_feedforward: dimension of the feedforward layers (default: 2048)
        :param dropout: dropout value (default: 0.1)
        :param activation: activation function, must be gelu or relu (default: gelu)
        :param layer_norm_eps: the eps value in layer normalization components (default=1e-5)
        :param device: device onto the model must be put (default: None)
        :param dtype:
        :param custom_self_attn_layer: custom self-attention layer (default: None)
        :param custom_cross_attn_layer: custom cross-attention (encoder-decoder) layer (default: None)
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()

        # Attention
        self.self_attn = custom_self_attn_layer if custom_self_attn_layer is not None \
            else ScaledDotProductAttention(d_model, nhead, dropout, **factory_kwargs)
        self.cross_attn = custom_cross_attn_layer if custom_cross_attn_layer is not None \
            else ScaledDotProductAttention(d_model, nhead, dropout, **factory_kwargs)

        # Feedforward modules
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, mem: Tensor, tgt_attn_mask: Optional[Tensor] = None,
                mem_attn_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                mem_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        :param tgt: target sequence of the decoder, shape (T,N,E)
        :param mem: hidden state from the encoder, shape (S,N,E)
        :param tgt_attn_mask: attention mask for the target sequence, shape (T,T) (default: None)
        :param mem_attn_mask: attention mask for the memory, shape (T,S) (default: None)
        :param tgt_key_padding_mask: padding mask of the target sequence, shape (T,N) (default: None)
        :param mem_key_padding_mask: padding mask of the hidden state, shape (S,N) (default: None)
        :return: output tensor of the decoder, shape (T,N,E)
        """
        tgt2 = self.self_attn(tgt, tgt, attn_mask=tgt_attn_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.cross_attn(tgt, mem, attn_mask=mem_attn_mask, key_padding_mask=mem_key_padding_mask)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class TransformerEncoder(Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int, norm: Optional[Module] = None):
        r"""TransformerEncoder, a stack of N encoder layers
        This module will stack num_layers copies of encoder_layer.
        It can be used with a TransformerDecoder to form a seq2seq model, or as is to
        form a model like BERT or GPT, trained to solve natural language understanding or
        natural language generation tasks for instance.

        :param encoder_layer: the base encoder layer
        :param num_layers: number of layers
        :param norm: layer normalization module (default: None)
        """
        super(TransformerEncoder, self).__init__()
        self.layers = ModuleList([deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src: Tensor, attn_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) \
            -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        :param src: source sequence of the decoder, shape (S,N,E)
        :param attn_mask: attention mask, shape (S,S) (default: None)
        :param src_key_padding_mask: padding mask of the hidden state, shape (N,S) (default: None)
        :return: output tensor of the decoder, shape (S,N,E)src_key_padding_mask
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_attn_mask=attn_mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int, norm: Optional[Module] = None):
        r"""TransformerDecoder, a stack of N decoder layers.
        This module will stack num_layers copies of decoder_layer.
        It is usually used with a TransformerEncoder to form a seq2seq model architecture,
        to solve neural machine translation or question answering tasks for instance.

        :param decoder_layer: the base decoder layer
        :param num_layers: number of layers
        :param norm: layer normalization module (default: None)
        """
        super(TransformerDecoder, self).__init__()
        self.layers = ModuleList([deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, tgt: Tensor, mem: Tensor, tgt_attn_mask: Optional[Tensor] = None,
                mem_attn_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                mem_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layers in turn.

        :param tgt: target sequence of the decoder, shape (T,N,E)
        :param mem: hidden state from the encoder, shape (S,N,E)
        :param tgt_attn_mask: attention mask for the target sequence, shape (T,T) (default: None)
        :param mem_attn_mask: attention mask for the memory, shape (T,S) (default: None)
        :param tgt_key_padding_mask: padding mask of the target sequence, shape (N,T) (default: None)
        :param mem_key_padding_mask: padding mask of the hidden state, shape (N,S) (default: None)
        :return: output tensor of the decoder, shape (T,N,E)
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, mem, tgt_attn_mask=tgt_attn_mask, mem_attn_mask=mem_attn_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask, mem_key_padding_mask=mem_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


def _get_activation_fn(activation: str):
    if activation == 'relu':
        return relu
    elif activation == 'gelu':
        return gelu

    raise RuntimeError(f'activation should be relu/gelu, not {activation}')
