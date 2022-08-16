# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, List

import torch
import torch.nn.functional as F
from fairseq.models import FairseqEncoder
from fairseq.models.speech_to_text import (
    ConvTransformerEncoder,
)
from fairseq.models.speech_to_text.utils import attention_suppression
from fairseq.models.speech_to_text.utils import (
    lengths_to_encoder_padding_mask,
    segments_to_sequence,
    sequence_to_segments,
)
from fairseq.modules import MultiheadAttention, TransformerEncoderLayer
from torch import nn, Tensor
import json
from fairseq.data.data_utils import lengths_to_padding_mask
# ------------------------------------------------------------------------------
#   AugmentedMemoryConvTransformerEncoder
# ------------------------------------------------------------------------------


class AugmentedMemoryConvTransformerEncoder_no_sum(ConvTransformerEncoder):
    def __init__(self, args):
        super().__init__(args)

        args.encoder_stride = self.stride()

        self.left_context_after_stride = args.left_context // args.encoder_stride
        self.right_context_after_stride = args.right_context // args.encoder_stride

        self.transformer_layers = nn.ModuleList([])

        # Layer sharing code
        encoder_weight_share_list = getattr(args, "share_encoder_ffn_attn_layer", None)
        if encoder_weight_share_list is None:
            encoder_weight_share_list = []
        else:
            shared_weights_layer = AugmentedMemoryTransformerEncoderLayer(args)
        print(f"Encoder: Sharing layers: {encoder_weight_share_list}")
        for layer_idx in range(args.encoder_layers):
            if layer_idx+1 in encoder_weight_share_list:
                self.transformer_layers.append(shared_weights_layer)
            else:
                self.transformer_layers.append(AugmentedMemoryTransformerEncoderLayer(args))

        self.share_mem_bank_layers = args.share_mem_bank_layers

    def stride(self):
        # Hard coded here. Should infer from convs in future
        stride = 4
        return stride

    def initialize_states(self, states):
        if states is None:
            # Creates states;
            states = [{"memory_banks": [], "encoder_states": None} for i in range(len(self.transformer_layers))]
            if self.share_mem_bank_layers is not None:
                # Initializes Memory banks
                rows = len(self.share_mem_bank_layers)
                for i in range(rows):
                    cols = len(self.share_mem_bank_layers[i]);
                    for j in range(cols):
                        states[self.share_mem_bank_layers[i][j]]["memory_banks"] = states[self.share_mem_bank_layers[i][0]]["memory_banks"]
        return states

    def forward(self, src_tokens, src_lengths, states=None):
        """Encode input sequence.
        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        bsz, max_seq_len, _ = src_tokens.size()
        x = (
            src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim)
            .transpose(1, 2)
            .contiguous()
        )
        x = self.conv(x)
        bsz, _, output_seq_len, _ = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous().view(output_seq_len, bsz, -1)
        x = self.out(x)
        x = self.embed_scale * x

        subsampling_factor = max_seq_len * 1.0 / output_seq_len

        input_lengths = torch.min(
            (src_lengths.float() / subsampling_factor).ceil().long(),
            x.size(0) * src_lengths.new_ones([src_lengths.size(0)]).long(),
        )

        encoder_padding_mask, _ = lengths_to_encoder_padding_mask(
            input_lengths, batch_first=True
        )

        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)

        x += positions

        x = F.dropout(x, p=self.dropout, training=self.training)

        # State to store memory banks etc.
        states = self.initialize_states(states)
        #print("states after initialize_states: ", states)

        for i, layer in enumerate(self.transformer_layers):
            # x size:
            # (self.left_size + self.segment_size + self.right_size)
            # / self.stride, num_heads, dim
            # TODO: Consider mask here 
            x = layer(x, states[i], i)
            if self.right_context_after_stride != 0:
                states[i]["encoder_states"] = x[self.left_context_after_stride : -self.right_context_after_stride]
            else:
                states[i]["encoder_states"] = x[self.left_context_after_stride : ]

        if self.right_context_after_stride != 0:
            lengths = (
                (
                    ~encoder_padding_mask[:, self.left_context_after_stride : -self.right_context_after_stride]
                )
                .sum(dim=1, keepdim=True)
                .long()
            )
        else:
            lengths = (
                (
                    ~encoder_padding_mask[:, self.left_context_after_stride :]
                )
                .sum(dim=1, keepdim=True)
                .long()
            )
        return states[-1]["encoder_states"], lengths, states


# ------------------------------------------------------------------------------
#   AugmentedMemoryTransformerEncoderLayer
# ------------------------------------------------------------------------------
class AugmentedMemoryTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)

        self.left_context = args.left_context // args.encoder_stride
        self.right_context = args.right_context // args.encoder_stride

        self.share_mem_bank_layers = args.share_mem_bank_layers
        self.mem_bank_after = args.mem_bank_after

        self.mem_bank_size = args.mem_bank_size
        self.pool = torch.nn.AdaptiveAvgPool1d(self.mem_bank_size)
        self.max_memory_size = args.max_memory_size
        self.increase_context = args.increase_context
        self.tanh_on_mem = args.tanh_on_mem
        if self.tanh_on_mem:
            self.squash_mem = torch.tanh
            self.nonlinear_squash_mem = True
        else:
            self.squash_mem = lambda x: x
            self.nonlinear_squash_mem = False

    def update_mem_banks(self, state, input, layer_num):
        length, _, _ = input.size()
        if self.increase_context:
            segment = input[0:length]
        else:
            segment = input[self.left_context:length-self.right_context]
        
        segment = segment.transpose(0,2)
        next_m = self.pool(segment)
        next_m = next_m.transpose(0,2)

        if self.share_mem_bank_layers is not None:
          if not any(layer_num in layer for layer in self.share_mem_bank_layers):
            next_m = self.squash_mem(next_m)
            state["memory_banks"].append(next_m)
          else:
            for pairs in self.share_mem_bank_layers:
                if layer_num == pairs[0]:
                    next_m = self.squash_mem(next_m)
                    state["memory_banks"].append(next_m)        
        else:
            next_m = self.squash_mem(next_m)
            state["memory_banks"].append(next_m) 
        return state
        
    def forward(self, x, state, layer_num):

        residual = x

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x = self.self_attn(input=x, state=state, layer_num=layer_num)

        x = self.dropout_module(x)

        x = residual + x

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.max_memory_size != 0 and self.mem_bank_after:
            state = self.update_mem_banks(state, x, layer_num)

        return x

    def build_self_attention(self, embed_dim, args):
        return AugmentedMemoryMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            tanh_on_mem=args.tanh_on_mem,
            max_memory_size=args.max_memory_size,
            share_mem_bank_layers=args.share_mem_bank_layers,
            mem_bank_size=args.mem_bank_size,
            left_context=args.left_context // args.encoder_stride,
            right_context=args.right_context // args.encoder_stride,
            increase_context=args.increase_context,
            mem_bank_after=args.mem_bank_after,
            shrink_mem_bank=args.shrink_mem_bank,
            shrink_depth=args.shrink_depth,
            shrink_factor=args.shrink_factor
        )


# ------------------------------------------------------------------------------
#   AugmentedMemoryMultiheadAttention
# ------------------------------------------------------------------------------
class AugmentedMemoryMultiheadAttention(MultiheadAttention):
    """
    Augmented Memory Attention from
    Streaming Transformer-based Acoustic Models
    Using Self-attention with Augmented Memory
    https://arxiv.org/abs/2005.08042
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        tanh_on_mem=False,
        memory_dim=None,
        std_scale=0.5,  # 0.5 based on https://arxiv.org/abs/2005.09137
        max_memory_size=-1,
        share_mem_bank_layers=None,
        mem_bank_size=1,
        left_context=0,
        right_context=0,
        increase_context=False,
        mem_bank_after=False,
        shrink_mem_bank=False,
        shrink_depth=3,
        shrink_factor=2,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            kdim,
            vdim,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            self_attention,
            encoder_decoder_attention,
            q_noise,
            qn_block_size,
        )

        self.memory_dim = memory_dim if memory_dim is not None else embed_dim
        self.std_scale = std_scale

        # This Operator was used for factorization in PySpeech
        self.v2e = lambda x: x

        if tanh_on_mem:
            self.squash_mem = torch.tanh
            self.nonlinear_squash_mem = True
        else:
            self.squash_mem = lambda x: x
            self.nonlinear_squash_mem = False

        self.max_memory_size = max_memory_size

        self.share_mem_bank_layers = share_mem_bank_layers
        self.mem_bank_size = mem_bank_size
        self.left_context=left_context
        self.right_context=right_context
        self.increase_context=increase_context
        
        self.mem_bank_size = mem_bank_size
        self.pool = torch.nn.AdaptiveAvgPool1d(self.mem_bank_size)
        self.mem_bank_after = mem_bank_after

        self.shrink_mem_bank = shrink_mem_bank
        if self.shrink_mem_bank:
            self.shrink_factor = shrink_factor
            self.pooltwo = torch.nn.AdaptiveAvgPool1d(self.shrink_factor)
            self.shrink_depth = shrink_depth

    def update_mem_banks(self, state, input, layer_num):
        length, _, _ = input.size()
        if self.increase_context:
            segment = input[0:length]
        else:
            segment = input[self.left_context:length-self.right_context]
        
        segment = segment.transpose(0,2)
        next_m = self.pool(segment)
        next_m = next_m.transpose(0,2)

        if self.share_mem_bank_layers is not None:
          if not any(layer_num in layer for layer in self.share_mem_bank_layers):
            next_m = self.squash_mem(next_m)
            state["memory_banks"].append(next_m)
          else:
            for pairs in self.share_mem_bank_layers:
                if layer_num == pairs[0]:
                    next_m = self.squash_mem(next_m)
                    state["memory_banks"].append(next_m)        
        else:
            next_m = self.squash_mem(next_m)
            state["memory_banks"].append(next_m) 
        return state

    def shrink_banks(self, memory):
        num_banks = len(memory)
        if num_banks > self.shrink_depth:
            self.pooltwo(memory[self.shrink_depth])
        length = self.shrink_depth*self.mem_bank_size + (num_banks - self.shrink_depth)*self.shrink_factor
        return memory, length 

    def forward(self, input, state, layer_num):
        """
        input: Encoder states of current segment with left or right context,
            plus one summarization query

        """

        length, batch_size, _ = input.shape

        memory = state["memory_banks"]
        # TODO: positional embedding on memory

        if self.max_memory_size > -1 and len(memory) > self.max_memory_size:
            memory = memory[-self.max_memory_size :]
            state["memory_banks"] = memory
        
        if self.shrink_mem_bank:
            mem_bank_len = 0
            memory, mem_bank_len = self.shrink_banks(memory)
            state["memory_banks"] = memory
        else:
            mem_bank_len = len(memory)*self.mem_bank_size

        memory_and_input = torch.cat(memory + [input], dim=0)
        input_query = input

        q = self.q_proj(self.v2e(input_query))
        k = self.k_proj(self.v2e(memory_and_input))
        v = self.v_proj(self.v2e(memory_and_input))

        q = (
            q.contiguous()
            .view(-1, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
            * self.scaling
        )
        k = (
            k.contiguous()
            .view(-1, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        v = (
            v.contiguous()
            .view(-1, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        attention_weights = torch.bmm(q, k.transpose(1, 2))
       
        if self.std_scale is not None:
            attention_weights = attention_suppression(attention_weights, self.std_scale)
       
        assert list(attention_weights.shape) == [
            batch_size * self.num_heads,
            length,
            length + mem_bank_len,
        ]

        attention_weights = torch.nn.functional.softmax(
            attention_weights.float(), dim=-1
        ).type_as(attention_weights)
      
        attention_probs = self.dropout_module(attention_weights)

        # [T, T, B, n_head] + [T, B, n_head, d_head] -> [T, B, n_head, d_head]
        attention = torch.bmm(attention_probs, v)

        assert list(attention.shape) == [
            batch_size * self.num_heads,
            length,
            self.head_dim,
        ]

        attention = (
            attention.transpose(0, 1)
            .contiguous()
            .view(length, batch_size, self.embed_dim)
        )
        
        output = self.out_proj(attention)
        
        if self.max_memory_size != 0 and not self.mem_bank_after:
            state = self.update_mem_banks(state, input, layer_num)

        return output


# ------------------------------------------------------------------------------
#   SequenceEncoder
# ------------------------------------------------------------------------------
class SequenceEncoder_no_sum(FairseqEncoder):
    """
    SequenceEncoder encodes sequences.

    More specifically, `src_tokens` and `src_lengths` in `forward()` should
    describe a batch of "complete" sequences rather than segments.

    Segment-by-segment inference can be triggered by `segment_size`:
    1) `segment_size` is None:
        SequenceEncoder treats the input sequence as one single segment.
    2) `segment_size` is not None (some int instead):
        SequenceEncoder does the following:
            1. breaks the input sequence into several segments
            2. inference on each segment and collect the outputs
            3. concatanete segment outputs into the output sequence.
    Note that `segment_size` here shouldn't include additional left/right
    contexts needed, for example if we wish to infer with LC-BLSTM where the
    middle chunk size is 100 and right context is 20, `segment_size` should be
    100.
    """

    def __init__(self, args, module):
        super().__init__(args)

        self.module = module
        self.input_time_axis = 1
        self.output_time_axis = 0
        self.segment_size = args.segment_size
        self.left_context = args.left_context
        self.right_context = args.right_context

    def forward(
        self,
        src_tokens: Tensor,
        src_lengths: Tensor,
        states=None,
    ):
    
        seg_src_tokens_lengths = sequence_to_segments(
            sequence=src_tokens,
            time_axis=self.input_time_axis,
            lengths=src_lengths,
            segment_size=self.segment_size,
            extra_left_context=self.left_context,
            extra_right_context=self.right_context,
        )

        seg_encoder_states_lengths: List[Tuple[Tensor, Tensor]] = []

        for seg_src_tokens, seg_src_lengths in seg_src_tokens_lengths:
            (seg_encoder_states, seg_enc_lengths, states) = self.module(
                seg_src_tokens,
                seg_src_lengths,
                states=states,
            )

            seg_encoder_states_lengths.append((seg_encoder_states, seg_enc_lengths))

        encoder_out, enc_lengths = segments_to_sequence(
            segments=seg_encoder_states_lengths, time_axis=self.output_time_axis
        )

        encoder_padding_mask, _ = lengths_to_encoder_padding_mask(
            enc_lengths, batch_first=True
        )

        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        return {
            "encoder_out": [encoder_out],
            "encoder_padding_mask": [encoder_padding_mask],
            "encoder_embedding": [],
            "encoder_states": [states],
            "src_tokens": [],
            "src_lengths": [],
        }

    def incremental_encode(
        self,
        seg_src_tokens: Tensor,
        seg_src_lengths: Tensor,
        states=None,
    ):
        """
        Different from forward function, this function takes segmented speech
        as input, and append encoder states to previous states
        """
        (seg_encoder_states, seg_enc_lengths, states) = self.module(
            seg_src_tokens,
            seg_src_lengths,
            states=states,
        )
        return seg_encoder_states, seg_enc_lengths, states


# ------------------------------------------------------------------------------
#   Augmented memory model decorator
# ------------------------------------------------------------------------------
def augmented_memory_no_sum(klass):
    class StreamSeq2SeqModel(klass):
        @staticmethod
        def add_args(parser):
            super(StreamSeq2SeqModel, StreamSeq2SeqModel).add_args(parser)
            parser.add_argument(
                "--segment-size", type=int, required=True, help="Length of the segment."
            )
            parser.add_argument(
                "--left-context",
                type=int,
                default=0,
                help="Left context for the segment.",
            )
            parser.add_argument(
                "--right-context",
                type=int,
                default=0,
                help="Right context for the segment.",
            )
            parser.add_argument(
                "--max-memory-size",
                type=int,
                default=-1,
                help="Right context for the segment.",
            )
            parser.add_argument(
                "--increase-context",
                action="store_true",
                default=False,
                help="if True, memory bank calculation uses left and center context",
            )
            parser.add_argument(
                "--share-mem-bank-layers",
                type=json.loads,
                default=None,
                help=":The list of memory bank sharing layers",
            )
            parser.add_argument(
                "--mem-bank-size",
                type=int,
                default=1,
                help="Size of mem_bank",
            )
            parser.add_argument(
                "--mem-bank-after",
                action="store_true",
                default=False,
                help="if True, average after attention",
            )
            parser.add_argument(
                "--tanh-on-mem",
                action="store_true",
                default=True,
                help="if True, squash memory banks",
            )
            parser.add_argument(
                "--shrink-mem-bank",
                action="store_true",
                default=False,
                help="if True, average after attention",
            )
            parser.add_argument(
                "--shrink-factor",
                type=int,
                default=1,
                help="Size of mem_bank",
            )
            parser.add_argument(
                "--shrink-depth",
                type=int,
                default=1,
                help="Size of mem_bank",
            )



    StreamSeq2SeqModel.__name__ = klass.__name__
    return StreamSeq2SeqModel
