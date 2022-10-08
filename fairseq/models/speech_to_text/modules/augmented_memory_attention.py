# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from math import ceil
from turtle import left
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

# ------------------------------------------------------------------------------
#   AugmentedMemoryConvTransformerEncoder
# ------------------------------------------------------------------------------


class RelativePositionEmbedding(nn.Module):
    """
    Implementation according to https://arxiv.org/abs/1803.02155
    """

    def __init__(self, head_dim, max_position, norm_init=True):
        super().__init__()
        self.head_dim = head_dim
        self.max_position = max_position
        self.embeddings = nn.Parameter(torch.Tensor(max_position * 2 + 1, head_dim))
        if norm_init:
            nn.init.xavier_normal_(self.embeddings)
        else:
            nn.init.xavier_uniform_(self.embeddings)

    def forward(self, input: Tensor):
        output = nn.functional.embedding(input.long(), self.embeddings)
        return output

class AugmentedMemoryConvTransformerEncoder(ConvTransformerEncoder):
    def __init__(self, args):
        super().__init__(args)

        args.encoder_stride = 4
        self.encoder_stride = args.encoder_stride

        self.left_context_after_stride = args.left_context // args.encoder_stride
        self.right_context_after_stride = args.right_context // args.encoder_stride
        self.segment_size = args.segment_size // args.encoder_stride

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
        self.max_relative_position = getattr(args, "max_relative_position", 0)
        self.max_memory_size = args.max_memory_size

        self.variable_left_context_method = getattr(args, "variable_left_context_method", None)
        self.encoder_left_context = getattr(args, "encoder_left_context", False)
        self.left_compression_factor = getattr(args, "left_compression_factor", 1)
        self.max_token_count = self.left_compression_factor*self.left_context_after_stride
        self.max_segment_count = ceil(self.max_token_count/self.segment_size) + 1
        self.summarize = torch.nn.AvgPool1d(kernel_size=self.left_compression_factor, stride=self.left_compression_factor, padding=0)

    def initialize_states(self, states):
        if states is None:
            # Creates states;
            states = [{"memory_banks": [], "encoder_states": None} for i in range(len(self.transformer_layers))]
            if self.share_mem_bank_layers is not None:
                # Initializes Memory banks
                rows = len(self.share_mem_bank_layers)
                for i in range(rows):
                    cols = len(self.share_mem_bank_layers[i])
                    for j in range(cols):
                        states[self.share_mem_bank_layers[i][j]]["memory_banks"] = states[self.share_mem_bank_layers[i][0]]["memory_banks"]
        return states

    def get_relative_position(
        self,
        input,
        mem_size: int,
    ):

        seq_len, bsz, x_dim = input.shape

        query_ranges = torch.arange(0, seq_len+1)
        key_ranges = torch.arange(-mem_size, seq_len)
        
        distance = key_ranges[None, :] - query_ranges[:, None] 
        distance_clamp = (
            torch.clamp(distance, -self.max_relative_position, self.max_relative_position)
            + self.max_relative_position
        )
        distance_clamp = distance_clamp.to(input.device).long().detach()
        return distance_clamp

    def get_membank_len(self, memory):
        if self.max_memory_size > -1 and len(memory) > self.max_memory_size:
            memory = memory[-self.max_memory_size :]
        return len(memory)

    def update_memory(self, memory, input):
        memory.append(input)

        if len(memory) > self.max_segment_count:
            memory.pop(0)
        return memory

    def add_memory(self, memory, input, src_lengths, old_left_context_size):
        mem_size = len(memory)
        if mem_size > 0 and self.left_context_after_stride != 0:
            left_context = torch.cat(memory, dim=0)
            if old_left_context_size != 0:
                left_context = left_context[:-old_left_context_size]
            left_context_size = left_context.size(0)
            
            if left_context_size - self.max_token_count > 0:
                left_context = left_context[left_context_size-self.max_token_count:]
            left_context = self.compress(left_context)
            left_context_size = left_context.size(0)
            src_lengths = src_lengths + left_context_size
            left_context_size = old_left_context_size + left_context_size

            input = torch.cat([left_context] + [input], dim=0)
        else:
            left_context_size = old_left_context_size

        return input, src_lengths, left_context_size

    def compress(self, left_context):
        left_context = left_context.transpose(0,2)
        left_context = self.summarize(left_context)
        left_context = left_context.transpose(0,2)
        return left_context

    def forward(self, src_tokens, src_lengths, left_context_size, states=None, left_memory=None, prev_output=None):
        """Encode input sequence.
        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        left_context_size = left_context_size // self.encoder_stride

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

        if self.encoder_left_context:
            if self.variable_left_context_method == "output":
                x_dim = x.size(0)
                if x_dim < self.segment_size and prev_output is not None:
                    left_context_size = self.segment_size-x_dim
                    prev_output = prev_output[self.segment_size-left_context_size:]
                    x = torch.cat([prev_output] + [x], dim=0)
                    input_lengths = input_lengths + left_context_size
            x, input_lengths, left_context_size = self.add_memory(left_memory, x, input_lengths, left_context_size)

        encoder_padding_mask, _ = lengths_to_encoder_padding_mask(
            input_lengths, batch_first=True
        )
        
        if self.max_relative_position <= 0:
            positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
            x += positions

        x = F.dropout(x, p=self.dropout, training=self.training)

        # State to store memory banks etc.
        states = self.initialize_states(states)
        if self.max_relative_position > 0:
            mem_bank_len = self.get_membank_len(states[0]["memory_banks"])
            rpe = self.get_relative_position(x, mem_bank_len)
        else:
            rpe = None

        for i, layer in enumerate(self.transformer_layers):
            # x size:
            # (self.left_size + self.segment_size + self.right_size)
            # / self.stride, num_heads, dim
            # TODO: Consider mask here 
            x = layer(x, states[i], i, rpe, left_context_size)
            if self.right_context_after_stride != 0:
                states[i]["encoder_states"] = x[left_context_size : -self.right_context_after_stride]
            else:
                states[i]["encoder_states"] = x[left_context_size : ]

        if self.right_context_after_stride != 0:
            lengths = (
                (
                    ~encoder_padding_mask[:, left_context_size : -self.right_context_after_stride]
                )
                .sum(dim=1, keepdim=True)
                .long()
            )
        else:
            lengths = (
                (
                    ~encoder_padding_mask[:, left_context_size :]
                )
                .sum(dim=1, keepdim=True)
                .long()
            )
            
        if self.encoder_left_context:
            left_memory = self.update_memory(left_memory, states[-1]["encoder_states"])

        return states[-1]["encoder_states"], lengths, states, left_memory, states[-1]["encoder_states"]

# ------------------------------------------------------------------------------
#   AugmentedMemoryTransformerEncoderLayer
# ------------------------------------------------------------------------------
class AugmentedMemoryTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)

        self.left_context = args.left_context // args.encoder_stride
        self.right_context = args.right_context // args.encoder_stride
        self.increase_context = args.increase_context

        self.share_mem_bank_layers = args.share_mem_bank_layers

    def forward(self, x, state, layer_num, rpe, new_left_context):
        self.left_context = new_left_context

        length, batch_size, x_dim = x.size()

        residual = x

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # TODO reseach new sum_query method
        if self.increase_context:
            seg_start = 0
        else:
            seg_start = self.left_context
        seg_end = length - self.right_context

        if seg_start < seg_end:
            summarization_query = torch.mean(x[seg_start:seg_end], keepdim=True, dim=0)
        else:
            summarization_query = x.new_zeros(1, batch_size, x_dim)

        x = torch.cat([x, summarization_query], dim=0)

        x = self.self_attn(input_and_summary=x, state=state, layer_num=layer_num, rpe=rpe)

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

        return x

    def build_self_attention(self, embed_dim, args):
        return AugmentedMemoryMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            tanh_on_mem=getattr(args, "tanh-on-mem", False),
            max_memory_size=args.max_memory_size,
            share_mem_bank_layers=args.share_mem_bank_layers,
            max_relative_position=getattr(args, "max_relative_position", 0)
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
        disable_mem_on_mem_attn=True,
        share_mem_bank_layers=None,
        max_relative_position=0,
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
        self.disable_mem_on_mem_attn = disable_mem_on_mem_attn

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

        if max_relative_position > 0:
            self.use_rpe = True
            self.rpe_k = RelativePositionEmbedding(
                head_dim=embed_dim // num_heads,
                max_position=max_relative_position,
            )
            self.rpe_v = RelativePositionEmbedding(
                head_dim=embed_dim // num_heads,
                max_position=max_relative_position,
            )
        else:
            self.use_rpe = False
            self.rpe_k = None
            self.rpe_v = None


    def update_mem_banks(self, state, output_and_memory, layer_num):
        if self.share_mem_bank_layers is not None:
          if not any(layer_num in layer for layer in self.share_mem_bank_layers):
            next_m = output_and_memory[-1:]
            next_m = self.squash_mem(next_m)
            state["memory_banks"].append(next_m)
          else:
            for pairs in self.share_mem_bank_layers:
                if layer_num == pairs[0]:
                    next_m = output_and_memory[-1:]
                    next_m = self.squash_mem(next_m)
                    state["memory_banks"].append(next_m)        
        else:
            next_m = output_and_memory[-1:]
            next_m = self.squash_mem(next_m)
            state["memory_banks"].append(next_m) 
        return state

    def forward(self, input_and_summary, state, layer_num, rpe):
        """
        input: Encoder states of current segment with left or right context,
            plus one summarization query

        """

        length, batch_size, _ = input_and_summary.shape
        length = length - 1  # not include sum_query, last index

        memory = state["memory_banks"]
        # TODO: positional embedding on memory

        if self.max_memory_size > -1 and len(memory) > self.max_memory_size:
            memory = memory[-self.max_memory_size :]
            state["memory_banks"] = memory

        memory_and_input = torch.cat(memory + [input_and_summary[:-1]], dim=0)
        input_and_sum_query = input_and_summary

        q = self.q_proj(self.v2e(input_and_sum_query))
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
        
        if self.use_rpe and rpe is not None and self.rpe_k is not None:
            r_k = self.rpe_k(rpe)
            # [q, B*h, d] * [q, k, d] -> [B*h, q, k]
            attention_weights_rpe = torch.matmul(
                q.transpose(0, 1), r_k.transpose(1, 2)
            ).transpose(0, 1)
            attention_weights = attention_weights + attention_weights_rpe

        if self.disable_mem_on_mem_attn:
            attention_weights = self.suppress_mem_on_mem_attention(
                batch_size, self.num_heads, len(memory), attention_weights
            )
       
        if self.std_scale is not None:
            attention_weights = attention_suppression(attention_weights, self.std_scale)
       
        assert list(attention_weights.shape) == [
            batch_size * self.num_heads,
            length + 1,
            length + len(memory),
        ]

        attention_weights = torch.nn.functional.softmax(
            attention_weights.float(), dim=-1
        ).type_as(attention_weights)
      
        attention_probs = self.dropout_module(attention_weights)

        # [T, T, B, n_head] + [T, B, n_head, d_head] -> [T, B, n_head, d_head]
        attention = torch.bmm(attention_probs, v)

        if self.use_rpe and rpe is not None and self.rpe_v is not None:
            r_v = self.rpe_v(rpe)
            attention_rpe = torch.matmul(
                attention_probs.transpose(0, 1), r_v
            ).transpose(0, 1)

            attention = attention + attention_rpe

        assert list(attention.shape) == [
            batch_size * self.num_heads,
            length + 1,
            self.head_dim,
        ]

        attention = (
            attention.transpose(0, 1)
            .contiguous()
            .view(length + 1, batch_size, self.embed_dim)
        )
        
        output_and_memory = self.out_proj(attention)
        
        output = output_and_memory[:-1]
        
        if self.max_memory_size != 0:
            state = self.update_mem_banks(state, output_and_memory, layer_num)

        return output

    def suppress_mem_on_mem_attention(
        self, B: int, num_heads: int, mem_size: int, attention_weight: Tensor
    ):
        """
        Arguments:
            - B: batch size
            - num_heads: number of attention heads
            - mem_size: size of memory bank
            - attention_weight: a [B*num_heads, T + 1, T + mem_size] vector

        Return:
            modified attention_weight with [B*num_heads, -1, :mem_size] = -inf
        """
        attention_weight[:, -1, :mem_size] = float("-inf")
        return attention_weight


# ------------------------------------------------------------------------------
#   SequenceEncoder
# ------------------------------------------------------------------------------
class SequenceEncoder(FairseqEncoder):
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
        super().__init__(None)

        self.module = module
        self.input_time_axis = 1
        self.output_time_axis = 0
        self.segment_size = args.segment_size
        self.left_context = args.left_context
        self.right_context = args.right_context
        self.variable_left_context_method = getattr(args, "variable_left_context_method", None)
        self.encoder_left_context = getattr(args, "encoder_left_context", False)
        self.left_compression_factor = getattr(args, "left_compression_factor", 1) 
        self.max_token_count = self.left_compression_factor*self.left_context
        self.max_segment_count = ceil(self.max_token_count/self.segment_size) + 1
        self.summarize = torch.nn.AvgPool1d(kernel_size=self.left_compression_factor, stride=self.left_compression_factor, padding=0)

    def update_memory(self, memory, input):
        memory.append(input)

        if len(memory) > self.max_segment_count:
            memory.pop(0)
        return memory

    def add_memory(self, memory, input, src_lengths, old_left_context_size):
        mem_size = len(memory)
        if mem_size > 0 and self.left_context != 0:
            left_context = torch.cat(memory, dim=self.input_time_axis)
            if old_left_context_size != 0:
                left_context = left_context[:,:-old_left_context_size]
            left_context_size = left_context.size(self.input_time_axis)

            if left_context_size - self.max_token_count > 0:
                left_context = left_context[:, left_context_size-self.max_token_count:]
            left_context = self.compress(left_context)
            left_context_size = left_context.size(self.input_time_axis)
            src_lengths = src_lengths + left_context_size
            left_context_size = old_left_context_size + left_context_size

            input = torch.cat([left_context] + [input], dim=self.input_time_axis)
        else:
            left_context_size = old_left_context_size

        return input, src_lengths, left_context_size

    def compress(self, left_context):
        left_context = left_context.transpose(self.input_time_axis,2)
        left_context = self.summarize(left_context)
        left_context = left_context.transpose(self.input_time_axis,2)
        return left_context   

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
            extra_left_context=0,
            extra_right_context=self.right_context,
        )

        left_memory = []
        seg_encoder_states_lengths: List[Tuple[Tensor, Tensor]] = []

        prev_input = None
        prev_output = None
        for seg_src_tokens, seg_src_lengths in seg_src_tokens_lengths:
            src_tokens = seg_src_tokens
            if self.right_context != 0:
                src_tokens = src_tokens[:,:-self.right_context]
            left_context_size = 0
            if self.variable_left_context_method == "input":
                seg_src_tokens_dim = seg_src_tokens.size(self.input_time_axis)
                if seg_src_tokens_dim < self.segment_size + self.right_context and prev_input is not None:
                    left_context_size = self.segment_size + self.right_context - seg_src_tokens_dim
                    prev_input = prev_input[:,self.segment_size-left_context_size:]
                    seg_src_tokens = torch.cat([prev_input] + [seg_src_tokens], dim=self.input_time_axis)
                    seg_src_lengths = seg_src_lengths + left_context_size
            
            if not self.encoder_left_context:
                seg_src_tokens, seg_src_lengths, left_context_size = self.add_memory(left_memory, seg_src_tokens, seg_src_lengths, left_context_size)

            (seg_encoder_states, seg_enc_lengths, states, left_memory, prev_output) = self.module(
            seg_src_tokens,
            seg_src_lengths,
            left_context_size,
            states=states,
            left_memory=left_memory,
            prev_output=prev_output,
            )

            seg_encoder_states_lengths.append((seg_encoder_states, seg_enc_lengths))
            if not self.encoder_left_context:
                left_memory = self.update_memory(left_memory, src_tokens)
            prev_input = src_tokens
            
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
def augmented_memory(klass):
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
                "--tanh-on-mem",
                action="store_true",
                default=False,
                help="if True, squash memory banks",
            )
            parser.add_argument(
                "--variable-left-context-method",
                default=None,
                choices=["input", "output"],
                help="Summarization method"
            )
            parser.add_argument(
                "--encoder-left-context",
                action="store_true",
                default=False,
                help="if True, squash memory banks",
            )
            parser.add_argument(
                "--left-compression-factor",
                type=int,
                default=1,
                help="Right context for the segment.",
            )
            parser.add_argument(
                "--max-relative-position",
                type=int,
                default=0,
                help="Relative Position",
            )

    StreamSeq2SeqModel.__name__ = klass.__name__
    return StreamSeq2SeqModel
