# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from math import ceil
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

class AugmentedMemoryConvTransformerEncoder_fill_sum_left(ConvTransformerEncoder):
    def __init__(self, args):
        super().__init__(args)

        args.encoder_stride = self.stride()
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

        self.max_relative_position = getattr(args, "max_relative_position", 0)
        self.max_memory_size = args.max_memory_size
        self.encoder_left_context = getattr(args, "encoder_left_context", False)
        if self.encoder_left_context:
            self.max_token_count = args.max_token_count // args.encoder_stride
            self.max_segment_count = ceil(self.max_token_count / self.segment_size)
            self.summarize = torch.nn.AdaptiveAvgPool1d(self.left_context_after_stride)

    def stride(self):
        # Hard coded here. Should infer from convs in future
        stride = 4
        return stride

    def initialize_states(self, states):
        if states is None:
            # Creates states;
            states = [{"memory_banks": [], "encoder_states": None} for i in range(len(self.transformer_layers))]
        return states
    def get_relative_position(
        self,
        input,
        mem_size: int,
    ):

        seq_len, bsz, x_dim = input.shape

        query_ranges = torch.arange(0, seq_len)
        key_ranges = torch.arange(-mem_size, seq_len)
        
        distance = key_ranges[None, :] - query_ranges[:, None] 
        distance_clamp = (
            torch.clamp(distance, -self.max_relative_position, self.max_relative_position)
            + self.max_relative_position
        )
        distance_clamp = distance_clamp.to(input.device).long().detach()
        return distance_clamp

    def update_left_banks(self, memory, input):
        memory.append(input)

        if len(memory) > self.max_segment_count:
            memory.pop(0)
        return memory

    def add_left_context(self, memory, input, mem_size, src_lengths):
        if mem_size > 0:
            left_context = memory[0]

            for i in range(1, mem_size):
                segment = memory[i]
                left_context = torch.cat([left_context] + [segment], dim=0)
            
            left_context_size = left_context.size()[0]
        
            if self.left_context_after_stride < left_context_size:
                if left_context_size > self.max_token_count:
                    left_context = left_context[left_context_size-self.max_token_count:]
                left_context = left_context.transpose(0,2)
                left_context = self.summarize(left_context)
                left_context = left_context.transpose(0,2)
                src_lengths = src_lengths + self.left_context_after_stride
                left_context_size = self.left_context_after_stride
            else:
                src_lengths = src_lengths + left_context_size

            input = torch.cat([left_context] + [input], dim=0)
        else:
            left_context_size = 0

        return input, src_lengths, left_context_size

    def forward(self, src_tokens, src_lengths, left_context_size, left_bank, states=None):
        """Encode input sequence.
        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        self.left_context_after_stride = left_context_size // self.encoder_stride

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
            x, input_lengths, self.left_context_after_stride = self.add_left_context(left_bank, x, len(left_bank), input_lengths)

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
            if len(states[0]["memory_banks"]) == 0:
                mem_bank_len = 0
            else:
                mem_bank_len = self.max_memory_size
            rpe = self.get_relative_position(x, mem_bank_len)
        else:
            rpe = None

        for i, layer in enumerate(self.transformer_layers):
            # x size:
            # (self.left_size + self.segment_size + self.right_size)
            # / self.stride, num_heads, dim
            # TODO: Consider mask here 
            x = layer(x, states[i], rpe, self.left_context_after_stride)
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

        if self.encoder_left_context:
            left_bank = self.update_left_banks(left_bank, states[-1]["encoder_states"])

        return states[-1]["encoder_states"], lengths, left_bank, states


# ------------------------------------------------------------------------------
#   AugmentedMemoryTransformerEncoderLayer
# ------------------------------------------------------------------------------
class AugmentedMemoryTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)

        self.left_context = args.left_context // args.encoder_stride
        self.right_context = args.right_context // args.encoder_stride
        self.segment_size = args.segment_size // args.encoder_stride

        self.mem_bank_after = getattr(args, "mem_bank_after", False)

        self.max_memory_size = args.max_memory_size

        self.max_memory_size = args.max_memory_size
        self.tanh_on_mem = getattr(args, "tanh_on_mem", False)
        if self.tanh_on_mem:
            self.squash_mem = torch.tanh
            self.nonlinear_squash_mem = True
        else:
            self.squash_mem = lambda x: x
            self.nonlinear_squash_mem = False
        
    def update_mem_banks(self, state, input):
        length, _, _ = input.size()
        if self.segment_size == length - (self.left_context + self.right_context):
            input = input[self.left_context:length-self.right_context]
            input = self.squash_mem(input)
            state["memory_banks"].append(input) 
        return state
        
    def forward(self, x, state, rpe, left_context):
        self.left_context = left_context

        residual = x

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x = self.self_attn(input=x, state=state, rpe=rpe)

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
            state = self.update_mem_banks(state, x)

        return x

    def build_self_attention(self, embed_dim, args):
        return AugmentedMemoryMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            tanh_on_mem=getattr(args, "tanh_on_mem", False),
            max_memory_size=args.max_memory_size,
            left_context=args.left_context // args.encoder_stride,
            right_context=args.right_context // args.encoder_stride,
            mem_bank_after=getattr(args, "mem_bank_after", False),
            max_relative_position=getattr(args, "max_relative_position", 0),
            segment_size=args.segment_size // args.encoder_stride,
            mem_bank_start=getattr(args, "mem_bank_start", 0) // args.encoder_stride,
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
        left_context=0,
        right_context=0,
        mem_bank_after=False,
        max_relative_position=0,
        segment_size=0,
        mem_bank_start = 0
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

        self.left_context = left_context
        self.right_context = right_context
        self.segment_size = segment_size
        
        self.mem_bank_after = mem_bank_after
        
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

        self.mem_bank_start = mem_bank_start

    def update_mem_banks(self, state, input):
        length, _, _ = input.size()
        if self.segment_size == length - (self.left_context + self.right_context):
            input = input[self.left_context:length-self.right_context]
            input = self.squash_mem(input)
            state["memory_banks"].append(input) 
        return state

    def add_memory(self, memory):
        mem_size = len(memory)
        new_memory = []
        if mem_size > 0:
            if mem_size == 1:
                for i in range(self.max_memory_size, 0, -1):
                    if (self.segment_size - self.mem_bank_start) % i == 0:
                        summarize = torch.nn.AdaptiveAvgPool1d(i)
                        new_memory = self.compress_memory(memory, 0, new_memory, summarize, mem_size)
                        break
            else:
                zp = mem_size - (self.max_memory_size % mem_size)
                pp = self.max_memory_size // mem_size
                for i in range(mem_size):
                    if i >= zp and self.segment_size % (pp+1) == 0:
                        summarize = torch.nn.AdaptiveAvgPool1d(pp+1)
                    elif i >= zp and (self.segment_size - self.mem_bank_start) % (pp+1) == 0 and i == mem_size - 1:
                        summarize = torch.nn.AdaptiveAvgPool1d(pp+1)
                    elif self.segment_size % pp == 0:
                        summarize = torch.nn.AdaptiveAvgPool1d(pp)
                    else:
                        new_pp = pp - self.segment_size % pp
                        summarize = torch.nn.AdaptiveAvgPool1d(new_pp)
                    new_memory = self.compress_memory(memory, i, new_memory, summarize, mem_size)

        return new_memory

    def compress_memory(self, memory, index, new_memory, summarize, mem_size):
        bank = memory[index]
        if index == mem_size-1 and self.mem_bank_start != 0:
            bank = bank[:-self.mem_bank_start]
        bank = bank.transpose(0,2)
        bank = summarize(bank)
        bank = bank.transpose(0,2)
        if new_memory == []:
            new_memory = bank
        else:
            new_memory = torch.cat([new_memory] + [bank], dim=0)
        return new_memory

    def forward(self, input, state, rpe):
        """
        input: Encoder states of current segment with left or right context,
            plus one summarization query

        """

        length, batch_size, _ = input.shape

        memory = state["memory_banks"]

        if self.max_memory_size > -1 and len(memory) > self.max_memory_size:
            memory = memory[-self.max_memory_size :]
            state["memory_banks"] = memory

        new_memory = self.add_memory(memory)

        if new_memory != []:
            memory_and_input = torch.cat([new_memory] + [input], dim=0)
            mem_bank_len = new_memory.size(0)
        else:
            memory_and_input = input
            mem_bank_len = 0

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
       
        if self.use_rpe and rpe is not None and self.rpe_k is not None:
            r_k = self.rpe_k(rpe)
            # [q, B*h, d] * [q, k, d] -> [B*h, q, k]
            attention_weights_rpe = torch.matmul(
                q.transpose(0, 1), r_k.transpose(1, 2)
            ).transpose(0, 1)
            attention_weights = attention_weights + attention_weights_rpe

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

        if self.use_rpe and rpe is not None and self.rpe_v is not None:
            r_v = self.rpe_v(rpe)
            attention_rpe = torch.matmul(
                attention_probs.transpose(0, 1), r_v
            ).transpose(0, 1)

            attention = attention + attention_rpe
                
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
            state = self.update_mem_banks(state, input)

        return output


# ------------------------------------------------------------------------------
#   SequenceEncoder
# ------------------------------------------------------------------------------
class SequenceEncoder_fill_sum_left(FairseqEncoder):
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
        self.encoder_left_context = getattr(args, "encoder_left_context", False)

    def forward(
        self,
        src_tokens: Tensor,
        src_lengths: Tensor,
        states=None,
    ):

        if self.encoder_left_context:
            left_bank = []
            seg_src_tokens_lengths = sequence_to_segments(
                sequence=src_tokens,
                time_axis=self.input_time_axis,
                lengths=src_lengths,
                segment_size=self.segment_size,
                extra_left_context=0,
                extra_right_context=self.right_context,
            )
        else:
            left_bank = None
            seg_src_tokens_lengths = sequence_to_segments(
                sequence=src_tokens,
                time_axis=self.input_time_axis,
                lengths=src_lengths,
                segment_size=self.segment_size,
                extra_left_context=self.left_context,
                extra_right_context=self.right_context,
            )
            count = 0

        seg_encoder_states_lengths: List[Tuple[Tensor, Tensor]] = []

        for seg_src_tokens, seg_src_lengths in seg_src_tokens_lengths:
            if self.encoder_left_context:
                left_context_size = self.left_context
                (seg_encoder_states, seg_enc_lengths, left_bank, states) = self.module(
                        seg_src_tokens,
                        seg_src_lengths,
                        left_context_size,
                        left_bank,
                        states=states,
                )
            else:
                left = self.left_context - count*self.segment_size
                if left > 0:
                    seg_src_tokens = seg_src_tokens[:, left:, :]
                    seg_src_lengths = seg_src_lengths - left
                else:
                    left=0
                count+=1
                (seg_encoder_states, seg_enc_lengths, left_bank, states) = self.module(
                    seg_src_tokens,
                    seg_src_lengths,
                    self.left_context - left,
                    left_bank,
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
def augmented_memory_fill_sum_left(klass):
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
                "--mem-bank-after",
                action="store_true",
                default=False,
                help="if True, average after attention",
            )
            parser.add_argument(
                "--tanh-on-mem",
                action="store_true",
                default=False,
                help="if True, squash memory banks",
            )
            parser.add_argument(
                "--max-relative-position",
                type=int,
                default=0,
                help="Relative Position",
            )
            parser.add_argument(
                "--mem-bank-start",
                type=int,
                default=0,
                help="Start position",
            )
            parser.add_argument(
                "--max-token-count",
                type=int,
                default=-1,
                help="Right context for the segment.",
            )
            parser.add_argument(
                "--encoder-left-context",
                action="store_true",
                default=False,
                help="if True, squash memory banks",
            )


    StreamSeq2SeqModel.__name__ = klass.__name__
    return StreamSeq2SeqModel
