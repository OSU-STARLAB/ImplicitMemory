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

class AugmentedMemoryConvTransformerEncoder_no_sum(ConvTransformerEncoder):
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

        self.share_mem_bank_layers = args.share_mem_bank_layers
        self.max_relative_position = getattr(args, "max_relative_position", 0)
        self.mem_bank_size = args.mem_bank_size
        self.shrink_mem_bank = getattr(args, "shrink_mem_bank", False)
        if self.shrink_mem_bank:
            self.shrink_depth = args.shrink_depth
            self.shrink_factor = args.shrink_factor
        self.max_memory_size = args.max_memory_size

        self.left_context_method = args.left_context_method
        if self.left_context_method is not None:
            self.summarization_method = args.summarization_method
            if self.summarization_method == "avg_pool":
                self.summarize = torch.nn.AdaptiveAvgPool1d(self.left_context_after_stride)
            elif self.summarization_method == "max_pool":
                self.summarize = torch.nn.AdaptiveMaxPool1d(self.left_context_after_stride)
            elif self.summarization_method == "linear":
                self.summarize = torch.nn.Linear(self.segment_size, self.left_context_after_stride)
            self.max_token_count = args.max_token_count // args.encoder_stride
            self.max_segment_count = ceil(self.max_token_count / self.segment_size)
            self.shrink_left_context = getattr(args, "shrink_left_context", False)

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

        query_ranges = torch.arange(0, seq_len)
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
        num_banks = len(memory)
        if self.shrink_mem_bank and num_banks > self.shrink_depth:
            length = self.shrink_depth*self.mem_bank_size + (num_banks - self.shrink_depth)*self.shrink_factor
        else:
            length = num_banks*self.mem_bank_size
        return length 

    def update_left_banks(self, memory, input):
        if self.right_context_after_stride != 0 and self.left_context_method == "after_input":
            input = input[:-self.right_context_after_stride]

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

        if self.left_context_method == "after_input" or self.left_context_method == "after_output":
            input = x
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
            mem_bank_len = self.get_membank_len(states[0]["memory_banks"])
            rpe = self.get_relative_position(x, mem_bank_len)
        else:
            rpe = None

        for i, layer in enumerate(self.transformer_layers):
            # x size:
            # (self.left_size + self.segment_size + self.right_size)
            # / self.stride, num_heads, dim
            # TODO: Consider mask here 
            x = layer(x, states[i], i, rpe, self.left_context_after_stride)
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
            
        if self.left_context_method == "after_input":
            left_bank = self.update_left_banks(left_bank, input)
        elif self.left_context_method == "after_output":
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

        self.share_mem_bank_layers = args.share_mem_bank_layers
        self.mem_bank_after = getattr(args, "mem_bank_after", False)

        self.mem_bank_size = args.mem_bank_size
        self.summarization_method = getattr(args, "summarization_method", "avg_pool")
        if self.summarization_method == "avg_pool":
            self.module = torch.nn.AdaptiveAvgPool1d(self.mem_bank_size)
        elif self.summarization_method == "max_pool":
            self.module = torch.nn.AdaptiveMaxPool1d(self.mem_bank_size)
        elif self.summarization_method == "linear":
            self.module = torch.nn.Linear(self.segment_size, self.mem_bank_size)

        self.max_memory_size = args.max_memory_size
        self.increase_context = args.increase_context
        self.tanh_on_mem = getattr(args, "tanh_on_mem", False)
        if self.tanh_on_mem:
            self.squash_mem = torch.tanh
            self.nonlinear_squash_mem = True
        else:
            self.squash_mem = lambda x: x
            self.nonlinear_squash_mem = False
        

    def update_mem_banks(self, state, input, layer_num):
        length, _, _ = input.size()
        if self.segment_size == length - (self.left_context + self.right_context):
            if self.increase_context:
                segment = input[0:length]
            else:
                segment = input[self.left_context:length-self.right_context]

            segment = segment.transpose(0,2)
            next_m = self.module(segment)
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
        
    def forward(self, x, state, layer_num, rpe, left_context):
        self.left_context = left_context
        
        residual = x

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x = self.self_attn(input=x, state=state, layer_num=layer_num, rpe=rpe)

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
            tanh_on_mem=getattr(args, "tanh_on_mem", False),
            max_memory_size=args.max_memory_size,
            share_mem_bank_layers=args.share_mem_bank_layers,
            mem_bank_size=args.mem_bank_size,
            left_context=args.left_context // args.encoder_stride,
            right_context=args.right_context // args.encoder_stride,
            increase_context=args.increase_context,
            mem_bank_after=getattr(args, "mem_bank_after", False),
            shrink_mem_bank=getattr(args, "shrink_mem_bank", False),
            shrink_depth=getattr(args, "shrink_depth",0),
            shrink_factor=getattr(args, "shrink_factor", 0),
            max_relative_position=getattr(args, "max_relative_position", 0),
            segment_size=args.segment_size // args.encoder_stride,
            summarization_method=getattr(args, "summarization_method", "avg_pool"),
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
        max_relative_position=0,
        segment_size=0,
        summarization_method="avg_pool"
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
        self.left_context = left_context
        self.right_context = right_context
        self.increase_context = increase_context
        self.segment_size = segment_size
        
        self.mem_bank_size = mem_bank_size
        self.mem_bank_after = mem_bank_after

        self.summarization_method = summarization_method
        if self.summarization_method == "avg_pool":
            self.module = torch.nn.AdaptiveAvgPool1d(self.mem_bank_size)
        elif self.summarization_method == "max_pool":
            self.module = torch.nn.AdaptiveMaxPool1d(self.mem_bank_size)
        elif self.summarization_method == "linear":
            self.module = torch.nn.Linear(self.segment_size, self.mem_bank_size)

        self.shrink_mem_bank = shrink_mem_bank
        if self.shrink_mem_bank:
            self.shrink_factor = shrink_factor
            if self.summarization_method == "avg_pool":
                self.moduletwo = torch.nn.AdaptiveAvgPool1d(self.shrink_factor)
            elif self.summarization_method == "max_pool":
                self.moduletwo = torch.nn.AdaptiveMaxPool1d(self.shrink_factor)
            elif self.summarization_method == "linear":
                self.moduletwo = torch.nn.Linear(self.segment_size, self.shrink_factor)
            self.shrink_depth = shrink_depth
        
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

    def update_mem_banks(self, state, input, layer_num):
        length, _, _ = input.size()
        if self.segment_size == length - (self.left_context + self.right_context):
            if self.increase_context:
                segment = input[0:length]
            else:
                segment = input[self.left_context:length-self.right_context]
            
            segment = segment.transpose(0,2)
            next_m = self.module(segment)
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
            pos = num_banks - self.shrink_depth - 1
            segment = memory[pos].transpose(0,2)
            segment = self.moduletwo(segment)
            memory[pos] = segment.transpose(0,2)
            length = self.shrink_depth*self.mem_bank_size + (num_banks - self.shrink_depth)*self.shrink_factor
        else:
            length = num_banks*self.mem_bank_size
        return memory, length 


    def forward(self, input, state, layer_num, rpe):
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
        super().__init__(None)

        self.module = module
        self.input_time_axis = 1
        self.output_time_axis = 0
        self.segment_size = args.segment_size
        self.left_context = args.left_context
        self.right_context = args.right_context
        self.left_context_method = getattr(args, "left_context_method", None)
        if self.left_context_method is not None:
            self.max_token_count = getattr(args, "max_token_count", 0) 
            self.max_segment_count = ceil(self.max_token_count / self.segment_size)
            self.shrink_left_context = getattr(args, "shrink_left_context", False)

        self.summarization_method = args.summarization_method
        if self.summarization_method == "avg_pool":
            self.summarize = torch.nn.AdaptiveAvgPool1d(self.left_context)
        elif self.summarization_method == "max_pool":
            self.summarize = torch.nn.AdaptiveMaxPool1d(self.left_context)
        elif self.summarization_method == "linear":
            self.summarize = torch.nn.Linear(self.segment_size, self.left_context)

    def update_left_banks(self, memory, input):
        if self.right_context != 0:
            input = input[:,:-self.right_context]

        memory.append(input)

        if len(memory) > self.max_segment_count:
            memory.pop(0)
        return memory

    def add_left_context(self, memory, input, mem_size, seg_src_lengths):
        if mem_size > 0:
            left_context = memory[0]

            for i in range(1, mem_size):
                segment = memory[i]
                left_context = torch.cat([left_context] + [segment], dim=self.input_time_axis)
            
            left_context_size = left_context.size()[self.input_time_axis]
        
            if self.left_context < left_context_size:
                if left_context_size > self.max_token_count:
                    left_context = left_context[left_context_size-self.max_token_count:]
                left_context = left_context.transpose(self.input_time_axis,2)
                left_context = self.summarize(left_context)
                left_context = left_context.transpose(self.input_time_axis,2)
                seg_src_lengths = seg_src_lengths + self.left_context
                left_context_size = self.left_context
            else:
                seg_src_lengths = seg_src_lengths + left_context_size

            input = torch.cat([left_context] + [input], dim=self.input_time_axis)
        else:
            left_context_size = 0

        return input, seg_src_lengths, left_context_size 

    def forward(
        self,
        src_tokens: Tensor,
        src_lengths: Tensor,
        states=None,
    ):

        if self.left_context_method is not None:
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
            if self.left_context_method is not None:
                if self.left_context_method == "before_input":
                    src_tokens = seg_src_tokens
                    mem_size = len(left_bank)
                    seg_src_tokens, seg_src_lengths, left_context_size = self.add_left_context(left_bank, seg_src_tokens, mem_size, seg_src_lengths)
                else:
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

            if self.left_context_method == "before_input":
                left_bank = self.update_left_banks(left_bank, src_tokens)

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
                default=False,
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
                default=0,
                help="Size of mem_bank",
            )
            parser.add_argument(
                "--max-relative-position",
                type=int,
                default=0,
                help="Relative Position",
            )
            parser.add_argument(
                "--summarization-method", 
                default="avg_pool",
                choices=["avg_pool", "max_pool", "linear"],
                help="Summarization method"
            )
            parser.add_argument(
                "--left-context-method", 
                default=None,
                choices=["after_output", "after_input", "before_input"],
                help="Summarization method"
            )
            parser.add_argument(
                "--max-token-count",
                type=int,
                default=-1,
                help="Right context for the segment.",
            )
            
    StreamSeq2SeqModel.__name__ = klass.__name__
    return StreamSeq2SeqModel