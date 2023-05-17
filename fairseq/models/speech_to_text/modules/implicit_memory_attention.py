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

# ------------------------------------------------------------------------------
#   ImplicitMemoryTransformerEncoder
# ------------------------------------------------------------------------------

class ImplicitMemoryTransformerEncoder(ConvTransformerEncoder):
    def __init__(self, args):
        super().__init__(args)

        args.encoder_stride = 4
        self.encoder_stride = args.encoder_stride

        self.left_context = args.left_context // args.encoder_stride
        self.right_context = args.right_context // args.encoder_stride
        self.segment_size = args.segment_size // args.encoder_stride

        self.transformer_layers = nn.ModuleList([])

        # Layer sharing code
        encoder_weight_share_list = getattr(args, "share_encoder_ffn_attn_layer", None)
        if encoder_weight_share_list is None:
            encoder_weight_share_list = []
        else:
            shared_weights_layer = ImplicitMemoryTransformerEncoderLayer(args)
        print(f"Encoder: Sharing layers: {encoder_weight_share_list}")
        for layer_idx in range(args.encoder_layers):
            if layer_idx+1 in encoder_weight_share_list:
                self.transformer_layers.append(shared_weights_layer)
            else:
                self.transformer_layers.append(ImplicitMemoryTransformerEncoderLayer(args))

        self.share_mem_bank_layers = getattr(args, "share_mem_bank_layers", None)
        self.max_relative_position = getattr(args, "max_relative_position", 0)


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
        mem_size
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

    def get_memory_size(self, memory):
        mem_size = len(memory)
        if mem_size*self.segment_size > self.left_context:
            return self.left_context
        elif mem_size == 0:
            return 0
        else:
            return mem_size*self.segment_size

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
        
        if self.max_relative_position <= 0:
            positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
            x += positions

        x = F.dropout(x, p=self.dropout, training=self.training)

        # State to store memory banks etc.
        states = self.initialize_states(states)
        if self.max_relative_position > 0:
            mem_size = self.get_memory_size(states[0]["memory_banks"])
            rpe = self.get_relative_position(x, mem_size)
        else:
            rpe = None

        for i, layer in enumerate(self.transformer_layers):
            # x size:
            # (self.left_size + self.segment_size + self.right_size)
            # / self.stride, num_heads, dim
            # TODO: Consider mask here 
            x = layer(x, states[i], i, rpe)
            if self.right_context != 0:
                states[i]["encoder_states"] = x[0 : -self.right_context]
            else:
                states[i]["encoder_states"] = x[0 : ]

        if self.right_context != 0:
            lengths = (
                (
                    ~encoder_padding_mask[:, 0 : -self.right_context]
                )
                .sum(dim=1, keepdim=True)
                .long()
            )
        else:
            lengths = (
                (
                    ~encoder_padding_mask[:, 0 :]
                )
                .sum(dim=1, keepdim=True)
                .long()
            )

        return states[-1]["encoder_states"], lengths, states

# ------------------------------------------------------------------------------
#   AugmentedMemoryTransformerEncoderLayer
# ------------------------------------------------------------------------------
class ImplicitMemoryTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)
        self.share_mem_bank_layers = getattr(args, "share_mem_bank_layers", None)
        self.left_context_method = getattr(args, "left_context_method", "input")
        self.left_context=args.left_context // args.encoder_stride
        self.segment_size = args.segment_size // args.encoder_stride
        self.max_memory = ceil(self.left_context/self.segment_size) + 1
        self.enable_left_grad = getattr(args, "enable_left_grad", False)

    def update_memory(self, state, output, layer_num):
        if self.share_mem_bank_layers is not None:
          if not any(layer_num in layer for layer in self.share_mem_bank_layers):
            state["memory_banks"].append(output)
          else:
            for pairs in self.share_mem_bank_layers:
                if layer_num == pairs[0]:
                    state["memory_banks"].append(output)        
        else:
            state["memory_banks"].append(output) 
        
        if len(state["memory_banks"]) > self.max_memory:
            state["memory_banks"] = state["memory_banks"][-self.max_memory:]

        return state

    def forward(self, x, state, layer_num, rpe):
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

        if self.left_context_method == "output" and not self.enable_left_grad:
            self.update_memory(state, x.detach(), layer_num)
        elif self.left_context_method == "output" and self.enable_left_grad:
            self.update_memory(state, x, layer_num)

        return x

    def build_self_attention(self, embed_dim, args):
        return ImplicitMemoryMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=args.encoder_attention_heads,
            attention_dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            share_mem_bank_layers=getattr(args, "share_mem_bank_layers", None),
            max_relative_position=getattr(args, "max_relative_position", 0),
            left_context_method=getattr(args, "left_context_method", "input"),
            left_context=args.left_context // args.encoder_stride,
            segment_size = args.segment_size // args.encoder_stride,
            enable_left_grad = getattr(args, "enable_left_grad", False),
        )


# ------------------------------------------------------------------------------
#   AugmentedMemoryMultiheadAttention
# ------------------------------------------------------------------------------
class ImplicitMemoryMultiheadAttention(MultiheadAttention):
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
        attention_dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        memory_dim=None,
        std_scale=0.5,  # 0.5 based on https://arxiv.org/abs/2005.09137
        disable_mem_on_mem_attn=True,
        share_mem_bank_layers=None,
        max_relative_position=0,
        left_context_method="input",
        left_context=0,
        segment_size=0,
        enable_left_grad=False,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            kdim,
            vdim,
            attention_dropout,
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

        self.left_context_method = left_context_method
        self.left_context = left_context
        self.segment_size = segment_size
        self.max_memory = ceil(self.left_context/self.segment_size) + 1
        self.enable_left_grad = enable_left_grad

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

    def update_memory(self, state, output, layer_num):
        if self.share_mem_bank_layers is not None:
          if not any(layer_num in layer for layer in self.share_mem_bank_layers):
            state["memory_banks"].append(output)
          else:
            for pairs in self.share_mem_bank_layers:
                if layer_num == pairs[0]:
                    state["memory_banks"].append(output)        
        else:
            state["memory_banks"].append(output) 
        
        if self.max_memory > -1 and len(state["memory_banks"]) > self.max_memory:
            state["memory_banks"] = state["memory_banks"][-self.max_memory :]

        return state

    def add_memory(self, input, memory):
        memory_and_input = input
        if memory != []:
            memory = torch.cat(memory, dim=0)
            if memory.size(0) > self.left_context:
                memory = memory[memory.size(0)-self.left_context:]
                
            memory_and_input = torch.cat([memory] + [input], dim=0)
        return memory_and_input

    def forward(self, input, state, layer_num, rpe):
        """
        input: Encoder states of current segment with left or right context,
            plus one summarization query

        """
        length, batch_size, _ = input.shape

        memory = state["memory_banks"]

        memory_and_input = self.add_memory(input, memory)

        q = self.q_proj(self.v2e(input))
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
            length + (0 if len(memory) == 0 else self.left_context if len(memory)*self.segment_size > self.left_context else len(memory)*self.segment_size),
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
        
        if self.left_context_method == "input" and not self.enable_left_grad:
            self.update_memory(state, input.detach(), layer_num)
        elif self.left_context_method == "input" and self.enable_left_grad:
            self.update_memory(state, input, layer_num)

        if self.left_context_method == "pre_output" and not self.enable_left_grad:
            self.update_memory(state, output.detach(), layer_num)
        elif self.left_context_method == "pre_output" and self.enable_left_grad:
            self.update_memory(state, output, layer_num)

        return output

# ------------------------------------------------------------------------------
#   SequenceEncoder
# ------------------------------------------------------------------------------
class ImplicitSequenceEncoder(FairseqEncoder):
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

        seg_encoder_states_lengths: List[Tuple[Tensor, Tensor]] = []

        for seg_src_tokens, seg_src_lengths in seg_src_tokens_lengths:
            src_tokens = seg_src_tokens

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
def implicit_memory(klass):
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
                "--share-mem-bank-layers",
                type=json.loads,
                default=None,
                help="The list of memory bank sharing layers",
            )
            parser.add_argument(
                "--left-context-method",
                default="input",
                choices=["input", "pre_output", "output"],
                help="Left context strategy for normal left context"
            )
            parser.add_argument(
                "--max-relative-position",
                type=int,
                default=0,
                help="Relative Position",
            )
            parser.add_argument(
                "--enable-left-grad",
                action="store_true",
                default=False,
                help="if True, squash memory banks",
            )

    StreamSeq2SeqModel.__name__ = klass.__name__
    return StreamSeq2SeqModel