# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from math import ceil, sqrt
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
from fairseq.modules import TransformerEncoderLayer
from fairseq.modules.quant_noise import quant_noise
from torch import nn, Tensor
from fairseq.modules.positional_encoding import RelPositionalEncoding
from fairseq.modules.espnet_multihead_attention import RelPositionMultiHeadedAttention
import json

# ------------------------------------------------------------------------------
#   AugmentedMemoryConvTransformerEncoder
# ------------------------------------------------------------------------------

class AugmentedMemoryConvTransformerEncoder_test(ConvTransformerEncoder):
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
            shared_weights_layer = AugmentedMemoryTransformerEncoderLayer(args)
        print(f"Encoder: Sharing layers: {encoder_weight_share_list}")
        for layer_idx in range(args.encoder_layers):
            if layer_idx+1 in encoder_weight_share_list:
                self.transformer_layers.append(shared_weights_layer)
            else:
                self.transformer_layers.append(AugmentedMemoryTransformerEncoderLayer(args))

        self.share_mem_bank_layers = getattr(args, "share_mem_bank_layers", None)
        self.variable_left_context_method = getattr(args, "variable_left_context_method", None)
        self.get_positional_emb = RelPositionalEncoding(self.left_context+self.segment_size+self.right_context, args.encoder_embed_dim)

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

    def forward(self, src_tokens, src_lengths, left_context_size, right_context_size, states=None, prev_output=None):
        """Encode input sequence.
        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        left_context_size = left_context_size // self.encoder_stride
        right_context_size = ceil(right_context_size / self.encoder_stride)
        
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

        if self.variable_left_context_method == "output":
            x_dim = x.size(0)
            if x_dim < self.segment_size and prev_output is not None:
                left_context_size = self.segment_size-x_dim
                prev_output = prev_output[self.segment_size-left_context_size:]
                x = torch.cat([prev_output] + [x], dim=0)
                input_lengths = input_lengths + left_context_size

        encoder_padding_mask, _ = lengths_to_encoder_padding_mask(
            input_lengths, batch_first=True
        )

        x = F.dropout(x, p=self.dropout, training=self.training)

        # State to store memory banks etc.
        states = self.initialize_states(states)

        pos_emb = self.get_positional_emb(torch.tensor(0.0, dtype=x.dtype, device=x.device).expand(1, self.left_context + left_context_size + x.size(0) + right_context_size))

        for i, layer in enumerate(self.transformer_layers):
            # x size:
            # (self.left_size + self.segment_size + self.right_size)
            # / self.stride, num_heads, dim
            # TODO: Consider mask here 
            x = layer(x, states[i], i, pos_emb, left_context_size, right_context_size)
            if right_context_size != 0:
                states[i]["encoder_states"] = x[left_context_size : -right_context_size]
            else:
                states[i]["encoder_states"] = x[left_context_size : ]

        if right_context_size != 0:
            lengths = (
                (
                    ~encoder_padding_mask[:, left_context_size : -right_context_size]
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

        return states[-1]["encoder_states"], lengths, states, states[-1]["encoder_states"]

# ------------------------------------------------------------------------------
#   AugmentedMemoryTransformerEncoderLayer
# ------------------------------------------------------------------------------
class AugmentedMemoryTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)
        self.share_mem_bank_layers = getattr(args, "share_mem_bank_layers", None)
        self.left_context_method = getattr(args, "left_context_method", "input")
        self.left_context=args.left_context // args.encoder_stride
        self.segment_size = args.segment_size // args.encoder_stride
        self.max_memory = ceil(self.left_context/self.segment_size) + 1
        self.enable_left_grad = getattr(args, "enable_left_grad", False)

        self.normalize_left = getattr(args, "normalize_left", False)
        if self.normalize_left:
            self.mem_layer_norm = torch.nn.LayerNorm(args.encoder_embed_dim)

        self.feed_forward_mem = getattr(args, "feed_forward_mem", False)
        if self.feed_forward_mem:
            self.mem_fc1 = quant_noise(torch.nn.Linear(self.embed_dim, self.encoder_ffn_embed_dim), self.quant_noise, self.quant_noise_block_size)
            self.mem_activation = torch.nn.functional.relu
            self.mem_fc2 = quant_noise(torch.nn.Linear(self.encoder_ffn_embed_dim, self.embed_dim), self.quant_noise, self.quant_noise_block_size)
    
    def process_output(self, output):
        if self.normalize_left:
            if self.feed_forward_mem:
                residual = output
                output = self.mem_activation(self.mem_fc1(output))
                output = self.activation_dropout_module(output)
                output = self.mem_fc2(output)
                output = self.dropout_module(output)
                output = self.mem_layer_norm(output+residual)
            else:
                output = self.mem_layer_norm(output)
        return output

    def update_memory(self, state, output, layer_num, new_left_context, new_right_context):
        if new_right_context != 0:
            output = output[new_left_context:-new_right_context]
        else:
            output = output[new_left_context:]

        if self.share_mem_bank_layers is not None:
          if not any(layer_num in layer for layer in self.share_mem_bank_layers):
            output = self.process_output(output)
            state["memory_banks"].append(output)
          else:
            for pairs in self.share_mem_bank_layers:
                if layer_num == pairs[0]:
                    output = self.process_output(output)
                    state["memory_banks"].append(output)        
        else:
            output = self.process_output(output)
            state["memory_banks"].append(output) 
        
        if len(state["memory_banks"]) > self.max_memory:
            state["memory_banks"] = state["memory_banks"][-self.max_memory:]

        return state

    def forward(self, x, state, layer_num, pos_emb, new_left_context, new_right_context):
        residual = x

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x = self.self_attn(input=x, state=state, layer_num=layer_num, pos_emb=pos_emb, new_left_context=new_left_context, new_right_context=new_right_context)

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
            self.update_memory(state, x.detach(), layer_num, new_left_context, new_right_context)
        elif self.left_context_method == "output" and self.enable_left_grad:
            self.update_memory(state, x, layer_num, new_left_context, new_right_context)

        return x

    def build_self_attention(self, embed_dim, args):
        return AugmentedMemoryMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=args.encoder_attention_heads,
            attention_dropout=args.attention_dropout,
            share_mem_bank_layers=getattr(args, "share_mem_bank_layers", None),
            left_context_method=getattr(args, "left_context_method", "input"),
            left_context=args.left_context // args.encoder_stride,
            segment_size = args.segment_size // args.encoder_stride,
            enable_left_grad = getattr(args, "enable_left_grad", False),
        )


# ------------------------------------------------------------------------------
#   AugmentedMemoryMultiheadAttention
# ------------------------------------------------------------------------------
class AugmentedMemoryMultiheadAttention(RelPositionMultiHeadedAttention):
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
        attention_dropout=0.0,
        share_mem_bank_layers=None,
        left_context_method="input",
        left_context=0,
        segment_size=0,
        enable_left_grad=False,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            attention_dropout,
        )

        self.left_context_method = left_context_method
        self.left_context = left_context
        self.segment_size = segment_size
        self.max_memory = ceil(self.left_context/self.segment_size) + 1
        self.enable_left_grad = enable_left_grad

        self.share_mem_bank_layers = share_mem_bank_layers

        #Relative Positional Encoding
        self.zero_triu = False
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(embed_dim, embed_dim, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x: Input tensor B X n_head X T X 2T-1
        Returns:
            torch.Tensor: Output tensor.
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def update_memory(self, state, output, layer_num, new_left_context, new_right_context):
        if new_right_context != 0:
            output = output[new_left_context:-new_right_context]
        else:
            output = output[new_left_context:]

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

    def add_memory(self, input, memory, new_left_context):
        memory_and_input = input
        if memory != []:
            memory = torch.cat(memory, dim=0)
            if new_left_context != 0:
                memory = memory[:-new_left_context]
            if memory.size(0) > self.left_context:
                memory = memory[memory.size(0)-self.left_context:]
                
            memory_and_input = torch.cat([memory] + [input], dim=0)
        return memory_and_input

    def forward(self, input, state, layer_num, pos_emb, new_left_context, new_right_context):
        """
        input: Encoder states of current segment with left or right context,
            plus one summarization query

        """

        memory = state["memory_banks"]

        memory_and_input = self.add_memory(input, memory, new_left_context)

        query = input.transpose(0, 1)
        key = memory_and_input.transpose(0, 1)
        value = memory_and_input.transpose(0, 1)
        pos_emb = pos_emb.transpose(0, 1)
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        scores = self.forward_attention(v, scores, None)
        output = scores.transpose(0, 1)
        
        if self.left_context_method == "input" and not self.enable_left_grad:
            self.update_memory(state, input.detach(), layer_num, new_left_context, new_right_context)
        elif self.left_context_method == "input" and self.enable_left_grad:
            self.update_memory(state, input, layer_num, new_left_context, new_right_context)

        if self.left_context_method == "pre_output" and not self.enable_left_grad:
            self.update_memory(state, output.detach(), layer_num, new_left_context, new_right_context)
        elif self.left_context_method == "pre_output" and self.enable_left_grad:
            self.update_memory(state, output, layer_num, new_left_context, new_right_context)

        return output

# ------------------------------------------------------------------------------
#   SequenceEncoder
# ------------------------------------------------------------------------------
class SequenceEncoder_test(FairseqEncoder):
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
        self.shift_right_context = getattr(args, "shift_right_context", False)
        
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
            extra_right_context=0,
        )

        seg_encoder_states_lengths: List[Tuple[Tensor, Tensor]] = []

        prev_input = None
        prev_output = None
        cur_seg_count = 0
        total_seg_count = len(seg_src_tokens_lengths)
        for seg_src_tokens, seg_src_lengths in seg_src_tokens_lengths:
            prev_input_tmp = prev_input
            src_tokens = seg_src_tokens
            left_context_size = 0
            if self.variable_left_context_method == "input":
                seg_src_tokens_dim = seg_src_tokens.size(self.input_time_axis)
                if seg_src_tokens_dim < self.segment_size and prev_input is not None:
                    left_context_size = self.segment_size - seg_src_tokens_dim
                    prev_input_tmp = prev_input[:,prev_input.size(self.input_time_axis)-left_context_size:]
                    seg_src_tokens = torch.cat([prev_input_tmp] + [seg_src_tokens], dim=self.input_time_axis)
                    prev_input_tmp = prev_input[:, :-prev_input_tmp.size(self.input_time_axis)]
                    seg_src_lengths = seg_src_lengths + left_context_size

            right_context_size = 0
            #Checks if right context available
            if self.right_context != 0:
                #Determines if current segment is not final segment
                if total_seg_count > cur_seg_count+1:
                    future_seg_src_tokens, _ = seg_src_tokens_lengths[cur_seg_count+1]
                    right_context_size = future_seg_src_tokens.size(self.input_time_axis)
                    #Determines method to apply right context to segment
                    if right_context_size > self.right_context:
                        future_seg_src_tokens = future_seg_src_tokens[:,:-(right_context_size-self.right_context)]
                        right_context_size = future_seg_src_tokens.size(self.input_time_axis)
                    seg_src_tokens = torch.cat([seg_src_tokens] + [future_seg_src_tokens], dim=self.input_time_axis)
                    seg_src_lengths = seg_src_lengths + right_context_size

                #Fill in additional space for right context with left context 
                if self.shift_right_context and right_context_size < self.right_context and prev_input_tmp is not None:                   
                    prev_input_tmp_size = prev_input_tmp.size(self.input_time_axis)
                    if prev_input_tmp_size-(self.right_context - right_context_size) > 0:
                        prev_input_tmp = prev_input_tmp[:,prev_input_tmp_size-(self.right_context - right_context_size):]  
                        prev_input_tmp_size = prev_input_tmp.size(self.input_time_axis)
                    seg_src_lengths = seg_src_lengths + prev_input_tmp_size
                    left_context_size = left_context_size + prev_input_tmp_size
                    seg_src_tokens = torch.cat([prev_input_tmp]+[seg_src_tokens], dim=self.input_time_axis)

                cur_seg_count += 1
            
            (seg_encoder_states, seg_enc_lengths, states, prev_output) = self.module(
            seg_src_tokens,
            seg_src_lengths,
            left_context_size,
            right_context_size,
            states=states,
            prev_output=prev_output,
            )

            seg_encoder_states_lengths.append((seg_encoder_states, seg_enc_lengths))
            
            if prev_input is None:
                prev_input = src_tokens
            elif prev_input.size(self.input_time_axis) < 2*self.segment_size:
                prev_input = torch.cat([prev_input] + [src_tokens], dim=1)
            else:
                prev_input = torch.cat([prev_input[:, self.segment_size:]] + [src_tokens], dim=1)
            
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
def augmented_memory_test(klass):
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
                "--variable-left-context-method",
                default=None,
                choices=["input", "output"],
                help="Left context strategy for variable length left context"
            )
            parser.add_argument(
                "--left-context-method",
                default="input",
                choices=["input", "pre_output", "output"],
                help="Left context strategy for normal left context"
            )
            parser.add_argument(
                "--shift-right-context",
                action="store_true",
                default=False,
                help="if True, squash memory banks",
            )
            parser.add_argument(
                "--enable-left-grad",
                action="store_true",
                default=False,
                help="if True, squash memory banks",
            )
            parser.add_argument(
                "--normalize-left",
                action="store_true",
                default=False,
                help="if True, squash memory banks",
            )
            parser.add_argument(
                "--feed-forward-mem",
                action="store_true",
                default=False,
                help="if True, squash memory banks",
            )

    StreamSeq2SeqModel.__name__ = klass.__name__
    return StreamSeq2SeqModel