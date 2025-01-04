import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import math
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig, GenerationConfig
from threading import Thread
import argparse
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer


@dataclass
class Config:
    hidden_size: int = 4096
    ffn_hidden_size: int = 13696
    kv_channels: int = 128
    num_layers: int = 40
    num_attention_heads: int = 32
    multi_query_attention: bool = True
    multi_query_group_num: int = 2
    padded_vocab_size: int = 151552
    seq_length: int = 8192
    layernorm_epsilon: float = 0.00000015625
    torch_dtype: float = "bfloat16"

    add_qkv_bias: bool = True # Means that the qkv linear layers have bias terms.
    post_layer_norm: bool = True # At the end of all layers, there is an additional RMSNorm.
    add_bias_linear: bool = False  # The linear layers in the FFN do not have bias terms.

    is_encoder_decoder: bool = False

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, rope_ratio=1, original_impl=False, device=None, dtype=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl
        self.rope_ratio = rope_ratio

    def forward_impl(
            self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
    ):
        """Enhanced Transformer with Rotary Position Embedding.
        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        base = base * self.rope_ratio
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(
            max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )


@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [b, np, sq, hn]
    b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:, :sq]
    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)



# TODO: Implement the RMSNorm class.
# Done
class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
    def forward(self, hidden_states: torch.Tensor):
        # Calculate the mean square of the hidden states across the last dimension
        in_type = hidden_states.dtype
        rms = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        
        hidden_states = hidden_states * torch.rsqrt(rms + self.eps)

        return (self.weight * hidden_states).to(in_type)

# TODO: Implement the Attention class.
# Done
class Attention(torch.nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config
        self.is_causal = True

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # query, key, value layer [batch, number_heads, sequence_length, hidden_size_per_head]
        output_size = (query_layer.size(0), query_layer.size(1), query_layer.size(2), key_layer.size(2))
        query_layer = query_layer.view(output_size[0] * output_size[1], output_size[2], -1)
        key_layer = key_layer.view(output_size[0] * output_size[1], output_size[3], -1)

        matmul_input_buffer = torch.empty(
            output_size[0] * output_size[1], output_size[2], output_size[3], dtype=query_layer.dtype,
            device=query_layer.device
        )
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer,  # [b * np, sq, hn]
            key_layer.transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )
        attention_scores = matmul_result.view(*output_size).float()

        # attention scores and attention mask [batch, number_heads, sequence_length, sequence_length]
        if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
            attention_mask = torch.ones(output_size[0], 1, output_size[2], output_size[3],
                                        device=attention_scores.device, dtype=torch.bool)
            attention_mask.tril_()
            attention_mask = ~attention_mask
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.type_as(value_layer)

        output_size = (value_layer.size(0), value_layer.size(1), query_layer.size(1), value_layer.size(3))
        value_layer = value_layer.view(output_size[0] * output_size[1], value_layer.size(2), -1)
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        context_layer = torch.bmm(attention_probs, value_layer)
        context_layer = context_layer.view(*output_size)
        context_layer = context_layer.transpose(1, 2).contiguous()
        new_size = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(*new_size)

        # context_layer [batch, sequence_length, hidden_size]
        return context_layer

# TODO: Implement the AttentionBlock class.
# Done
class AttentionBlock(torch.nn.Module):
    """Parallel self-attention layer abstract class.
    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config, device=None, dtype=torch.bfloat16):
        super(AttentionBlock, self).__init__()

        self.dtype = dtype
        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                    self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        self.query_key_value = nn.Linear(config.hidden_size, self.qkv_hidden_size,
                                          bias=config.add_bias_linear or config.add_qkv_bias,
                                          device=device, dtype=dtype)

        self.core_attention = Attention(config)
        self.dense = nn.Linear(self.projection_size, config.hidden_size, bias=config.add_bias_linear,
                                 device=device, dtype=dtype)

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
    ):

        mixed_x_layer = self.query_key_value(hidden_states)
        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:
            new_shape = mixed_x_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head * 3)

            query_layer, key_layer, value_layer = mixed_x_layer.view(*new_shape).chunk(3, dim=-1)

        query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)


        if kv_cache is not None:
            key_layer = torch.cat([kv_cache[0], key_layer], dim=2)
            value_layer = torch.cat([kv_cache[1], value_layer], dim=2)
        if use_cache:
            if kv_cache is None:
                new_kv_cache = torch.cat((key_layer.unsqueeze(0).unsqueeze(0), value_layer.unsqueeze(0).unsqueeze(0)), dim=1)
            else:
                new_kv_cache = (key_layer, value_layer)
        else:
            new_kv_cache = None

        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(2).expand(-1, -1,
                    self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1)
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
            )
            value_layer = value_layer.unsqueeze(2).expand(-1, -1,
                    self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1)
            
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
            )


        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)


        # =================
        # Output. [sequence_length, batch, hidden size]
        # =================

        output = self.dense(context_layer)

        return output, new_kv_cache


# TODO: Implement the MLP class.
# Done
class MLP(torch.nn.Module):
    """MLP.
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config, device=None, dtype=torch.bfloat16):
        super(MLP, self).__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, 
                                        config.ffn_hidden_size * 2,
                                        bias=config.add_bias_linear, 
                                        device=device, 
                                        dtype=dtype)
        
        self.dense_4h_to_h = nn.Linear(config.ffn_hidden_size, 
                                       config.hidden_size, 
                                       bias=config.add_bias_linear, 
                                       device=device, 
                                       dtype=dtype)

    def _swiglu(self, x):
        x = torch.chunk(x, 2, dim=-1)
        return torch.nn.functional.silu(x[0]) * x[1]

    def forward(self, hidden_states):
        intermediate_output = self.dense_h_to_4h(hidden_states)
        intermediate_output = self._swiglu(intermediate_output)
        output = self.dense_4h_to_h(intermediate_output)
        return output

# TODO: Implement the Layer class.
# Done
class Layer(torch.nn.Module):
    def __init__(self, config, device):
        super(Layer, self).__init__()
        self.dtype = torch.bfloat16 if config.torch_dtype == "bfloat16" else torch.float32
        self.self_attention = AttentionBlock(config, device=device, dtype=self.dtype)
        self.input_layernorm = RMSNorm(config.hidden_size, 
                                       eps=config.layernorm_epsilon, 
                                       device=device, 
                                       dtype=self.dtype)
        self.output_layernorm = RMSNorm(config.hidden_size, 
                                        eps=config.layernorm_epsilon, 
                                        device=device, 
                                        dtype=self.dtype)
        self.mlp = MLP(config, device=device, dtype=self.dtype)

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True,
    ):
        # hidden_states: [sequence_length, batch, hidden size]

        layernorm_output = self.input_layernorm(hidden_states)
        self_attn_output, new_kv_cache = self.self_attention(
            layernorm_output, attention_mask, rotary_pos_emb, kv_cache=kv_cache, use_cache=use_cache
        )
        layernorm_input = hidden_states + self_attn_output
        layernorm_output = self.output_layernorm(layernorm_input)

        mlp_output = self.mlp(layernorm_output)
        output = layernorm_input + mlp_output

        # output: [sequence_length, batch, hidden size]
        return output, new_kv_cache

# TODO: Implement the Transformer class.
# Done
class Transformer(torch.nn.Module):
    def __init__(self, config, device):
        super(Transformer, self).__init__()
        self.num_layers = config.num_layers
        self.post_layer_norm = config.post_layer_norm
        self.dtype = torch.bfloat16 if config.torch_dtype == "bfloat16" else torch.float32
        self.layers = nn.ModuleList([Layer(config, device) for _ in range(self.num_layers)])
        if self.post_layer_norm:
            self.output_layernorm = RMSNorm(config.hidden_size, 
                                            eps=config.layernorm_epsilon, 
                                            device=device, 
                                            dtype=self.dtype)

    def _get_layer(self, layer_id):
        return self.layers[layer_id]

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
    ):
        if not kv_caches:
            kv_caches = [None] * self.num_layers
        # Mark diff
        present_key_values = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        for i in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer = self._get_layer(i)
            layer_ret = layer(
                hidden_states, attention_mask, rotary_pos_emb, kv_cache=kv_caches[i], use_cache=use_cache
            )
            hidden_states, present_kv_cache = layer_ret
            if use_cache:
                if kv_caches[0] is not None:
                    present_key_values += (present_kv_cache,)
                else:
                    if len(present_key_values) == 0:
                        present_key_values = present_kv_cache
                    else:
                        present_key_values = torch.cat((present_key_values, present_kv_cache.to(present_key_values.device)), dim=0)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.post_layer_norm:
            hidden_states = self.output_layernorm(hidden_states)
        new_kv_caches = present_key_values
        # new_kv_caches is a tuple
        # length: num_layers, each element is a tuple of length 2 (key, value cache)
        # key shape: [batch, multi_query_group_num, seq_len, kv_channels]
        return hidden_states, new_kv_caches


class GLM4(torch.nn.Module):
    def __init__(self, config, device):
        super(GLM4, self).__init__()
        self.dtype = torch.bfloat16 if config.torch_dtype == "bfloat16" else torch.float32
        self.word_embedding = torch.nn.Embedding(config.padded_vocab_size, config.hidden_size, dtype=self.dtype, device=device)
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels
        self.seq_length = config.seq_length

        self.model = Transformer(config, device=device)
        self.output_layer = nn.Linear(config.hidden_size, config.padded_vocab_size, bias=False, dtype=self.dtype, device=device)
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, rope_ratio=1,
                                              original_impl=True,
                                              device=device, dtype=self.dtype)

    def word_embedding_forward(self, input_ids):
        return self.word_embedding(input_ids)

    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        if self.config._attn_implementation == "flash_attention_2":
            if padding_mask is not None and not padding_mask.all():
                return padding_mask
            return None
        batch_size, seq_length = input_ids.shape
        full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
        full_attention_mask.tril_()
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
        if past_length:
            full_attention_mask = torch.cat((torch.ones(batch_size, seq_length, past_length,
                                                        device=input_ids.device), full_attention_mask), dim=-1)
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask.unsqueeze_(1)
        return full_attention_mask

    def forward(self, input_ids, position_ids = None,
                past_key_values=None, full_attention_mask=None, attention_mask=None,
                use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        inputs_embeds = self.word_embedding_forward(input_ids)
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
        hidden_states, presents = self.model(
            inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, use_cache=use_cache
        )
        if presents is not None and type(presents) is torch.Tensor:
            presents = presents.split(1, dim=0)
            presents = list(presents)
            presents = [list(x.squeeze(0).split(1, dim=0)) for x in presents]
            presents = [tuple([x.squeeze(0) for x in y]) for y in presents]
            presents = tuple(presents)

        return hidden_states, presents


class ChatGLMForConditionalGeneration(PreTrainedModel):
    def __init__(self, config, device=None):
        pretrain_config = PretrainedConfig(is_decoder=True, is_encoder_decoder=False)
        super().__init__(pretrain_config)

        self.max_sequence_length = 2500
        self.transformer = GLM4(config, device=device)
        self.config = config

    def get_position_ids(self, input_ids, device):
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        return position_ids

    def _update_model_kwargs_for_generation(
            self,
            outputs,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        cache_name, cache = self._extract_past_from_model_output(outputs)
        model_kwargs[cache_name] = cache

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_position_id], dim=-1
            )

        model_kwargs["is_first_forward"] = False
        return model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            is_first_forward: bool = True,
            **kwargs
    ) -> dict:
        # only last token for input_ids if past is not None
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids, device=input_ids.device)
        if not is_first_forward:
            if past_key_values is not None:
                position_ids = position_ids[..., -1:]
                input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "return_last_logit": True,
            "use_cache": use_cache
        }

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_last_logit: Optional[bool] = False,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        if return_last_logit:
            hidden_states = hidden_states[:, -1:]
        lm_logits = self.transformer.output_layer(hidden_states)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return output

        return CausalLMOutputWithPast(
            logits=lm_logits,
            past_key_values=transformer_outputs[1],
        )

    @staticmethod
    def _reorder_cache(
            past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        Output shares the same memory storage as `past`.
        """
        return tuple(
            (
                layer_past[0].index_select(0, beam_idx.to(layer_past[0].device)),
                layer_past[1].index_select(0, beam_idx.to(layer_past[1].device)),
            )
            for layer_past in past
        )


def convert_ckpt():
    huggingface_model = AutoModelForCausalLM.from_pretrained(
        "../models--THUDM--glm-4-9b-chat/snapshots/1ff770585cbf3c1ece419f34f8161c88c7e9a224",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    model_dict = huggingface_model.state_dict()
    new_model_dict = {}
    for k, v in model_dict.items():
        new_model_dict[k.replace("encoder", "model")
                       .replace("final_layernorm", "output_layernorm")
                       .replace("post_attention_layernorm", "output_layernorm")
                       .replace("embedding.word_embeddings", "word_embedding")] = v
    torch.save(new_model_dict, "glm4.pt")


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = model.generation_config.eos_token_id
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # 设置 GPU 编号，如果单机单卡指定一个，单机多卡指定多个 GPU 编号
    MODEL_PATH = "../models--THUDM--glm-4-9b-chat/snapshots/1ff770585cbf3c1ece419f34f8161c88c7e9a224"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = ChatGLMForConditionalGeneration(config=Config(), device=device).eval()
    model.load_state_dict(torch.load("glm4.pt"))
    generation_config = GenerationConfig(
        eos_token_id=[151329,151336,151338],
        pad_token_id= 151329,
        do_sample= True,
        temperature= 0.8,
        max_new_tokens= 8192,
        top_p= 0.8,
        top_k= 1,
        transformers_version= "4.44.0")
    model.generation_config = generation_config
    history = []
    stop = StopOnTokens()

    parser = argparse.ArgumentParser(description="GLM-4")
    parser.add_argument("-d", "--detailed", action="store_true")
    args = parser.parse_args()

    print("Welcome to the GLM-4-9B CLI chat. Type your messages below.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        history.append([user_input, ""])

        messages = []
        for idx, (user_msg, model_msg) in enumerate(history):
            if idx == len(history) - 1 and not model_msg:
                messages.append({"role": "user", "content": user_msg})
                break
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if model_msg:
                messages.append({"role": "assistant", "content": model_msg})
        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        if not args.detailed:
            streamer = TextIteratorStreamer(
                tokenizer=tokenizer,
                timeout=60,
                skip_prompt=True,
                skip_special_tokens=True
            )
            generate_kwargs = {
                "input_ids": model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"],
                "input_ids": model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"],
                "streamer": streamer,
                "stopping_criteria": StoppingCriteriaList([stop]),
                "repetition_penalty": 1.2,
            }
            t = Thread(target=model.generate, kwargs=generate_kwargs)
            t.start()
            print("GLM-4:", end="", flush=True)
            delete_first_newline = False
            for new_token in streamer:
                if not delete_first_newline:
                    delete_first_newline = True
                    continue
                if new_token:
                    print(new_token, end="", flush=True)
                    history[-1][1] += new_token
            print()
        else:
            generation_config = GenerationConfig(
            eos_token_id=[151329,151336,151338],
            pad_token_id= 151329,
            do_sample= True,
            temperature= 0.8,
            max_new_tokens= 8192,
            top_p= 0.8,
            top_k= 1,
            transformers_version= "4.44.0")

            with torch.no_grad():
                start_time = time.time()
                outputs = model.generate(**model_inputs, generation_config=generation_config)
                outputs = outputs[:, model_inputs['input_ids'].shape[1]:]
                print("GLM-4:", tokenizer.decode(outputs[0], skip_special_tokens=True)[1:])
                print("Answer vector length: %d" % outputs.shape[1])
                print("Answer generated in %f s" % (time.time() - start_time))

        history[-1][1] = history[-1][1].strip()