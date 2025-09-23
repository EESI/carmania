import inspect
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.utils.import_utils import is_torch_fx_available

from torch.utils.checkpoint import checkpoint
from functools import partial



# Try to import flash-attn; if unavailable or fails to initialize on this device
# we will set a flag and provide a fallback implementation below.
try:
    from flash_attn import flash_attn_func as _flash_attn_func
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa: F401

    HAVE_FLASH_ATTN = True
except Exception:
    _flash_attn_func = None
    _flash_attn_varlen_func = None
    index_first_axis = None
    pad_input = None
    unpad_input = None
    HAVE_FLASH_ATTN = False



def _repeat_kv_for_gqa(x: torch.Tensor, repeat: int) -> torch.Tensor:
    # x: [B, S, Hk, D] -> [B, S, Hq, D], where Hq = Hk * repeat
    if repeat == 1:
        return x
    B, S, Hk, D = x.shape
    x = x.unsqueeze(2).expand(B, S, repeat, Hk, D)   # [B,S,repeat,Hk,D]
    return x.reshape(B, S, repeat * Hk, D)

@torch.no_grad()
def _build_window_mask(
    Sq: int, Sk: int, left: int, right: int, causal: bool, device: torch.device
) -> torch.Tensor:
    """
    FA2 window semantics:
      valid j for query i: j ∈ [ i + Sk - Sq - left, i + Sk - Sq + right ]
    FA2.1 causal alignment (bottom-right): additionally disallow j > i + Sk - Sq
    Return: float mask [1,1,Sq,Sk] with 0 for keep, -inf for mask.
    """
    i = torch.arange(Sq, device=device).view(-1, 1)  # [Sq,1]
    j = torch.arange(Sk, device=device).view(1, -1)  # [1,Sk]
    shift = Sk - Sq
    j_min = i + shift - left
    j_max = i + shift + right
    allowed = (j >= j_min) & (j <= j_max)
    if causal:
        # forbid looking ahead relative to FA2.1 alignment
        allowed &= (j <= (i + shift))
    masked = ~allowed
    m = torch.full((Sq, Sk), 0.0, device=device)
    m[masked] = -torch.finfo(m.dtype).max  # -inf
    return m.view(1, 1, Sq, Sk).contiguous()

@torch.no_grad()
def _build_causal_mask_fa21(
    Sq: int, Sk: int, device: torch.device
) -> torch.Tensor:
    """
    FA2.1 causal only (no window): mask positions with j > i + (Sk - Sq).
    Returns float mask [1,1,Sq,Sk] with 0 keep, -inf mask.
    """
    i = torch.arange(Sq, device=device).view(-1, 1)
    j = torch.arange(Sk, device=device).view(1, -1)
    shift = Sk - Sq
    allowed = (j <= (i + shift))
    masked = ~allowed
    m = torch.full((Sq, Sk), 0.0, device=device)
    m[masked] = -torch.finfo(m.dtype).max
    return m.view(1, 1, Sq, Sk).contiguous()

def _sdpa_flash_attn_compat(
    q: torch.Tensor,  # [B,Sq,Hq,D]
    k: torch.Tensor,  # [B,Sk,Hk,D]
    v: torch.Tensor,  # [B,Sk,Hk,D]
    *,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,    # default 1/sqrt(D) if None
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),  # (-1,-1) == no window
    alibi_slopes: Optional[torch.Tensor] = None,  # (Hq,) or (B,Hq)
    training: Optional[bool] = None,
) -> torch.Tensor:
    """
    SDPA path emulating flash_attn_func semantics (v2):
      - supports GQA (Hq divisible by Hk)
      - FA2.1 causal alignment when Sq != Sk
      - sliding window: j in [i + Sk - Sq - left, i + Sk - Sq + right]
      - ALiBi additive bias
    Returns: [B,Sq,Hq,D] with original dtype.
    """
    assert q.dim() == k.dim() == v.dim() == 4, "Expect [B,S,H,D] tensors"
    B, Sq, Hq, D = q.shape
    Bk, Sk, Hk, Dk = k.shape
    assert (Bk, Sk, Dk) == (B, k.shape[1], D), "Batch/Dim mismatch"
    assert v.shape[:3] == k.shape[:3] and v.shape[3] == D, "K/V mismatch"
    assert Hq % Hk == 0, "Hq must be divisible by Hk for GQA/MQA"
    repeat = Hq // Hk

    # GQA: expand K,V heads to match Q heads so SDPA sees [B,Hq,*,D]
    k_exp = _repeat_kv_for_gqa(k, repeat)  # [B,Sk,Hq,D]
    v_exp = _repeat_kv_for_gqa(v, repeat)  # [B,Sk,Hq,D]

    # layout for SDPA: [B,H,S,D]
    qh = q.permute(0, 2, 1, 3).to(torch.float32)     # [B,Hq,Sq,D]
    kh = k_exp.permute(0, 2, 1, 3).to(torch.float32) # [B,Hq,Sk,D]
    vh = v_exp.permute(0, 2, 1, 3).to(torch.float32) # [B,Hq,Sk,D]
    in_dtype = q.dtype
    device = q.device

    # softmax scale: default 1/sqrt(D); emulate custom s by scaling Q by s*sqrt(D)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)
    qh = qh * (softmax_scale * math.sqrt(D))

    # Build float mask (+ALiBi) as additive bias; pass is_causal=False to SDPA.
    left, right = window_size
    use_window = (left, right) != (-1, -1)
    attn_bias = None  # [B,Hq,Sq,Sk] float, 0 for keep, -inf for mask, +ALiBi

    if use_window:
        # Per FA2 semantics; also clamp look-ahead under causal
        if causal and right > 0:
            right = 0
        base = _build_window_mask(Sq, Sk, left, right, causal, device)  # [1,1,Sq,Sk]
        attn_bias = base.expand(B, Hq, Sq, Sk)
        is_causal = False
    elif causal:
        base = _build_causal_mask_fa21(Sq, Sk, device)  # [1,1,Sq,Sk]
        attn_bias = base.expand(B, Hq, Sq, Sk)
        is_causal = False
    else:
        is_causal = False
        attn_bias = None  # fastest path

    # ALiBi: add -(slope * |(i + Sk - Sq) - j|) to logits (i=0..Sq-1, j=0..Sk-1)
    if alibi_slopes is not None:
        # make slopes shape [B,Hq,1,1]
        if alibi_slopes.dim() == 1:
            # [Hq] -> [1,Hq,1,1]
            alibi = alibi_slopes.view(1, Hq, 1, 1).to(dtype=torch.float32, device=device)
            alibi = alibi.expand(B, Hq, 1, 1)
        elif alibi_slopes.dim() == 2:
            # [B,Hq] -> [B,Hq,1,1]
            alibi = alibi_slopes.view(B, Hq, 1, 1).to(dtype=torch.float32, device=device)
        else:
            raise ValueError("alibi_slopes must be (Hq,) or (B,Hq)")
        i = torch.arange(Sq, device=device).view(1, 1, -1, 1)
        j = torch.arange(Sk, device=device).view(1, 1, 1, -1)
        shift = Sk - Sq
        dist = (i + shift - j).abs().to(torch.float32)  # [1,1,Sq,Sk]
        alibi_term = -(alibi * dist)                     # [B,Hq,Sq,Sk]
        if attn_bias is None:
            attn_bias = alibi_term
        else:
            attn_bias = attn_bias + alibi_term

    # Dropout (train) vs eval
    if training is None:
        training = (dropout_p > 0.0) and any(t.requires_grad for t in (q, k, v))
    dp = dropout_p if training else 0.0

    out = F.scaled_dot_product_attention(
        qh, kh, vh,
        attn_mask=attn_bias,   # float additive mask/bias or None
        dropout_p=dp,
        is_causal=is_causal,   # we encode causal via mask/bias when needed
    )  # [B,Hq,Sq,D] fp32

    return out.permute(0, 2, 1, 3).to(in_dtype).contiguous()  # [B,Sq,Hq,D]



def _attn_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    window_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Dispatches to either flash attention or the SDPA fallback.  This function
    accepts and returns tensors shaped ``[batch, seq_len, num_heads, head_dim]``.
    """
    if HAVE_FLASH_ATTN:
        # If flash attention is available we use it directly.  Note that
        # ``flash_attn_func`` accepts the same tensor layout and returns a
        # tensor with identical shape.  Additional keyword arguments such as
        # ``softmax_scale`` and ``dropout_p`` will use default values.
        return _flash_attn_func(
            q,
            k,
            v,
            causal=causal,
            window_size=window_size,
        )
    # Otherwise use the fallback implementation.
    return _sdpa_flash_attn_compat(
        q,
        k,
        v,
        causal=causal,
        window_size=window_size,
    )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Optional[torch.Tensor],
    k: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Applies rotary position embeddings to the query and key tensors.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if q is not None:
        q_embed = (q * cos) + (rotate_half(q) * sin)
    else:
        q_embed = None
    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        k_embed = None
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent to ``torch.repeat_interleave(x, dim=1, repeats=n_rep)``.  Converts
    hidden states from shape (batch, num_key_value_heads, seq_len, head_dim) to
    (batch, num_attention_heads, seq_len, head_dim).
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RotaryEmbedding(nn.Module):
    """
    Computes rotary position embeddings.  See
    https://arxiv.org/abs/2104.09864 for details.
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids is None:
            # position_ids shape: [batch, seq_len]
            position_ids = torch.arange(x.shape[2], device=x.device, dtype=torch.int64).unsqueeze(0).expand(x.shape[0], -1)
        # x shape: [batch, num_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        # Force float32 for numerical stability on long contexts.
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    """
    Root Mean Square layer normalization.  Equivalent to T5LayerNorm.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Attention(nn.Module):
    """
    Multi‑head attention module with optional rotary positional embeddings and
    windowed attention.  Uses flash attention when available, otherwise falls
    back to PyTorch's scaled dot product attention.
    """

    def __init__(
        self,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_head_size: int,
        attention_window_size: Optional[int] = None,
        seq_length: Optional[int] = None,
        use_positional_embedding: bool = False,
        rope_base: Optional[int] = None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_head_size = attention_head_size
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_window_size = attention_window_size
        self.seq_length = seq_length
        self.use_positional_embedding = use_positional_embedding
        self.rope_base = rope_base
        if self.use_positional_embedding:
            self.rotary_emb = RotaryEmbedding(dim=self.attention_head_size, base=self.rope_base)

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        bsz, q_len, _ = query_states.size()
        # Reshape to [batch, seq_len, num_heads, head_dim] and bring heads to axis 2.
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2).contiguous()
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.attention_head_size).transpose(1, 2).contiguous()
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.attention_head_size).transpose(1, 2).contiguous()
        # Repeat keys/values if there are more query heads than key/value heads.
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # Apply rotary positional embeddings if requested.
        if self.use_positional_embedding:
            cos, sin = self.rotary_emb(query_states)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # Move the seq_len dimension back to axis 1: [B, S, H, D].
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        # Compute attention.  Window size is specified as a tuple when present.
        if self.attention_window_size is not None:
            ws = (self.attention_window_size, self.attention_window_size)
        else:
            ws = None
        attn_outputs = _attn_dispatch(
            query_states,
            key_states,
            value_states,
            causal=True,
            window_size=ws,
        )
        # Merge heads back: [B, S, H*D].
        attn_outputs = attn_outputs.reshape(bsz, q_len, int(self.num_attention_heads * self.attention_head_size)).contiguous()
        return attn_outputs


class Block(nn.Module):
    """
    Basic transformer block consisting of an input projection into query/key/value
    and residual channels, a single attention layer, layer normalization and an
    output projection.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        attention_window_size: Optional[int] = None,
        seq_length: Optional[int] = None,
        use_positional_embedding: bool = False,
        rope_base: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        # In this architecture the intermediate size equals the hidden size.
        self.intermediate_size = self.hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_head_size = int(self.intermediate_size / self.num_attention_heads)
        # The latent dimension contains the residual channel (intermediate_size)
        # plus separate query/key/value projections.  The factor of 2 accounts
        # for concatenated key and value tensors.
        self.latent_dim = self.intermediate_size + self.attention_head_size * self.num_key_value_heads * 2
        self.pre_avg_layernorm = RMSNorm(self.intermediate_size)
        self.in_proj = nn.Linear(self.hidden_size, self.latent_dim, bias=True)
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.self_attn = Attention(
            self.num_attention_heads,
            self.num_key_value_heads,
            self.attention_head_size,
            attention_window_size,
            seq_length,
            use_positional_embedding,
            rope_base,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        # Project to queries, keys, values, and residuals.
        hidden_states = self.in_proj(hidden_states).transpose(1, 2)
        # Split into (q,k,v,residual).  Note: tensor_split returns views.
        q, k, v, residual = hidden_states.tensor_split(
            (
                self.intermediate_size,
                self.intermediate_size + self.attention_head_size * self.num_key_value_heads,
                self.intermediate_size + self.attention_head_size * self.num_key_value_heads * 2,
            ),
            dim=1,
        )
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Apply self attention.
        attn_outputs = self.self_attn(
            query_states=q,
            key_states=k,
            value_states=v,
        )
        # Normalize and project back to hidden size.
        hidden_states = self.pre_avg_layernorm(attn_outputs)
        contextualized_states = self.out_proj(hidden_states)
        return contextualized_states
