"""Window-based 3D bidirectional cross-attention between T0 and T1 encoder feature maps."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# Window partition helpers
def _partition_valid_mask(
    mask: torch.Tensor,
    window_size: Tuple[int, int, int],
) -> torch.Tensor:
    """Partition a bool mask into windows.

    Args:
        mask: Bool mask, shape `(B, 1, D, H, W)`.
        window_size: Window extents `(wD, wH, wW)`.

    Returns:
        Windowed bool mask, shape `(B·nW, L)` where `L = wD*wH*wW`.
    """
    B, _, D, H, W = mask.shape
    wD, wH, wW = window_size
    pad_d = (wD - D % wD) % wD
    pad_h = (wH - H % wH) % wH
    pad_w = (wW - W % wW) % wW
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        # Zero-pad so window-alignment fill reads as invalid, not real content.
        mask = F.pad(mask, (0, pad_w, 0, pad_h, 0, pad_d), value=0)

    _, _, D_pad, H_pad, W_pad = mask.shape
    nD = D_pad // wD
    nH = H_pad // wH
    nW_count = W_pad // wW
    m = mask.reshape(B, 1, nD, wD, nH, wH, nW_count, wW)
    m = m.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
    m = m.reshape(B * nD * nH * nW_count, wD * wH * wW)
    return m.bool()


def _partition_windows(
    x: torch.Tensor,
    window_size: Tuple[int, int, int],
) -> Tuple[torch.Tensor, Tuple[int, int, int], Tuple[int, int, int]]:
    """Reshape `(B, C, D, H, W)` -> `(B·nW, L, C)` by partitioning into 3D windows.

    Args:
        x: Feature map, shape `(B, C, D, H, W)`.
        window_size: Window extents `(wD, wH, wW)`.

    Returns:
        Tuple of windowed features `(B·nW, L, C)`, the original spatial dims
        `(D, H, W)`, and the padded spatial dims `(D_pad, H_pad, W_pad)`.
    """
    B, C, D, H, W = x.shape
    wD, wH, wW = window_size

    # F.pad order is reversed: (W_lo, W_hi, H_lo, H_hi, D_lo, D_hi)
    pad_d = (wD - D % wD) % wD
    pad_h = (wH - H % wH) % wH
    pad_w = (wW - W % wW) % wW
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

    _, _, D_pad, H_pad, W_pad = x.shape
    nD = D_pad // wD
    nH = H_pad // wH
    nW_count = W_pad // wW

    x = x.reshape(B, C, nD, wD, nH, wH, nW_count, wW)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
    windows = x.reshape(B * nD * nH * nW_count, wD * wH * wW, C)

    return windows, (D, H, W), (D_pad, H_pad, W_pad)


def _unpartition_windows(
    windows: torch.Tensor,
    window_size: Tuple[int, int, int],
    orig_dims: Tuple[int, int, int],
    pad_dims: Tuple[int, int, int],
    B: int,
) -> torch.Tensor:
    """Inverse of `_partition_windows`: reassemble `(B·nW, L, C)` -> `(B, C, D, H, W)`.

    Args:
        windows: Windowed features, shape `(B·nW, L, C)`.
        window_size: Window extents `(wD, wH, wW)` used to partition.
        orig_dims: Original (pre-padding) spatial dims `(D, H, W)`.
        pad_dims: Padded spatial dims `(D_pad, H_pad, W_pad)`.
        B: Batch size.

    Returns:
        Reassembled feature map, shape `(B, C, D, H, W)`, padding cropped.
    """
    D, H, W = orig_dims
    D_pad, H_pad, W_pad = pad_dims
    wD, wH, wW = window_size
    C = windows.shape[-1]

    nD = D_pad // wD
    nH = H_pad // wH
    nW_count = W_pad // wW

    windows = windows.reshape(B, nD, nH, nW_count, wD, wH, wW, C)
    x = windows.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    x = x.reshape(B, C, D_pad, H_pad, W_pad)
    x = x[:, :, :D, :H, :W].contiguous()

    return x


def _build_relative_position_index(
    window_size: Tuple[int, int, int],
) -> torch.Tensor:
    """Return the `(L, L)` long index into a relative-position-bias table.

    Args:
        window_size: Window extents `(wD, wH, wW)`.
    """
    wD, wH, wW = window_size
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(wD), torch.arange(wH), torch.arange(wW), indexing="ij"
        )
    )  # (3, wD, wH, wW)
    coords_flat = coords.reshape(3, -1)  # (3, L)
    rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (3, L, L)
    rel = rel.permute(1, 2, 0).contiguous()  # (L, L, 3)
    rel[:, :, 0] += wD - 1
    rel[:, :, 1] += wH - 1
    rel[:, :, 2] += wW - 1
    rel[:, :, 0] *= (2 * wH - 1) * (2 * wW - 1)
    rel[:, :, 1] *= (2 * wW - 1)
    return rel.sum(-1)  # (L, L) long


def _compute_shift_attn_mask(
    pad_dims: Tuple[int, int, int],
    window_size: Tuple[int, int, int],
    shift: Tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """Swin cyclic-shift attention mask.

    Args:
        pad_dims: Padded spatial dims `(D_pad, H_pad, W_pad)`.
        window_size: Window extents `(wD, wH, wW)`.
        shift: Cyclic shift per axis `(sD, sH, sW)`.
        device: Device to build the mask on.

    Returns:
        Additive attention mask, shape `(nW, L, L)`; 0 within a shift region,
        -100 across region boundaries.
    """
    Dp, Hp, Wp = pad_dims
    wD, wH, wW = window_size
    sD, sH, sW = shift
    img_mask = torch.zeros((1, 1, Dp, Hp, Wp), device=device)
    cnt = 0

    def _slices(dim: int, w: int, s: int):
        if s == 0:
            return (slice(0, dim),)
        return (slice(0, dim - w), slice(dim - w, dim - s), slice(dim - s, dim))

    for d in _slices(Dp, wD, sD):
        for h in _slices(Hp, wH, sH):
            for w in _slices(Wp, wW, sW):
                img_mask[:, :, d, h, w] = cnt
                cnt += 1

    mask_windows, _, _ = _partition_windows(img_mask, window_size)  # (nW, L, 1)
    mask_windows = mask_windows.squeeze(-1)  # (nW, L)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, L, L)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
        attn_mask == 0, 0.0
    )
    return attn_mask  # (nW, L, L)


class CrossTimepointAttentionStage(nn.Module):
    """Bidirectional window cross-attention for a single encoder stage.

    Args:
        channels: Number of feature channels at this stage.
        num_heads: Attention heads; auto-chosen from `channels` when `None`.
        window_size: Window extents `(wD, wH, wW)`.
        dropout: Dropout applied to attention weights and output projections.
        shift: If `True`, cyclically shift by half a window.
        use_rel_pos_bias: Add a learnable relative-position bias per head.
        use_null_key: Add a learnable sink key/value per direction.
        use_ffn: Apply a post-fusion feedforward block with a residual.
        use_layerscale: Scale residual branches by a learnable per-channel factor.
        mlp_ratio: Hidden-dim multiplier for the FFN.
        layerscale_init: Initial value for the LayerScale parameters.
    """

    def __init__(
        self,
        channels: int,
        num_heads: Optional[int] = None,
        window_size: Tuple[int, int, int] = (4, 4, 4),
        dropout: float = 0.0,
        shift: bool = False,
        use_rel_pos_bias: bool = True,
        use_null_key: bool = True,
        use_ffn: bool = True,
        use_layerscale: bool = True,
        mlp_ratio: float = 2.0,
        layerscale_init: float = 1e-4,
    ) -> None:
        super().__init__()

        if num_heads is None:
            num_heads = max(1, min(8, channels // 32))
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})."
            )

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_null_key = use_null_key
        self.use_ffn = use_ffn
        self.attn_dropout = dropout

        if shift:
            self.shift = tuple(w // 2 for w in window_size)
            self._shifted = any(s > 0 for s in self.shift)
        else:
            self.shift = (0, 0, 0)
            self._shifted = False

        self.norm_T0 = nn.LayerNorm(channels)
        self.norm_T1 = nn.LayerNorm(channels)

        self.q_T1 = nn.Linear(channels, channels)   # T1->T0: query from T1
        self.kv_T0 = nn.Linear(channels, 2 * channels)
        self.q_T0 = nn.Linear(channels, channels)   # T0->T1: query from T0
        self.kv_T1 = nn.Linear(channels, 2 * channels)
        self.out_T1 = nn.Linear(channels, channels)
        self.out_T0 = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(dropout)

        L = window_size[0] * window_size[1] * window_size[2]
        if use_rel_pos_bias:
            n_rel = (
                (2 * window_size[0] - 1)
                * (2 * window_size[1] - 1)
                * (2 * window_size[2] - 1)
            )
            self.rel_pos_bias_table = nn.Parameter(torch.zeros(n_rel, num_heads))
            self.register_buffer(
                "rel_pos_index",
                _build_relative_position_index(window_size).reshape(-1),
                persistent=False,
            )

        if use_null_key:
            self.null_kv_T0 = nn.Parameter(torch.zeros(1, 1, 2 * channels))
            self.null_kv_T1 = nn.Parameter(torch.zeros(1, 1, 2 * channels))

        # Fuse T0 and T1 features after bidirectional attention.
        self.fusion_proj = nn.Conv3d(2 * channels, channels, kernel_size=1, bias=True)
        self.norm_fused = nn.LayerNorm(channels)

        if use_ffn:
            hidden = int(channels * mlp_ratio)
            self.ffn_norm = nn.LayerNorm(channels)
            self.ffn = nn.Sequential(
                nn.Conv3d(channels, hidden, kernel_size=1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv3d(hidden, channels, kernel_size=1),
                nn.Dropout(dropout),
            )

        if use_layerscale:
            self.ls_T0 = nn.Parameter(layerscale_init * torch.ones(channels))
            self.ls_T1 = nn.Parameter(layerscale_init * torch.ones(channels))
            self.ls_ffn = nn.Parameter(layerscale_init * torch.ones(channels))
        else:
            self.register_parameter("ls_T0", None)
            self.register_parameter("ls_T1", None)
            self.register_parameter("ls_ffn", None)

        self._L = L

    # helpers
    def _rel_bias(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return `(1, heads, L, L)` relative-position bias, excluding any null key.

        Args:
            device: Device to build the bias tensor on.
            dtype: Dtype to cast the bias tensor to.
        """
        L = self._L
        bias = self.rel_pos_bias_table[self.rel_pos_index]  # (L*L, heads)
        bias = bias.reshape(L, L, self.num_heads).permute(2, 0, 1)  # (heads, L, L)
        return bias.unsqueeze(0).to(dtype=dtype, device=device)

    def _attend(
        self,
        q_win: torch.Tensor,
        kv_win: torch.Tensor,
        q_proj: nn.Linear,
        kv_proj: nn.Linear,
        out_proj: nn.Linear,
        null_kv: Optional[torch.Tensor],
        key_padding: Optional[torch.Tensor],
        shift_mask: Optional[torch.Tensor],
        q_valid: Optional[torch.Tensor],
        return_attn: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Manual multi-head cross-attention within a window.

        Args:
            q_win: Pre-norm query windows, shape `(Bw, L, C)`.
            kv_win: Pre-norm key/value windows, shape `(Bw, L, 2C)`.
            q_proj: Query projection for this direction.
            kv_proj: Combined key/value projection for this direction.
            out_proj: Output projection for this direction.
            null_kv: Optional learnable sink key/value, shape `(1, 1, 2C)`.
            key_padding: Bool mask, shape `(Bw, L)`, `True` = mask key.
            shift_mask: Additive float mask, shape `(Bw, L, L)`, or `None`.
            q_valid: Bool mask, shape `(Bw, L)`, `True` = valid query.
            return_attn: If `True`, compute attention manually instead of
                using the fused SDPA kernel.

        Returns:
            Tuple of the attended output, shape `(Bw, L, C)`, and the
            key-importance map, shape `(Bw, L)`, or `None` when
            `return_attn=False`.
        """
        Bw, L, C = q_win.shape
        H, hd = self.num_heads, self.head_dim

        q = q_proj(q_win)                                   # (Bw, L, C)
        kv = kv_proj(kv_win)                                # (Bw, L, 2C)
        if null_kv is not None:
            kv = torch.cat([kv, null_kv.expand(Bw, -1, -1)], dim=1)  # (Bw, L+1, 2C)
        k, v = kv.split(C, dim=-1)
        Lk = k.shape[1]

        q = q.reshape(Bw, L, H, hd).transpose(1, 2)         # (Bw, H, L, hd)
        k = k.reshape(Bw, Lk, H, hd).transpose(1, 2)        # (Bw, H, Lk, hd)
        v = v.reshape(Bw, Lk, H, hd).transpose(1, 2)

        # (1, 1, L, Lk)
        bias = torch.zeros(1, 1, L, Lk, device=q.device, dtype=torch.float32)
        if self.use_rel_pos_bias:
            rb = self._rel_bias(q.device, torch.float32)    # (1, H, L, L)
            if Lk > L:  # null key column gets zero bias
                rb = F.pad(rb, (0, Lk - L))
            bias = bias + rb
        if shift_mask is not None:
            sm = shift_mask.unsqueeze(1)                    # (Bw, 1, L, L)
            if Lk > L:
                sm = F.pad(sm, (0, Lk - L))                 # null column attendable
            bias = bias + sm
        if key_padding is not None:
            kp = key_padding
            if Lk > L:  # null key never masked
                kp = F.pad(kp, (0, Lk - L), value=False)
            pad_bias = torch.zeros_like(kp, dtype=torch.float32)
            pad_bias = pad_bias.masked_fill(kp, float("-inf"))
            bias = bias + pad_bias[:, None, None, :]        # (Bw, 1, 1, Lk)

        if return_attn:
            scores = (q.float() @ k.float().transpose(-2, -1)) * self.scale
            scores = scores + bias
            attn = scores.softmax(dim=-1)                   # (Bw, H, L, Lk)
            attn = F.dropout(attn, p=self.attn_dropout, training=self.training)
            out = (attn.to(v.dtype) @ v).transpose(1, 2).reshape(Bw, L, C)

            real = attn[..., :L]                            # (Bw, H, L, L)
            if q_valid is not None:
                real = real * q_valid[:, None, :, None].to(real.dtype)
            key_imp = real.sum(dim=-2)                      # (Bw, H, L) sum over queries
            key_imp = key_imp.amax(dim=1)                   # (Bw, L) max over heads
        else:
            attn_mask = bias.to(q.dtype).expand(Bw, H, L, Lk)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
            )
            out = out.transpose(1, 2).reshape(Bw, L, C)
            key_imp = None

        out = self.proj_drop(out_proj(out))
        return out, key_imp


    def forward(
        self,
        feat_T0: torch.Tensor,
        feat_T1: torch.Tensor,
        valid_T0: Optional[torch.Tensor] = None,
        valid_T1: Optional[torch.Tensor] = None,
        return_attn: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply bidirectional window cross-attention.

        Args:
            feat_T0: T0 feature map, shape `(B, C, D, H, W)`.
            feat_T1: T1 feature map, same shape as `feat_T0`.
            valid_T0: Optional bool valid mask for T0, shape `(B, 1, D, H, W)`.
            valid_T1: Optional bool valid mask for T1, same shape as `valid_T0`.
            return_attn: If `True`, materialise key-importance attention maps
                in the returned dict.

        Returns:
            Tuple of the fused feature map, shape `(B, C, D, H, W)`, and a
            dict of attention maps (`'T1_to_T0'`, `'T0_to_T1'`), empty when
            `return_attn=False`.
        """
        if return_attn is None:
            return_attn = False
        B = feat_T0.shape[0]

        # Pre-norm (channel-last LayerNorm).
        def _prenorm(f: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
            return ln(f.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        f0n = _prenorm(feat_T0, self.norm_T0)
        f1n = _prenorm(feat_T1, self.norm_T1)

        v0 = valid_T0
        v1 = valid_T1

        # Swin cyclic shift
        if self._shifted:
            sD, sH, sW = self.shift
            roll = (-sD, -sH, -sW)
            f0n = torch.roll(f0n, shifts=roll, dims=(2, 3, 4))
            f1n = torch.roll(f1n, shifts=roll, dims=(2, 3, 4))
            if v0 is not None:
                v0 = torch.roll(v0, shifts=roll, dims=(2, 3, 4))
            if v1 is not None:
                v1 = torch.roll(v1, shifts=roll, dims=(2, 3, 4))

        # Partition -> (B·nW, L, C).
        win_T0, orig_dims, pad_dims = _partition_windows(f0n, self.window_size)
        win_T1, _, _ = _partition_windows(f1n, self.window_size)

        # Key-padding + query-validity per window.
        kp_T0 = (~_partition_valid_mask(v0, self.window_size)) if v0 is not None else None
        kp_T1 = (~_partition_valid_mask(v1, self.window_size)) if v1 is not None else None
        qv_T1 = _partition_valid_mask(v1, self.window_size) if v1 is not None else None  # T1->T0 queries
        qv_T0 = _partition_valid_mask(v0, self.window_size) if v0 is not None else None  # T0->T1 queries

        # Without a null key, a fully-masked window would softmax to NaN; guard here.
        if not self.use_null_key:
            if kp_T0 is not None:
                kp_T0 = kp_T0 & ~kp_T0.all(dim=1, keepdim=True)
            if kp_T1 is not None:
                kp_T1 = kp_T1 & ~kp_T1.all(dim=1, keepdim=True)

        # Shift mask.
        shift_mask = None
        if self._shifted:
            sm = _compute_shift_attn_mask(
                pad_dims, self.window_size, self.shift, win_T0.device
            )  # (nW, L, L)
            shift_mask = sm.repeat(B, 1, 1)  # (B·nW, L, L)

        null_T0 = self.null_kv_T0 if self.use_null_key else None
        null_T1 = self.null_kv_T1 if self.use_null_key else None

        out_T1, imp_T0 = self._attend(
            win_T1, win_T0, self.q_T1, self.kv_T0, self.out_T1,
            null_T0, kp_T0, shift_mask, qv_T1, return_attn,
        )
        out_T0, imp_T1 = self._attend(
            win_T0, win_T1, self.q_T0, self.kv_T1, self.out_T0,
            null_T1, kp_T1, shift_mask, qv_T0, return_attn,
        )

        out_T1_sp = _unpartition_windows(out_T1, self.window_size, orig_dims, pad_dims, B)
        out_T0_sp = _unpartition_windows(out_T0, self.window_size, orig_dims, pad_dims, B)

        # Undo the shift on the attention outputs.
        if self._shifted:
            sD, sH, sW = self.shift
            unroll = (sD, sH, sW)
            out_T1_sp = torch.roll(out_T1_sp, shifts=unroll, dims=(2, 3, 4))
            out_T0_sp = torch.roll(out_T0_sp, shifts=unroll, dims=(2, 3, 4))

        # LayerScale residual updates.
        if self.ls_T0 is not None:
            out_T0_sp = out_T0_sp * self.ls_T0.view(1, -1, 1, 1, 1)
            out_T1_sp = out_T1_sp * self.ls_T1.view(1, -1, 1, 1, 1)
        updated_T0 = feat_T0 + out_T0_sp
        updated_T1 = feat_T1 + out_T1_sp

        # Fuse: concat -> 1×1×1 conv -> post-norm.
        fused = self.fusion_proj(torch.cat([updated_T0, updated_T1], dim=1))
        fused = self.norm_fused(
            fused.permute(0, 2, 3, 4, 1)
        ).permute(0, 4, 1, 2, 3).contiguous()

        # FFN with LayerScale residual.
        if self.use_ffn:
            ff_in = self.ffn_norm(fused.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            ff = self.ffn(ff_in)
            if self.ls_ffn is not None:
                ff = ff * self.ls_ffn.view(1, -1, 1, 1, 1)
            fused = fused + ff

        attn_maps: Dict[str, torch.Tensor] = {}
        if return_attn and imp_T0 is not None and imp_T1 is not None:
            map_T0 = _unpartition_windows(
                imp_T0.unsqueeze(-1), self.window_size, orig_dims, pad_dims, B
            )
            map_T1 = _unpartition_windows(
                imp_T1.unsqueeze(-1), self.window_size, orig_dims, pad_dims, B
            )
            if self._shifted:
                sD, sH, sW = self.shift
                unroll = (sD, sH, sW)
                map_T0 = torch.roll(map_T0, shifts=unroll, dims=(2, 3, 4))
                map_T1 = torch.roll(map_T1, shifts=unroll, dims=(2, 3, 4))
            attn_maps = {"T1_to_T0": map_T0, "T0_to_T1": map_T1}

        return fused, attn_maps


class CrossTimepointAttentionStack(nn.Module):
    """One `CrossTimepointAttentionStage` per encoder stage.

    Args:
        features_per_stage: Channel dims for each encoder stage.
        window_sizes: A single `(wD, wH, wW)` shared across stages, a list
            with one entry per stage, or `None` for a `(4, 4, 4)` default.
        num_heads_per_stage: Attention heads per stage; `None` entries
            auto-choose from the stage's channel count.
        dropout: Dropout passed to every stage.
        use_gradient_checkpointing: Checkpoint each active stage during training.
        pass_through_stages: Stage indices that skip cross-attention and use
            a mean blend instead.
        use_shifted_windows: Alternate the Swin half-window shift across
            active stages.
        use_rel_pos_bias: Forwarded to each stage.
        use_null_key: Forwarded to each stage.
        use_ffn: Forwarded to each stage.
        use_layerscale: Forwarded to each stage.
        mlp_ratio: Forwarded to each stage.
    """

    def __init__(
        self,
        features_per_stage: Tuple[int, ...],
        window_sizes: Optional[
            Union[Tuple[int, int, int], List[Tuple[int, int, int]]]
        ] = None,
        num_heads_per_stage: Optional[List[Optional[int]]] = None,
        dropout: float = 0.0,
        use_gradient_checkpointing: bool = False,
        pass_through_stages: Optional[Sequence[int]] = None,
        use_shifted_windows: bool = True,
        use_rel_pos_bias: bool = True,
        use_null_key: bool = True,
        use_ffn: bool = True,
        use_layerscale: bool = True,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.pass_through_stages = set(int(s) for s in (pass_through_stages or ()))

        n = len(features_per_stage)

        if window_sizes is None:
            ws_list: List[Tuple[int, int, int]] = [(4, 4, 4)] * n
        elif isinstance(window_sizes[0], int):
            ws_list = [tuple(window_sizes)] * n  # type: ignore[assignment]
        else:
            ws_list = list(window_sizes)  # type: ignore[arg-type]
            if len(ws_list) != n:
                raise ValueError(
                    f"window_sizes has {len(ws_list)} entries but "
                    f"features_per_stage has {n}."
                )

        if num_heads_per_stage is None:
            heads_list: List[Optional[int]] = [None] * n
        else:
            if len(num_heads_per_stage) != n:
                raise ValueError(
                    f"num_heads_per_stage has {len(num_heads_per_stage)} entries "
                    f"but features_per_stage has {n}."
                )
            heads_list = list(num_heads_per_stage)

        # Alternate the Swin shift across active stages.
        active_idx = [i for i in range(n) if i not in self.pass_through_stages]
        shift_flags = {idx: (rank % 2 == 1) for rank, idx in enumerate(active_idx)}

        self.stages = nn.ModuleList(
            [
                CrossTimepointAttentionStage(
                    channels=features_per_stage[i],
                    num_heads=heads_list[i],
                    window_size=ws_list[i],
                    dropout=dropout,
                    shift=use_shifted_windows and shift_flags.get(i, False),
                    use_rel_pos_bias=use_rel_pos_bias,
                    use_null_key=use_null_key,
                    use_ffn=use_ffn,
                    use_layerscale=use_layerscale,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(n)
            ]
        )

        # Per-channel difference scale for pass-through fusion; starts at 0, a mean blend.
        self.pt_diff_scale = nn.ParameterList(
            [nn.Parameter(torch.zeros(features_per_stage[i])) for i in range(n)]
        )

    def forward(
        self,
        stage_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        valid_pairs: Optional[List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]] = None,
        return_attn: Optional[bool] = None,
        skip_attn_stages: Sequence[int] = (),
    ) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """Apply cross-timepoint attention at every stage.

        Args:
            stage_pairs: One `(feat_T0, feat_T1)` tuple per encoder stage.
            valid_pairs: Optional one `(valid_T0, valid_T1)` tuple per stage.
            return_attn: If `True`, materialise attention maps for stages not
                listed in `skip_attn_stages`.
            skip_attn_stages: Stage indices to exclude from attention-map
                materialisation even when `return_attn=True`.

        Returns:
            Tuple of the fused feature map per stage and the attention-map
            dict per stage (empty dict for pass-through or skipped stages).

        Raises:
            ValueError: If `stage_pairs` or `valid_pairs` doesn't have one
                entry per configured stage.
        """
        if len(stage_pairs) != len(self.stages):
            raise ValueError(
                f"Expected {len(self.stages)} stage pairs, got {len(stage_pairs)}."
            )
        if valid_pairs is not None and len(valid_pairs) != len(self.stages):
            raise ValueError(
                f"valid_pairs has {len(valid_pairs)} entries but stack has "
                f"{len(self.stages)} stages."
            )

        fused_maps: List[torch.Tensor] = []
        attn_maps_per_stage: List[Dict[str, torch.Tensor]] = []

        from torch.utils.checkpoint import checkpoint

        skip_set = set(skip_attn_stages)
        for i, (stage_module, (feat_T0, feat_T1)) in enumerate(zip(self.stages, stage_pairs)):
            # Pass-through: mean blend plus a gamma-scaled difference, no cross-attention.
            if i in self.pass_through_stages:
                gamma = self.pt_diff_scale[i].view(1, -1, 1, 1, 1)
                fused_maps.append(0.5 * (feat_T0 + feat_T1) + gamma * (feat_T1 - feat_T0))
                attn_maps_per_stage.append({})
                continue

            v0, v1 = (valid_pairs[i] if valid_pairs is not None else (None, None))
            stage_return = bool(return_attn) and i not in skip_set
            use_ckpt = (
                self.use_gradient_checkpointing and self.training and not stage_return
            )
            if use_ckpt:
                def _run(f0, f1, _mod=stage_module, _v0=v0, _v1=v1):
                    return _mod(f0, f1, valid_T0=_v0, valid_T1=_v1, return_attn=False)
                fused, attn_maps = checkpoint(_run, feat_T0, feat_T1, use_reentrant=False)
            else:
                fused, attn_maps = stage_module(
                    feat_T0, feat_T1,
                    valid_T0=v0, valid_T1=v1,
                    return_attn=stage_return,
                )
            fused_maps.append(fused)
            attn_maps_per_stage.append(attn_maps)

        return fused_maps, attn_maps_per_stage
