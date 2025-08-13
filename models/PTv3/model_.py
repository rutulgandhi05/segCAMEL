# models/PTv3/model.py
"""
Point Transformer V3 (Pointcept-style) with stability guards,
**no environment-variable overrides**.

Everything is configured via constructor kwargs only:
  - enable_flash (bool): use FlashAttention path if the package is available.
  - enc/dec_patch_size (tuple[int]): per-stage serialized attention window.
  - enable_rpe, upcast_attention, upcast_softmax: classic-attention options.
"""

import sys
import math
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
import torch_scatter
from addict import Dict
from timm.layers import DropPath

# FlashAttention availability
try:
    import flash_attn  # noqa: F401
    _HAVE_FLASH = True
except Exception:
    flash_attn = None
    _HAVE_FLASH = False

from .serialization import encode  # Pointcept’s Morton/Hilbert utilities


# ---------------------------
# Small functional utilities
# ---------------------------
def _offset2bincount(offset: torch.Tensor) -> torch.Tensor:
    return torch.diff(offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long))

def _offset2batch(offset: torch.Tensor) -> torch.Tensor:
    bincount = _offset2bincount(offset)
    return torch.arange(len(bincount), device=offset.device, dtype=torch.long).repeat_interleave(bincount)

def _batch2offset(batch: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(batch.bincount(), dim=0).long()

@torch.inference_mode()
def offset2bincount(offset):  # kept for backward compat
    return _offset2bincount(offset)

@torch.inference_mode()
def offset2batch(offset):     # kept for backward compat
    return _offset2batch(offset)

@torch.inference_mode()
def batch2offset(batch):      # kept for backward compat
    return _batch2offset(batch)


# ---------------------------
# Core Point structure
# ---------------------------
class Point(Dict):
    """
    Pointcept-compatible point structure (batched point cloud).

    Required keys:
      - coord           : (N, 3) raw coordinates
      - grid_coord|coord+grid_size : discrete coords for voxel/grid (auto-built if missing)
      - feat            : (N, C) per-point features
    Batch keys (autofilled if one exists):
      - batch | offset

    Serialization keys (built by .serialization()):
      - serialized_depth, serialized_code, serialized_order, serialized_inverse

    SpConv keys (built by .sparsify()):
      - sparse_shape, sparse_conv_feat
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = _offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = _batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth

        # protect int64 code capacity (Pointcept rule of thumb)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        assert depth <= 16

        orders = [order] if isinstance(order, str) else order
        codes = [encode(self.grid_coord, self.batch, depth, order=o) for o in orders]
        code = torch.stack(codes)  # (k, n)
        order_idx = torch.argsort(code)
        inverse = torch.zeros_like(order_idx).scatter_(
            1, order_idx, torch.arange(code.shape[1], device=order_idx.device).repeat(code.shape[0], 1)
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0], device=code.device)
            code = code[perm]; order_idx = order_idx[perm]; inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order_idx
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(torch.max(self.grid_coord, dim=0).values, pad).tolist()

        sct = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat([self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].item() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sct


# ---------------------------
# Modules
# ---------------------------
class PointModule(nn.Module):
    """Base point-aware module; accepts/returns Point or SparseConvTensor."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class PointSequential(PointModule):
    """Sequential that routes Point / SparseConv / Tensor correctly (Pointcept style)."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f"index {idx} is out of range")
        if idx < 0: idx += len(self)
        it = iter(self._modules.values())
        for _ in range(idx): next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        name = str(len(self._modules)) if name is None else name
        if name in self._modules:
            raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, inp):
        for _, module in self._modules.items():
            # Point-aware submodule
            if isinstance(module, PointModule):
                inp = module(inp)
            # SpConv path
            elif spconv.modules.is_spconv_module(module):
                if isinstance(inp, Point):
                    inp.sparse_conv_feat = module(inp.sparse_conv_feat)
                    inp.feat = inp.sparse_conv_feat.features
                else:
                    inp = module(inp)
            # Plain torch
            else:
                if isinstance(inp, Point):
                    inp.feat = module(inp.feat)
                    if "sparse_conv_feat" in inp.keys():
                        inp.sparse_conv_feat = inp.sparse_conv_feat.replace_feature(inp.feat)
                elif isinstance(inp, spconv.SparseConvTensor):
                    if inp.indices.shape[0] != 0:
                        inp = inp.replace_feature(module(inp.features))
                else:
                    inp = module(inp)
        return inp


class PDNorm(PointModule):
    """Per-dataset norm wrapper (kept intact)."""
    def __init__(self, num_features, norm_layer, context_channels=256,
                 conditions=("ScanNet", "S3DIS", "Structured3D"),
                 decouple=True, adaptive=False):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.norm = norm_layer
        if self.adaptive:
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True))

    def forward(self, point):
        assert {"feat", "condition"}.issubset(point.keys())
        condition = point.condition if isinstance(point.condition, str) else point.condition[0]
        norm = self.norm[self.conditions.index(condition)] if self.decouple else self.norm
        point.feat = norm(point.feat)
        if self.adaptive:
            assert "context" in point.keys()
            shift, scale = self.modulation(point.context).chunk(2, dim=1)
            point.feat = point.feat * (1.0 + scale) + shift
        return point


class RPE(nn.Module):
    """Relative positional encoding (disabled when FlashAttention is enabled)."""
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (coord.clamp(-self.pos_bnd, self.pos_bnd)
               + self.pos_bnd
               + torch.arange(3, device=coord.device) * self.rpe_num)
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        return out.permute(0, 3, 1, 2)  # (N,H,K,K)


class SerializedAttention(PointModule):
    """
    Serialized attention with dual backend:
      - FlashAttention path (fp16; no RPE/upcast)
      - Classic path (supports RPE; can upcast qk/softmax to fp32)
    """
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index

        # Backend selection (constructor-only; no env)
        use_flash = bool(enable_flash) and _HAVE_FLASH
        if bool(enable_flash) and not _HAVE_FLASH:
            print("[PTv3] FlashAttention requested but not available; falling back to classic attention.")
        self.enable_flash = use_flash

        # Flash path forbids RPE/upcasting; classic path can use both
        if self.enable_flash:
            self.enable_rpe = False
            self.upcast_attention = False
            self.upcast_softmax = False
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            self.enable_rpe = bool(enable_rpe)
            self.upcast_attention = bool(upcast_attention)
            self.upcast_softmax = bool(upcast_softmax)
            self.patch_size_max = patch_size
            self.patch_size = 0  # will be set to the min bincount per forward
            self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if (not self.enable_flash and self.enable_rpe) else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        key = f"rel_pos_{self.order_index}"
        if key not in point.keys():
            grid = point.grid_coord[order].reshape(-1, K, 3)
            point[key] = grid.unsqueeze(2) - grid.unsqueeze(1)
        return point[key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key, unpad_key, cu_key = "pad", "unpad", "cu_seqlens_key"
        if (pad_key not in point.keys()) or (unpad_key not in point.keys()) or (cu_key not in point.keys()):
            offset = point.offset
            bincount = _offset2bincount(offset)
            bincount_pad = (torch.div(bincount + self.patch_size - 1, self.patch_size, rounding_mode="trunc") * self.patch_size)
            mask_pad = bincount > self.patch_size
            bincount_pad = (~mask_pad) * bincount + mask_pad * bincount_pad

            _off = F.pad(offset, (1, 0))
            _off_pad = F.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))

            pad = torch.arange(_off_pad[-1], device=offset.device)
            unpad = torch.arange(_off[-1], device=offset.device)
            cu = []
            for i in range(len(offset)):
                unpad[_off[i]:_off[i+1]] += _off_pad[i] - _off[i]
                if bincount[i] != bincount_pad[i]:
                    pad[_off_pad[i+1] - self.patch_size + (bincount[i] % self.patch_size): _off_pad[i+1]] = \
                        pad[_off_pad[i+1] - 2*self.patch_size + (bincount[i] % self.patch_size): _off_pad[i+1] - self.patch_size]
                pad[_off_pad[i]:_off_pad[i+1]] -= _off_pad[i] - _off[i]
                cu.append(torch.arange(_off_pad[i], _off_pad[i+1], step=self.patch_size,
                                       dtype=torch.int32, device=offset.device))
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_key] = F.pad(torch.concat(cu), (0, 1), value=_off_pad[-1])
        return point[pad_key], point[unpad_key], point[cu_key]

    def forward(self, point):
        if not self.enable_flash:
            # adapt patch size to the smallest sample in batch
            self.patch_size = min(_offset2bincount(point.offset).min().item(), self.patch_size_max)

        H, C = self.num_heads, self.channels
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        qkv = self.qkv(point.feat)[order]
        if not self.enable_flash:
            # classic attention path
            q, k, v = (qkv.reshape(-1, self.patch_size, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0))
            if self.upcast_attention:
                q = q.float(); k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)                # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            # flash-attention (requires fp16 qkvpacked)
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0.0,
                softmax_scale=self.scale,
            ).reshape(-1, C).to(qkv.dtype)

        feat = feat[inverse]
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

        nn.init.trunc_normal_(self.fc1.weight, std=0.02); nn.init.zeros_(self.fc1.bias)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02); nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels, num_heads,
        patch_size=48, mlp_ratio=4.0,
        qkv_bias=True, qk_scale=None,
        attn_drop=0.0, proj_drop=0.0,
        drop_path=0.0, norm_layer=nn.LayerNorm, act_layer=nn.GELU,
        pre_norm=True, order_index=0,
        cpe_indice_key=None,
        enable_rpe=False, enable_flash=True,
        upcast_attention=True, upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(channels, channels, kernel_size=3, bias=True, indice_key=cpe_indice_key),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels, patch_size=patch_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop,
            order_index=order_index, enable_rpe=enable_rpe, enable_flash=enable_flash,
            upcast_attention=upcast_attention, upcast_softmax=upcast_softmax
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(MLP(in_channels=channels, hidden_channels=int(channels * mlp_ratio),
                                       out_channels=channels, act_layer=act_layer, drop=proj_drop))
        self.drop_path = PointSequential(DropPath(drop_path) if drop_path > 0.0 else nn.Identity())

    def forward(self, point: Point):
        # CPE
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat

        # Attn
        shortcut = point.feat
        if self.pre_norm: point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm: point = self.norm1(point)

        # MLP
        shortcut = point.feat
        if self.pre_norm: point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm: point = self.norm2(point)

        # Keep sparse tensor feature in sync
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(self, in_channels, out_channels, stride=2,
                 norm_layer=None, act_layer=None, reduce="max",
                 shuffle_orders=True, traceable=True):
        super().__init__()
        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        assert reduce in ["sum", "mean", "min", "max"]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {"serialized_code","serialized_order","serialized_inverse","serialized_depth"}.issubset(point.keys()), \
            "Call point.serialization() before SerializedPooling."

        code = point.serialized_code >> pooling_depth * 3
        code0, cluster, counts = torch.unique(code[0], sorted=True, return_inverse=True, return_counts=True)
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]

        # build downsampled code/order/inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(1, order,
                   torch.arange(code.shape[1], device=order.device).repeat(code.shape[0], 1))

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0], device=code.device)
            code = code[perm]; order = order[perm]; inverse = inverse[perm]

        point_dict = Dict(
            feat=torch_scatter.segment_csr(self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce),
            coord=torch_scatter.segment_csr(point.coord[indices], idx_ptr, reduce="mean"),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code, serialized_order=order, serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )
        if "condition" in point.keys(): point_dict["condition"] = point.condition
        if "context" in point.keys():   point_dict["context"]   = point.context
        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"]  = point

        point = Point(point_dict)
        if hasattr(self, "norm"): point = self.norm(point)
        if hasattr(self, "act"):  point  = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(self, in_channels, skip_channels, out_channels,
                 norm_layer=None, act_layer=None, traceable=False):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))
        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels)); self.proj_skip.add(norm_layer(out_channels))
        if act_layer is not None:
            self.proj.add(act_layer()); self.proj_skip.add(act_layer())
        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys() and "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent"); inverse = point.pop("pooling_inverse")
        point  = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]
        if self.traceable: parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(self, in_channels, embed_channels, norm_layer=None, act_layer=None):
        super().__init__()
        self.stem = PointSequential(conv=spconv.SubMConv3d(
            in_channels, embed_channels, kernel_size=5, padding=1, bias=False, indice_key="stem"))
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        return self.stem(point)


# ---------------------------
# PointTransformerV3 network
# ---------------------------
class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()

        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths) == len(enc_channels) == len(enc_num_head) == len(enc_patch_size)
        assert self.cls_mode or (self.num_stages == len(dec_depths) + 1 == len(dec_channels) + 1 == len(dec_num_head) + 1 == len(dec_patch_size) + 1)

        # Norm layers
        if pdnorm_bn:
            bn_layer = partial(PDNorm,
                norm_layer=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine),
                conditions=pdnorm_conditions, decouple=pdnorm_decouple, adaptive=pdnorm_adaptive)
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        if pdnorm_ln:
            ln_layer = partial(PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions, decouple=pdnorm_decouple, adaptive=pdnorm_adaptive)
        else:
            ln_layer = nn.LayerNorm

        act_layer = nn.GELU

        # Stem
        self.embedding = Embedding(in_channels=in_channels, embed_channels=enc_channels[0],
                                   norm_layer=bn_layer, act_layer=act_layer)

        # Encoder
        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_dp_slice = enc_drop_path[sum(enc_depths[:s]): sum(enc_depths[:s+1])]
            enc = PointSequential()
            if s > 0:
                enc.add(SerializedPooling(in_channels=enc_channels[s-1], out_channels=enc_channels[s],
                                          stride=stride[s-1], norm_layer=bn_layer, act_layer=act_layer),
                        name="down")
            for i in range(enc_depths[s]):
                enc.add(Block(channels=enc_channels[s], num_heads=enc_num_head[s],
                              patch_size=enc_patch_size[s], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=proj_drop, drop_path=enc_dp_slice[i],
                              norm_layer=ln_layer, act_layer=act_layer, pre_norm=True,
                              order_index=i % len(self.order), cpe_indice_key=f"stage{s}",
                              enable_rpe=enable_rpe, enable_flash=enable_flash,
                              upcast_attention=upcast_attention, upcast_softmax=upcast_softmax),
                       name=f"block{i}")
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # Decoder
        if not self.cls_mode:
            dec_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_dp_slice = dec_drop_path[sum(dec_depths[:s]): sum(dec_depths[:s+1])]
                dec_dp_slice.reverse()
                dec = PointSequential()
                dec.add(SerializedUnpooling(in_channels=dec_channels[s+1], skip_channels=enc_channels[s],
                                            out_channels=dec_channels[s], norm_layer=bn_layer, act_layer=act_layer),
                        name="up")
                for i in range(dec_depths[s]):
                    dec.add(Block(channels=dec_channels[s], num_heads=dec_num_head[s],
                                  patch_size=dec_patch_size[s], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  attn_drop=attn_drop, proj_drop=proj_drop, drop_path=dec_dp_slice[i],
                                  norm_layer=ln_layer, act_layer=act_layer, pre_norm=True,
                                  order_index=i % len(self.order), cpe_indice_key=f"stage{s}",
                                  enable_rpe=enable_rpe, enable_flash=enable_flash,
                                  upcast_attention=upcast_attention, upcast_softmax=upcast_softmax),
                           name=f"block{i}")
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        """
        Expects:
          - feat:        (N, C_in) per-point features
          - grid_coord:  (N, 3) int voxel coords OR ("coord" + "grid_size")
          - offset or batch
        """
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        if not self.cls_mode:
            point = self.dec(point)
        return point
