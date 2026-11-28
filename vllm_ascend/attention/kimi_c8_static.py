from __future__ import annotations

import functools
import logging
from dataclasses import dataclass

import torch

FP8_E4M3_MAX = 448.0
PA_NZ_TILE = 32
PA_NZ_BLOCK_SIZE = 128  # Prolog/V2 table granularity; FA uses cache.shape[3].
K3_MLA_CACHE_WIDTH = 576


logger = logging.getLogger(__name__)

_ANTI_QUANT_PROBE_CODES = (0x04, 0x38, 0x3C, 0x45, 0xB8, 0xC2)
_ANTI_QUANT_PROBE_SCALE = 0.1372941
_ANTI_QUANT_PROBE_ROWS = 2


def _quantize_static_fp8_validated(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    # Avoid mutating caller-owned FP32 inputs: .float() returns an alias on FP32,
    # whereas non-FP32 inputs allocate a fresh temporary that can be scaled in place.
    scaled = x.div(s) if x.dtype == torch.float32 else x.float().div_(s)
    return scaled.clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def quantize_static_fp8(x: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    s = torch.as_tensor(scale, device=x.device, dtype=torch.float32)
    if s.numel() != 1 or not bool(torch.isfinite(s).all()) or bool((s <= 0).any()):
        raise ValueError("static scale must be one finite positive scalar")
    return _quantize_static_fp8_validated(x, s)


def _probe_npu_fp8_anti_quant(anti_quant, device: torch.device, dtype: torch.dtype) -> bool:
    sample = torch.tensor(_ANTI_QUANT_PROBE_CODES, dtype=torch.uint8, device=device)
    total_elems = _ANTI_QUANT_PROBE_ROWS * K3_MLA_CACHE_WIDTH
    repeats = (total_elems + sample.numel() - 1) // sample.numel()
    probe_u8 = sample.repeat(repeats)[:total_elems].reshape(_ANTI_QUANT_PROBE_ROWS, 1, K3_MLA_CACHE_WIDTH)
    probe_s = torch.tensor([_ANTI_QUANT_PROBE_SCALE], dtype=torch.float32, device=device)
    try:
        actual = anti_quant(probe_u8.view(torch.float8_e4m3fn), probe_s, dst_dtype=dtype)
    except (NotImplementedError, RuntimeError) as exc:
        logger.info("C8 npu_anti_quant probe failed for (%s, %s); using FP32 dequant fallback: %s", device, dtype, exc)
        return False
    expected = probe_u8.view(torch.float8_e4m3fn).float().mul_(probe_s).to(dtype)
    matched = actual.shape == expected.shape and actual.dtype == dtype and bool(torch.equal(actual, expected))
    if not matched:
        logger.info("C8 npu_anti_quant probe parity mismatch for (%s, %s); using FP32 dequant fallback.", device, dtype)
    return matched


@functools.lru_cache(maxsize=8)
def resolve_npu_fp8_anti_quant(device: torch.device, dtype: torch.dtype):
    """Probe once per (device, dtype) with the real [T, 1, 576] layout and return the op handle or None."""
    if device.type != "npu" or dtype not in (torch.bfloat16, torch.float16):
        return None
    try:
        import torch_npu  # noqa: PLC0415
    except ImportError:
        return None
    anti_quant = getattr(torch_npu, "npu_anti_quant", None)
    if anti_quant is None or not _probe_npu_fp8_anti_quant(anti_quant, device, dtype):
        return None
    return anti_quant


def supports_npu_fp8_anti_quant(device: torch.device, dtype: torch.dtype) -> bool:
    return resolve_npu_fp8_anti_quant(device, dtype) is not None


def dequantize_static_fp8(x: torch.Tensor, scale: torch.Tensor | float, dtype=torch.bfloat16) -> torch.Tensor:
    s = torch.as_tensor(scale, device=x.device, dtype=torch.float32)
    if s.numel() != 1 or not bool(torch.isfinite(s).all()) or bool((s <= 0).any()):
        raise ValueError("static scale must be one finite positive scalar")
    return _dequantize_static_fp8(x, s, dtype)


def _dequantize_static_fp8(
    x: torch.Tensor,
    s: torch.Tensor,
    dtype: torch.dtype,
    *,
    anti_quant=None,
) -> torch.Tensor:
    if x.numel() == 0:
        return torch.empty(x.shape, dtype=dtype, device=x.device)
    if anti_quant is not None:
        return anti_quant(x.view(torch.float8_e4m3fn), s, dst_dtype=dtype)
    return x.view(torch.float8_e4m3fn).float().mul_(s).to(dtype)


def pa_nz_indices(
    slots: torch.Tensor, block_size: int, width: int, *, page_stride_bytes: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if width <= 0 or width % PA_NZ_TILE:
        raise ValueError("width must be a positive multiple of 32")
    _kernel_blocks_per_manager(block_size)
    page_stride_bytes = block_size * width if page_stride_bytes is None else page_stride_bytes
    flat = slots.reshape(-1).to(torch.int64)
    valid = flat >= 0
    slot = flat[valid]
    tiles = width // PA_NZ_TILE
    indices = (
        (slot // block_size)[:, None] * (page_stride_bytes // PA_NZ_TILE)
        + torch.arange(tiles, device=slot.device)[None, :] * block_size
        + slot.remainder(block_size)[:, None]
    ).reshape(-1)
    return valid, indices


def write_static_fp8_cache(merged: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor, scale) -> None:
    if merged.ndim != 2 or cache.ndim != 5 or cache.dtype != torch.uint8:
        raise ValueError("merged/cache layout is invalid")
    block_size, tiles = cache.shape[3], cache.shape[2]
    if cache.shape[1] != 1 or cache.shape[4] != PA_NZ_TILE or merged.shape[1] != tiles * PA_NZ_TILE:
        raise ValueError("cache and merged widths do not match PA_NZ")
    if slots.numel() != merged.shape[0]:
        raise ValueError("C8 writer requires one slot per input token")
    if bool((slots >= cache.shape[0] * block_size).any()):
        raise ValueError("C8 writer slot exceeds physical cache capacity")
    valid, indices = pa_nz_indices(slots, block_size, merged.shape[1], page_stride_bytes=cache.stride(0))
    rows = quantize_static_fp8(merged[valid], scale).reshape(-1, PA_NZ_TILE)
    pa_nz_storage_view(cache).view(-1, PA_NZ_TILE).index_copy_(0, indices, rows)


@dataclass(frozen=True)
class C8CacheWritePlan:
    """One step's validated PA_NZ scatter indices shared across MLA layers."""

    num_tokens: int
    valid: torch.Tensor
    indices: torch.Tensor
    all_valid: bool

    @classmethod
    def prepare(cls, cache: torch.Tensor, slots: torch.Tensor) -> C8CacheWritePlan:
        pa_nz_prolog_view(cache)
        block_size = cache.shape[3]
        if bool((slots >= cache.shape[0] * block_size).any()):
            raise ValueError("C8 writer slot exceeds physical cache capacity")
        valid, indices = pa_nz_indices(
            slots,
            block_size,
            K3_MLA_CACHE_WIDTH,
            page_stride_bytes=cache.stride(0),
        )
        return cls(
            num_tokens=slots.numel(),
            valid=valid,
            indices=indices,
            all_valid=bool(valid.all()),
        )

    def write(self, merged: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor, validated_scale) -> None:
        if merged.ndim != 2 or merged.shape[0] != self.num_tokens or merged.shape[1] != K3_MLA_CACHE_WIDTH:
            raise ValueError("merged/cache layout is invalid")
        if self.indices.numel() == 0:
            return
        src = merged if self.all_valid else merged[self.valid]
        if (
            isinstance(validated_scale, torch.Tensor)
            and validated_scale.device == src.device
            and validated_scale.dtype == torch.float32
        ):
            s = validated_scale
        else:
            s = torch.as_tensor(validated_scale, device=src.device, dtype=torch.float32)
        rows = _quantize_static_fp8_validated(src, s)
        pa_nz_storage_view(cache).view(-1, PA_NZ_TILE).index_copy_(0, self.indices, rows.reshape(-1, PA_NZ_TILE))


def pa_nz_cache_view(
    raw: torch.Tensor,
    num_blocks: int,
    block_size: int,
    width: int = K3_MLA_CACHE_WIDTH,
    *,
    page_stride_bytes: int | None = None,
) -> torch.Tensor:
    """Expose one native PA_NZ block per manager page, including page stride."""
    if raw.dtype != torch.uint8 or raw.ndim != 1 or raw.stride(0) != 1:
        raise ValueError("raw PA_NZ backing must be a contiguous uint8 vector")
    if width <= 0 or width % PA_NZ_TILE or num_blocks <= 0:
        raise ValueError("invalid PA_NZ geometry")
    page_stride_bytes = block_size * width if page_stride_bytes is None else page_stride_bytes
    validate_pa_nz_page_stride(block_size, page_stride_bytes, width)
    required = num_blocks * page_stride_bytes
    if raw.numel() < required:
        raise ValueError("raw backing is smaller than PA_NZ payload")
    return torch.as_strided(
        raw[:required],
        size=(num_blocks, 1, width // PA_NZ_TILE, block_size, PA_NZ_TILE),
        stride=(page_stride_bytes, block_size * width, block_size * PA_NZ_TILE, PA_NZ_TILE, 1),
        storage_offset=raw.storage_offset(),
    )


def _kernel_blocks_per_manager(block_size: int) -> int:
    if block_size <= 0 or block_size % PA_NZ_BLOCK_SIZE:
        raise ValueError("C8 manager block size must be a positive multiple of 128")
    return block_size // PA_NZ_BLOCK_SIZE


def pa_nz_prolog_view(cache: torch.Tensor) -> torch.Tensor:
    """Expose the logical PA descriptor without changing physical NZ bytes."""
    if (
        cache.ndim != 5
        or cache.shape[1:3] != (1, K3_MLA_CACHE_WIDTH // PA_NZ_TILE)
        or cache.shape[4] != PA_NZ_TILE
        or cache.dtype not in (torch.uint8, torch.float8_e4m3fn)
        or not cache[0].is_contiguous()
    ):
        raise ValueError("C8 requires [blocks,1,18,block_tokens,32] PA_NZ cache")
    return cache.view(torch.float8_e4m3fn).view(cache.shape[0], cache.shape[3], 1, K3_MLA_CACHE_WIDTH)


def pa_nz_storage_view(cache: torch.Tensor) -> torch.Tensor:
    """Byte coverage of complete physical pages for zero/CoW, including padding."""
    return torch.as_strided(
        cache.view(torch.uint8),
        size=(cache.shape[0], cache.stride(0)),
        stride=(cache.stride(0), 1),
        storage_offset=cache.storage_offset(),
    )


def read_static_fp8_cache(
    cache: torch.Tensor,
    kernel_block_table: torch.Tensor,
    seq_lens: torch.Tensor | list[int],
    scale: torch.Tensor | float,
    *,
    starts: torch.Tensor | list[int] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Gather eager prefill history as request-ordered [sum(lengths),1,576].

    The table contains native physical PA_NZ block IDs. Each length is
    the requested chunk length; starts are absolute positions in each request.
    Only the requested rows are read, so unused negative/padded table entries
    are harmless. Physical NZ tiles are gathered before merging feature lanes.
    """
    pa_nz_prolog_view(cache)
    block_size = cache.shape[3]
    if kernel_block_table.ndim != 2 or kernel_block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("C8 history block table must be a rank-2 integer tensor")
    # Keep scheduler-owned host lengths/starts on CPU for validation and the
    # output size. Copying them to NPU before sum().item() stalls every layer.
    lengths = torch.as_tensor(seq_lens, dtype=torch.int64)
    offsets = torch.zeros_like(lengths) if starts is None else torch.as_tensor(starts, dtype=torch.int64)
    if lengths.ndim != 1 or offsets.shape != lengths.shape or lengths.numel() != kernel_block_table.shape[0]:
        raise ValueError("C8 history lengths/starts must match block table requests")
    if bool((lengths < 0).any()) or bool((offsets < 0).any()):
        raise ValueError("C8 history lengths/starts must be non-negative")
    host_bounds = lengths.device.type == offsets.device.type == "cpu"
    if host_bounds and bool(((lengths > 0) & (offsets + lengths > kernel_block_table.shape[1] * block_size)).any()):
        raise ValueError("C8 history block table does not cover requested positions")
    total = int(lengths.sum().item())
    if total == 0:
        return torch.empty((0, 1, K3_MLA_CACHE_WIDTH), dtype=dtype, device=cache.device)
    rows = _gather_static_fp8_rows(
        cache,
        kernel_block_table.to(cache.device),
        lengths.to(cache.device),
        offsets.to(cache.device),
        total,
        check_columns=not host_bounds,
        check_blocks=True,
    )
    return dequantize_static_fp8(rows, scale, dtype).unsqueeze(1)


def _gather_static_fp8_rows(
    cache: torch.Tensor,
    table: torch.Tensor,
    lengths: torch.Tensor,
    offsets: torch.Tensor,
    total: int,
    *,
    check_columns: bool = False,
    check_blocks: bool = False,
) -> torch.Tensor:
    block_size = cache.shape[3]
    requests = torch.repeat_interleave(torch.arange(lengths.numel(), device=cache.device), lengths, output_size=total)
    positions = torch.arange(total, device=cache.device) - (lengths.cumsum(0) - lengths)[requests] + offsets[requests]
    columns = positions // block_size
    if check_columns and bool((columns >= table.shape[1]).any()):
        raise ValueError("C8 history block table does not cover requested positions")
    block_ids = table[requests, columns].long()
    if check_blocks and bool(((block_ids < 0) | (block_ids >= cache.shape[0])).any()):
        raise ValueError("C8 history active block table entry is outside cache bounds")
    tiles = cache.view(torch.uint8)[block_ids, 0, :, positions % block_size, :]
    return tiles.reshape(total, K3_MLA_CACHE_WIDTH)


@dataclass(frozen=True)
class C8PrefillHistory:
    """One prefill step's validated page table and compact per-chunk read metadata."""

    block_table: torch.Tensor
    lengths: torch.Tensor
    starts: torch.Tensor
    totals: list[int]
    chunk_block_ids: tuple[torch.Tensor | None, ...] = ()

    @classmethod
    def prepare(
        cls, cache: torch.Tensor, block_table: torch.Tensor, lengths: torch.Tensor, starts: torch.Tensor
    ) -> C8PrefillHistory:
        pa_nz_prolog_view(cache)
        if block_table.ndim != 2 or block_table.dtype not in (torch.int32, torch.int64):
            raise ValueError("C8 history block table must be a rank-2 integer tensor")
        if lengths.ndim != 2 or starts.shape != lengths.shape or lengths.shape[1] != block_table.shape[0]:
            raise ValueError("C8 history lengths/starts must match block table requests")
        lengths = lengths.to(device="cpu", dtype=torch.int64)
        starts = starts.to(device="cpu", dtype=torch.int64)
        if bool(((lengths < 0) | (starts < 0)).any()):
            raise ValueError("C8 history lengths/starts must be non-negative")
        totals = lengths.sum(dim=1).tolist()
        block_table = block_table.to(cache.device)
        chunk_block_ids: list[torch.Tensor | None] = [None] * len(totals)
        if any(totals):
            block_size, columns = cache.shape[3], block_table.shape[1]
            active = lengths > 0
            ends = starts + lengths
            if bool((active & (ends > columns * block_size)).any()):
                raise ValueError("C8 history block table does not cover requested positions")
            # Mark the union of active page intervals without a chunk-by-page
            # tensor. Empty chunks and gaps may contain unmapped pages.
            first = torch.where(active, starts // block_size, 0)
            last = torch.where(active, (ends - 1) // block_size + 1, 0)
            changes = torch.zeros((block_table.shape[0], columns + 1), dtype=torch.int32)
            counts = active.T.to(torch.int32)
            changes.scatter_add_(1, first.T, counts)
            changes.scatter_add_(1, last.T, -counts)
            used = (changes[:, :-1].cumsum(dim=1) > 0).to(cache.device)
            invalid = (block_table < 0) | (block_table >= cache.shape[0])
            if bool((used & invalid).any()):
                raise ValueError("C8 history active block table entry is outside cache bounds")

            # Precompute block-level index_select lists for chunks whose active
            # segments start on block boundaries (and only the last segment may
            # end mid-block). Reusing these across all MLA layers avoids 5D
            # per-token fancy indexing on every layer.
            lengths_rows = lengths.tolist()
            starts_rows = starts.tolist()
            for chunk_idx, total in enumerate(totals):
                if total == 0:
                    continue
                active_reqs = [req_idx for req_idx, length in enumerate(lengths_rows[chunk_idx]) if length > 0]
                if not all(starts_rows[chunk_idx][req_idx] % block_size == 0 for req_idx in active_reqs):
                    continue
                if not all(lengths_rows[chunk_idx][req_idx] % block_size == 0 for req_idx in active_reqs[:-1]):
                    continue
                slices = [
                    block_table[
                        req_idx,
                        starts_rows[chunk_idx][req_idx] // block_size : (
                            starts_rows[chunk_idx][req_idx] + lengths_rows[chunk_idx][req_idx] + block_size - 1
                        )
                        // block_size,
                    ]
                    for req_idx in active_reqs
                ]
                chunk_block_ids[chunk_idx] = (
                    slices[0].to(torch.int64) if len(slices) == 1 else torch.cat(slices, dim=0).to(torch.int64)
                )
        return cls(block_table, lengths.to(cache.device), starts.to(cache.device), totals, tuple(chunk_block_ids))

    def read_raw(self, cache: torch.Tensor, chunk: int) -> torch.Tensor:
        """Gather raw uint8 FP8 history rows as [total, 1, 576] without dequantizing."""
        total = self.totals[chunk]
        if total == 0:
            return torch.empty((0, 1, K3_MLA_CACHE_WIDTH), dtype=torch.uint8, device=cache.device)
        blk_ids = self.chunk_block_ids[chunk] if chunk < len(self.chunk_block_ids) else None
        if blk_ids is not None:
            selected = cache.view(torch.uint8).index_select(0, blk_ids)
            rows = selected.squeeze(1).permute(0, 2, 1, 3).reshape(-1, K3_MLA_CACHE_WIDTH)
            if rows.shape[0] != total:
                rows = rows[:total]
        else:
            rows = _gather_static_fp8_rows(cache, self.block_table, self.lengths[chunk], self.starts[chunk], total)
        return rows.unsqueeze(1)

    def read(
        self,
        cache: torch.Tensor,
        chunk: int,
        validated_scale: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Read from a cache with the geometry validated by prepare."""
        raw = self.read_raw(cache, chunk)
        return _dequantize_static_fp8(raw, validated_scale, dtype)


def validate_pa_nz_page_stride(block_size: int, page_stride_bytes: int, width: int = K3_MLA_CACHE_WIDTH) -> None:
    """Allow only complete physical kernel pages between manager-page bases."""
    _kernel_blocks_per_manager(block_size)
    if (
        width <= 0
        or width % PA_NZ_TILE
        or page_stride_bytes < block_size * width
        or page_stride_bytes % (PA_NZ_BLOCK_SIZE * width)
    ):
        raise ValueError("C8 PA_NZ page padding must contain whole 128-token kernel pages")


def map_pa_nz_slots(slots: torch.Tensor, block_size: int, page_stride_bytes: int) -> torch.Tensor:
    """Native-page slots remain logical token indices; cache stride carries padding."""
    return slots


def map_pa_nz_kernel_block_table(kernel_ids: torch.Tensor, block_size: int, page_stride_bytes: int) -> torch.Tensor:
    """Coalesce V2's small table units into native FA page IDs."""
    ratio = _kernel_blocks_per_manager(block_size)
    return torch.div(kernel_ids[..., ::ratio], ratio, rounding_mode="floor").contiguous()
