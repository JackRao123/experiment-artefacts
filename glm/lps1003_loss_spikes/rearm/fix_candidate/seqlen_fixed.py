from typing import Optional
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr


@cute.jit
def ld_i32_cv(t, idx):
    """Coherent (L1-bypassing, cop='cv') scalar int32 load from a cute tensor.

    Metadata tensors (cu_seqlens, seqused, q_causal_offsets) live in small
    caching-allocator blocks that are recycled across kernels every layer.
    Plain ld.global can hit a stale per-SM cache line left by a PREVIOUS
    kernel's reads of the same VA (observed on sm103 with
    CUDA_DEVICE_MAX_CONNECTIONS unset: the kernel consumed a predecessor
    tensor's bytes while torch-level reads of the same address saw the fresh
    values — LPS-1003). cop='cv' fetches from the coherence point every time.
    """
    return Int32(cute.arch.load(t.iterator + idx, cutlass.Int32, cop="cv"))

"""
This consolidates all the info related to sequence length. This is so that we can do all
the gmem reads once at the beginning of each tile, rather than having to repeat these reads
to compute various things like n_block_min, n_block_max, etc.
"""


@cute.jit
def seqlen_info(
    mCuSeqlensQ,
    mCuSeqlensK,
    batch_idx: Int32,
    seqlen_q_static: Int32,
    seqlen_k_static: Int32,
):
    """Return batch-local offsets and lengths for BSHD or THD mode."""
    if const_expr(mCuSeqlensQ is None):
        return Int32(0), Int32(0), seqlen_q_static, seqlen_k_static
    else:
        q_offset = ld_i32_cv(mCuSeqlensQ, batch_idx)
        seqlen_q_b = ld_i32_cv(mCuSeqlensQ, batch_idx + Int32(1)) - q_offset
        k_offset = ld_i32_cv(mCuSeqlensK, batch_idx)
        seqlen_k_b = ld_i32_cv(mCuSeqlensK, batch_idx + Int32(1)) - k_offset
        return q_offset, k_offset, seqlen_q_b, seqlen_k_b


@dataclass(frozen=True)
class SeqlenInfoQK:
    offset_q: cutlass.Int32
    offset_k: cutlass.Int32
    padded_offset_q: cutlass.Int32
    padded_offset_k: cutlass.Int32
    seqlen_q: cutlass.Int32
    seqlen_k: cutlass.Int32
    has_cu_seqlens_q: cutlass.Constexpr[bool]
    has_cu_seqlens_k: cutlass.Constexpr[bool]
    has_seqused_q: cutlass.Constexpr[bool]
    has_seqused_k: cutlass.Constexpr[bool]

    @staticmethod
    def create(
        batch_idx: cutlass.Int32,
        seqlen_q_static: cutlass.Int32,
        seqlen_k_static: cutlass.Int32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        tile_m: cutlass.Constexpr[cutlass.Int32] = 128,
        tile_n: cutlass.Constexpr[cutlass.Int32] = 128,
    ):
        offset_q = 0 if const_expr(mCuSeqlensQ is None) else ld_i32_cv(mCuSeqlensQ, batch_idx)
        offset_k = 0 if const_expr(mCuSeqlensK is None) else ld_i32_cv(mCuSeqlensK, batch_idx)
        padded_offset_q = 0 if const_expr(mCuSeqlensQ is None) else (offset_q + batch_idx * tile_m) // tile_m * tile_m
        padded_offset_k = 0 if const_expr(mCuSeqlensK is None) else (offset_k + batch_idx * tile_n) // tile_n * tile_n
        if const_expr(mSeqUsedQ is not None):
            seqlen_q = ld_i32_cv(mSeqUsedQ, batch_idx)
        else:
            seqlen_q = seqlen_q_static if const_expr(mCuSeqlensQ is None) else ld_i32_cv(mCuSeqlensQ, batch_idx + 1) - offset_q
        if const_expr(mSeqUsedK is not None):
            seqlen_k = ld_i32_cv(mSeqUsedK, batch_idx)
        else:
            seqlen_k = seqlen_k_static if const_expr(mCuSeqlensK is None) else ld_i32_cv(mCuSeqlensK, batch_idx + 1) - offset_k
        has_cu_seqlens_q: int = mCuSeqlensQ is not None
        has_cu_seqlens_k: int = mCuSeqlensK is not None
        has_seqused_q: int = mSeqUsedQ is not None
        has_seqused_k: int = mSeqUsedK is not None
        return SeqlenInfoQK(
            offset_q,
            offset_k,
            padded_offset_q,
            padded_offset_k,
            seqlen_q,
            seqlen_k,
            has_cu_seqlens_q,
            has_cu_seqlens_k,
            has_seqused_q,
            has_seqused_k,
        )

    def offset_batch_Q(
        self,
        mQ: cute.Tensor,
        batch_idx: Int32,
        dim: int,
        padded: cutlass.Constexpr[bool] = False,
    ) -> cute.Tensor:
        """Seqlen must be the first dimension of mQ"""
        if const_expr(not self.has_cu_seqlens_q):
            idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mQ) - 1 - dim)
            return mQ[idx]
        else:
            offset_q = self.offset_q if const_expr(not padded) else self.padded_offset_q
            offset = offset_q if const_expr(cute.rank(mQ.shape[0]) == 1) else (0, offset_q)
            idx = (offset,) + (0,) * (cute.rank(mQ) - 1)
            return cute.domain_offset(idx, mQ)

    def offset_batch_K(
        self,
        mK: cute.Tensor,
        batch_idx: Int32,
        dim: int,
        padded: cutlass.Constexpr[bool] = False,
    ) -> cute.Tensor:
        """Seqlen must be the first dimension of mK"""
        if const_expr(not self.has_cu_seqlens_k):
            idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mK) - 1 - dim)
            return mK[idx]
        else:
            offset_k = self.offset_k if const_expr(not padded) else self.padded_offset_k
            idx = (offset_k,) + (0,) * (cute.rank(mK) - 1)
            return cute.domain_offset(idx, mK)
