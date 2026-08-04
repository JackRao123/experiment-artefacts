"""Regression test for the cuDNN-frontend DSA indexer forward launch stream.

LPS-1003: the CuTe-DSL indexer forward kernel and its ``-inf`` output
prefill used to run in an arrangement where, with
``CUDA_DEVICE_MAX_CONNECTIONS`` unset (the production default), the prefill
could lose ordering against the compiled kernel launch and land AFTER the
kernel's stores — erasing freshly-written score rows. Reads then saw the
prefill's ``-inf`` for whole trailing halves of packed THD rows, top-k
selected garbage KV, and the first forward after any cold start (boot,
in-process rebuild, ``torch.cuda.empty_cache()``) destroyed the NLL of
every document in the affected rows (5-11 nats), healing on the next
forward. Verified A/B on GLM-5.2-FP8 cp16 262k on B300: unfixed wheel
corrupts 10/10 boot-window and 14/14 empty_cache-treated forwards; the
fixed wheel runs 0/10 treated + 150-rep soak clean with 0/2896 bitwise
self-disagreements.

Root cause: the wheel's ``torch_stream_context`` built
``torch.cuda.ExternalStream(0)``, which on every torch release <= 2.12
returns a rotating POOL stream rather than the legacy default stream, so
the prefill was never ordered against the DSL kernel launch. Our interim
fix (1.26.0+dsatopk5, retired) pinned indexer_fwd's prefill+launch to a
dedicated event-chained stream; the vendored wheel now carries upstream's
own fix (NVIDIA/cudnn-frontend PR #354, in develop @ 74785165), which
switches ``torch_stream_context`` to
``torch.cuda.get_stream_from_external`` and covers every DSA call site.
A/B on this wheel (B300, 12 reps x 5 geometries): pristine 1.26.0 fires
12/12 at 4 GiB out with the erasure signature; the pinned-develop wheel
0/12 everywhere + 100-rep identical-forward soak clean on both kernel and
module level.

Three tests:

* the *arrangement* assertion (CPU-only, runs everywhere the wheel is
  installed): the vendored wheel's stream glue must never construct
  ``torch.cuda.ExternalStream`` (for handle 0, every torch release <= 2.12
  returns a rotating POOL stream, never the legacy default stream — the
  LPS-1003 root cause); ``torch_stream_context`` must route through
  ``torch.cuda.get_stream_from_external``, which is correct for all handles
  on all torch versions (upstream fix: NVIDIA/cudnn-frontend PR #354);
* THD varlen output matches a pure-torch reference (guards against the
  stream arrangement breaking semantics); and
* a standalone single-GPU repro of the race itself: the race only
  manifests when the ``-inf`` prefill exceeds 2**31 BYTES, because torch
  splits elementwise kernels into two launches at that size and it is the
  *second* fill launch that loses ordering against the DSL kernel's stores
  (which is also why production erasures looked like "trailing halves" of
  packed rows). Verified on a B300 devbox: at out = 2.25 GiB the unfixed
  wheel bitwise self-disagrees on every double-exec rep (6/6, 8/8 at
  4 GiB) with the pure erasure signature (finite-mask flips, zero numeric
  diff on both-finite positions), while sub-threshold geometry passes
  0/8 even unfixed. The fixed wheel runs it clean.

The vendored wheel only ships for linux/x86_64 cp312 (the pyproject marker)
and CI installs it via ``uv sync --extra worker`` (dev_job/bootstrap_ci.sh),
so on those platforms a missing module is a hard FAILURE — the regression
guard must never silently skip in CI. Other platforms (e.g. darwin dev
machines) skip.
"""

from __future__ import annotations

import importlib
import inspect
import platform
import sys
from types import ModuleType

import pytest
import torch

# Mirrors the pyproject vendored-wheel marker.
_WHEEL_REQUIRED = (
    sys.platform == "linux"
    and platform.machine() == "x86_64"
    and sys.version_info[:2] == (3, 12)
)

try:
    _cudnn_interface: ModuleType | None = importlib.import_module(
        "cudnn.deepseek_sparse_attention.indexer_forward._interface"
    )
except ImportError:
    _cudnn_interface = None


def _require_wheel() -> ModuleType:
    if _cudnn_interface is not None:
        return _cudnn_interface
    if _WHEEL_REQUIRED:
        pytest.fail(
            "vendored nvidia-cudnn-frontend wheel is missing on linux/x86_64 "
            "cp312 — the LPS-1003 launch-stream regression guard cannot run. "
            "CI must install it (uv sync --extra worker); do not let this "
            "test skip silently."
        )
    pytest.skip("vendored nvidia-cudnn-frontend only ships for linux/x86_64 cp312")


def test_stream_glue_never_constructs_external_stream() -> None:
    """The whole DSA python tree must be free of torch.cuda.ExternalStream:
    a handle-0 ExternalStream is a pool stream on every torch we ship, so any
    call site that survives is a latent LPS-1003. torch_stream_context must
    use get_stream_from_external instead (PR #354)."""
    iface = _require_wheel()
    import pathlib

    runtime = importlib.import_module("cudnn.deepseek_sparse_attention.utils.runtime")
    assert "get_stream_from_external" in inspect.getsource(runtime.torch_stream_context)

    dsa_root = pathlib.Path(inspect.getfile(iface)).parents[1]
    offenders = [
        str(p.relative_to(dsa_root))
        for p in sorted(dsa_root.rglob("*.py"))
        if "ExternalStream(" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        "torch.cuda.ExternalStream constructed in DSA glue (pool-stream "
        f"footgun, LPS-1003): {offenders}"
    )


def _thd_case() -> dict:
    """Deterministic THD varlen inputs; rebuilt fresh per call so reruns
    reallocate (allocator churn is part of the cold-start fingerprint)."""
    torch.manual_seed(1234)
    device = "cuda"
    n_heads, head_dim = 32, 128
    total_q, total_k = 512, 1024
    return {
        "q": torch.randn(
            total_q, n_heads, head_dim, dtype=torch.bfloat16, device=device
        ),
        "k": torch.randn(total_k, 1, head_dim, dtype=torch.bfloat16, device=device),
        "w": torch.randn(total_q, n_heads, dtype=torch.bfloat16, device=device),
        "cu_q": torch.tensor([0, 256, 512], dtype=torch.int32, device=device),
        "cu_k": torch.tensor([0, 512, 1024], dtype=torch.int32, device=device),
        "offsets": [100, 300],
        "max_q": 256,
        "max_k": 512,
    }


def _run_indexer(iface: ModuleType, case: dict) -> torch.Tensor:
    out = iface.indexer_fwd(
        case["q"],
        case["k"],
        case["w"],
        ratio=1,
        cu_seqlens_q=case["cu_q"],
        cu_seqlens_k=case["cu_k"],
        max_seqlen_q=case["max_q"],
        max_seqlen_k=case["max_k"],
        q_causal_offsets=torch.tensor(
            case["offsets"], dtype=torch.int32, device=case["q"].device
        ),
    )
    torch.cuda.synchronize()
    return out


def _skip_unless_sm100() -> None:
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("DSA CuTe indexer forward requires sm100+")


@pytest.mark.gpu
def test_indexer_fwd_thd_matches_reference() -> None:
    iface = _require_wheel()
    _skip_unless_sm100()
    case = _thd_case()
    out = _run_indexer(iface, case)
    q, k, w = case["q"], case["k"], case["w"]
    cu_q, cu_k, offsets = case["cu_q"], case["cu_k"], case["offsets"]
    total_q = q.shape[0]

    # torch reference: S[t, j] = sum_h relu(q[t,h] @ k[j]) * w[t,h], with the
    # ratio-causal mask kv >= (offset + local_t + 1) // ratio -> -inf, laid
    # out in local-K columns per THD segment.
    device = q.device
    ref = torch.full((total_q, 512), float("-inf"), dtype=torch.float32, device=device)
    for b in range(2):
        q0, q1 = int(cu_q[b]), int(cu_q[b + 1])
        k0, k1 = int(cu_k[b]), int(cu_k[b + 1])
        qs = q[q0:q1].float()
        ks = k[k0:k1, 0].float()
        ws = w[q0:q1].float()
        scores = torch.einsum("thd,jd->thj", qs, ks).relu()
        s = torch.einsum("thj,th->tj", scores, ws)
        loc = torch.arange(q1 - q0, device=device)
        lim = offsets[b] + loc + 1  # ratio=1
        col = torch.arange(k1 - k0, device=device)
        mask = col.unsqueeze(0) < lim.unsqueeze(1).clamp(max=k1 - k0)
        ref[q0:q1, : k1 - k0] = torch.where(mask, s, torch.full_like(s, float("-inf")))

    fin_out = torch.isfinite(out)
    fin_ref = torch.isfinite(ref)
    assert bool((fin_out == fin_ref).all()), "causal write region mismatch"
    assert torch.allclose(out[fin_out], ref[fin_ref], atol=2e-2, rtol=2e-2), (
        "indexer scores diverge from torch reference"
    )


@pytest.mark.gpu
def test_indexer_fwd_bitwise_stable_across_cold_starts() -> None:
    """Standalone GPU repro of the LPS-1003 race (see module docstring for
    the mechanism and A/B results). Uses the 2**31-byte geometry that
    triggers it; empty_cache between reps re-arms the allocator (the
    production trigger), double-exec catches the race within a rep, and
    the baseline compare catches it across reps. ~7 GiB GPU memory."""
    iface = _require_wheel()
    _skip_unless_sm100()
    torch.manual_seed(1234)
    device = "cuda"
    n_heads, head_dim = 32, 128
    total_q, seg_k, n_segs = 8192, 73728, 2  # out = 8192*2*73728*4 B = 2.25 GiB
    seg_q = total_q // n_segs
    total_k = seg_k * n_segs
    case = {
        "q": torch.randn(
            total_q, n_heads, head_dim, dtype=torch.bfloat16, device=device
        ),
        "k": torch.randn(total_k, 1, head_dim, dtype=torch.bfloat16, device=device),
        "w": torch.randn(total_q, n_heads, dtype=torch.bfloat16, device=device),
        "cu_q": torch.arange(0, total_q + 1, seg_q, dtype=torch.int32, device=device),
        "cu_k": torch.arange(0, total_k + 1, seg_k, dtype=torch.int32, device=device),
        "offsets": [seg_k - seg_q] * n_segs,
        "max_q": seg_q,
        "max_k": seg_k,
    }
    baseline = None
    for rep in range(3):
        torch.cuda.empty_cache()
        first = _run_indexer(iface, case)
        second = _run_indexer(iface, case)
        assert torch.equal(first, second), (
            f"rep {rep}: double-exec bitwise self-disagreement — prefill/launch "
            "race erasing kernel stores (LPS-1003)"
        )
        if baseline is None:
            baseline = first
        else:
            assert torch.equal(first, baseline), (
                f"rep {rep} bitwise-diverges from rep 0 after "
                "torch.cuda.empty_cache() — cold-start erasure (LPS-1003)"
            )
        del first, second
