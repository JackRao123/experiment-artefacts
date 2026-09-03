"""Pod-only diagnostic: log why FusedIndexerSparseAttnFunc takes the compaction backward.

Wraps FusedIndexerSparseAttnFunc.forward and, once per process, prints the five
inputs that decide ctx.all_sparse_bwd_rows_nonempty. Installed by appending
`import debug_dsa_bwd_flags; debug_dsa_bwd_flags.install()` in backend.py on the
pod checkout (never committed).
"""
import inspect
import os

_done = False


def install():
    from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels as k

    fn = k.FusedIndexerSparseAttnFunc.forward
    params = list(inspect.signature(fn).parameters)

    def wrapped(*args, **kwargs):
        global _done
        if not _done:
            _done = True
            bound = dict(zip(params, args))
            bound.update(kwargs)
            pick = {n: bound.get(n) for n in ("varlen_starts", "varlen_ends", "key_positions", "query_valid_rows", "use_local_indexer_varlen", "single_packed_thd_sequence")}
            desc = {n: (None if v is None else (v if isinstance(v, (bool, int)) else f"Tensor{tuple(v.shape)}")) for n, v in pick.items()}
            out = f"/root/.cache/user_artifacts/lps1062_bench/glm_nsys_gpu_metrics_262k_20260902/ab/dsa_bwd_flags.rank{os.environ.get('RANK')}.txt"
            with open(out, "a") as fh:
                fh.write(f"sq={bound.get('query').shape if bound.get('query') is not None else None} {desc}\n")
        return fn(*args, **kwargs)

    k.FusedIndexerSparseAttnFunc.forward = staticmethod(wrapped)
