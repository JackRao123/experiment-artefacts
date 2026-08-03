# LPS-1003 hot-swap: reload patched kernel modules (cop='cv' metadata loads)
# into the live trainer, then reinstall the v2.8 instrument over fresh raw.
import importlib as _il
import json as _json
import os as _os
import time as _time

_log = open(f"/root/lps1003_local/adjud2/hotswap_rank{_os.environ.get('RANK','x')}.jsonl", "a", buffering=1)
try:
    import cudnn.deepseek_sparse_attention.utils.seqlen as _seq
    _il.reload(_seq)
    _has_cv = hasattr(_seq, "ld_i32_cv")
    import cudnn.deepseek_sparse_attention.indexer_forward.indexer_fwd_sm100 as _km
    _il.reload(_km)
    import cudnn.deepseek_sparse_attention.indexer_forward._interface as _im
    _il.reload(_im)  # fresh empty _compile_cache -> recompile from patched source
    import cudnn.deepseek_sparse_attention.indexer_forward.api as _am
    for _attr in ("_v25", "_v26", "_v27", "_v28"):
        if hasattr(_am, _attr):
            delattr(_am, _attr)
    _il.reload(_am)
    exec(open("/root/lps1003_local/adjudicator28.py").read())
    _log.write(_json.dumps({"ts": _time.time(), "kind": "hotswap_ok",
                            "ld_i32_cv": _has_cv}) + "\n")
    print(f"[hotswap r{_os.environ.get('RANK','x')}] ok cv={_has_cv}", flush=True)
except Exception as _e:  # noqa: BLE001
    _log.write(_json.dumps({"ts": _time.time(), "kind": "hotswap_err", "err": repr(_e)}) + "\n")
    print(f"[hotswap r{_os.environ.get('RANK','x')}] FAILED {_e!r}", flush=True)
