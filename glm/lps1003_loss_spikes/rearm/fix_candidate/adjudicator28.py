# LPS-1003 v2.5: clean-chain stale-metadata capture + TVM-FFI stream-pin A/B.
# - removes harness6's MetaPathFinder (it re-wraps the api module on every
#   reload — that is what killed the v2.4 install), then reloads api to the
#   pristine raw wrapper and installs exactly ONE instrument.
# - per call: mode-file toggles {sync_before, pin_stream} -> pre-clone
#   cu_seqlens_q -> exec1 (optionally inside tvm_ffi.use_torch_stream()) ->
#   post-clone -> mapping deltas -> exec2 -> bitwise compare.
# - one-time stream identity log: torch current stream vs tvm-ffi env stream.
# No bare asserts; every failure is written to the node-local jsonl.
import json as _json
import os as _os
import sys as _sys
import time as _time
import importlib as _importlib

_V = "2.8"
_rank = _os.environ.get("RANK", "x")
_dir = "/root/lps1003_local/adjud2"
_os.makedirs(_dir, exist_ok=True)
_f = open(f"{_dir}/v28_rank{_rank}.jsonl", "a", buffering=1)

try:
    import cudnn.deepseek_sparse_attention.indexer_forward.api as _api

    if getattr(_api, "_v28", None) != _V:
        # 1. remove harness6 meta-path finder(s)
        _removed = 0
        for _mp in list(_sys.meta_path):
            _m = getattr(type(_mp), "__module__", "")
            if "sitecustomize" in _m or "harness" in _m.lower():
                _sys.meta_path.remove(_mp)
                _removed += 1
        # 2. reload to pristine
        _importlib.reload(_api)
        _api._v28 = _V
        import torch as _torch
        import cudnn as _cudnn

        _true = _api.indexer_forward_wrapper
        _ok = (getattr(_true, "__name__", "") == "indexer_forward_wrapper"
               and _true.__closure__ is None)
        _f.write(_json.dumps({"ts": _time.time(), "kind": "install",
                              "finders_removed": _removed, "raw_ok": _ok,
                              "true_name": getattr(_true, "__name__", "?")}) + "\n")
        if not _ok:
            raise RuntimeError(f"raw not recovered: {_true!r}")

        try:
            import tvm_ffi as _tvm_ffi
        except Exception:
            _tvm_ffi = None

        _MODEF = "/root/lps1003_local/v24_mode.json"
        _state = {"n": 0, "streamlogged": False}

        def _mode():
            try:
                return _json.load(open(_MODEF))
            except Exception:
                return {}

        def _stats():
            try:
                s = _torch.cuda.memory_stats()
                return {"map": s.get("num_device_alloc", -1),
                        "unmap": s.get("num_device_free", -1),
                        "seg": s.get("segment.all.current", -1)}
            except Exception:
                return {}

        def _streamlog():
            rec = {"ts": _time.time(), "kind": "streams"}
            try:
                rec["torch_stream"] = _torch.cuda.current_stream().cuda_stream
                rec["torch_device"] = _torch.cuda.current_device()
            except Exception as e:  # noqa: BLE001
                rec["torch_err"] = repr(e)
            try:
                from tvm_ffi import core as _core
                rec["env_stream"] = int(_core._env_get_current_stream(2, _torch.cuda.current_device()))
            except Exception as e:  # noqa: BLE001
                rec["env_err"] = repr(e)
            _f.write(_json.dumps(rec) + "\n")

        def _w(*a, **k):
            _state["n"] += 1
            n = _state["n"]
            rec = {"ts": _time.time(), "call": n}
            out1 = None
            try:
                if not _state["streamlogged"]:
                    _state["streamlogged"] = True
                    _streamlog()
                cu = k.get("cu_seqlens_q")
                try:
                    rec["ptrs"] = {
                        "cu_q": cu.data_ptr() if cu is not None else None,
                        "cu_k": k["cu_seqlens_k"].data_ptr() if k.get("cu_seqlens_k") is not None else None,
                        "off": k["q_causal_offsets"].data_ptr() if k.get("q_causal_offsets") is not None else None,
                        "q": a[0].data_ptr() if len(a) else None,
                    }
                except Exception:
                    pass
                md = _mode()
                if md.get("poison_call") == n:
                    # POISON TEST: does the compiled kernel honor the per-call
                    # cu pointer, or a stale address? Pass a FRESH clone while
                    # the ORIGINAL tensor memory holds a poison pattern.
                    try:
                        truth_dev = cu.clone()
                        out_ref = _true(*a, **k)   # normal exec for reference
                        fresh = cu.clone()
                        cu.copy_(cu.clamp(max=int(md.get("poison_clamp", 4000))))
                        k2 = dict(k)
                        k2["cu_seqlens_q"] = fresh
                        out_p = _true(*a, **k2)
                        cu.copy_(truth_dev)        # restore before anyone else reads
                        r1 = out_ref["scores"] if not _torch.is_tensor(out_ref) else out_ref
                        r2 = out_p["scores"] if not _torch.is_tensor(out_p) else out_p
                        pm = (r1 != r2) & ~(_torch.isnan(r1) & _torch.isnan(r2))
                        pd = int(pm.sum())
                        prec = {"ts": _time.time(), "kind": "poison", "call": n,
                                "poison_diff": pd}
                        if pd:
                            prows = pm.any(dim=-1)
                            pidx = prows.nonzero().flatten()
                            prec["poison_bad_rows"] = [int(pidx[0]), int(pidx[-1]), int(prows.sum())]
                            prec["ninf_in_poisoned"] = int(_torch.isinf(r2[pm]).sum())
                        del pm
                        _f.write(_json.dumps(prec) + "\n")
                        print(f"[v28 r{_rank}] POISON call={n} diff={pd} {prec.get('poison_bad_rows')}", flush=True)
                        return out_ref
                    except Exception as pe:  # noqa: BLE001
                        try:
                            cu.copy_(truth_dev)
                        except Exception:
                            pass
                        _f.write(_json.dumps({"kind": "poison_err", "call": n, "err": repr(pe)}) + "\n")
                if md.get("clone_args"):
                    # A/B: hand the kernel FRESH tensors (different allocator
                    # block => different address history) with true content.
                    k = dict(k)
                    for _nm in ("cu_seqlens_q", "cu_seqlens_k", "q_causal_offsets"):
                        if k.get(_nm) is not None:
                            k[_nm] = k[_nm].clone()
                    cu = k.get("cu_seqlens_q")
                    rec["cloned"] = True
                if md.get("sync_before"):
                    _torch.cuda.current_stream().synchronize()
                    rec["synced"] = True
                s0 = _stats()
                pre = cu.clone() if cu is not None else None
                if md.get("pin_stream") and _tvm_ffi is not None:
                    rec["pinned"] = True
                    with _tvm_ffi.use_torch_stream():
                        out1 = _true(*a, **k)
                else:
                    out1 = _true(*a, **k)
                post = cu.clone() if cu is not None else None
                s1 = _stats()
                rec["map_run1"] = {kk: s1.get(kk, 0) - s0.get(kk, 0) for kk in s0}
                if md.get("compare"):
                    out2 = _true(*a, **k)
                    s2 = _stats()
                    o1 = out1["scores"] if not _torch.is_tensor(out1) else out1
                    o2 = out2["scores"] if not _torch.is_tensor(out2) else out2
                    with _torch.no_grad():
                        m = (o1 != o2) & ~(_torch.isnan(o1) & _torch.isnan(o2))
                        d12 = int(m.sum())
                        rec["d12"] = d12
                        if d12:
                            rows = m.any(dim=-1)
                            ridx = rows.nonzero().flatten()
                            rec["bad_rows"] = [int(ridx[0]), int(ridx[-1]), int(rows.sum())]
                            rec["ninf_o1"] = int(_torch.isinf(o1[m]).sum())
                        del m
                    rec["map_run2"] = {kk: s2.get(kk, 0) - s1.get(kk, 0) for kk in s1}
                else:
                    # single-exec fidelity: detect corruption from out1 alone —
                    # count -inf rows inside the causal-valid region of seg tail
                    o1 = out1["scores"] if not _torch.is_tensor(out1) else out1
                    with _torch.no_grad():
                        col0_inf = _torch.isinf(o1[:, 0])
                        rec["ninf_col0"] = int(col0_inf.sum())
                        if rec["ninf_col0"]:
                            ridx = col0_inf.nonzero().flatten()
                            rec["inf_rows"] = [int(ridx[0]), int(ridx[-1])]
                if cu is not None:
                    truth = cu.detach().cpu()
                    pre_h, post_h = pre.cpu(), post.cpu()
                    rec["pre_eq_true"] = bool((pre_h == truth).all())
                    rec["post_eq_true"] = bool((post_h == truth).all())
                    if (not rec["pre_eq_true"]) or (not rec["post_eq_true"]) or rec.get("d12"):
                        rec["cu_true"] = truth.tolist()
                        rec["cu_pre"] = pre_h.tolist()
                        rec["cu_post"] = post_h.tolist()
            except Exception as e:  # noqa: BLE001
                rec["err"] = repr(e)
                if out1 is None:
                    out1 = _true(*a, **k)
            try:
                _f.write(_json.dumps(rec) + "\n")
            except Exception:
                pass
            return out1

        _api.indexer_forward_wrapper = _w
        _ns = getattr(_cudnn, "DSA", None)
        if _ns is not None:
            try:
                _ns.indexer_forward_wrapper = _w
            except Exception:
                pass
        print(f"[v28 r{_rank}] installed (finders_removed={_removed})", flush=True)
except Exception as _e:  # noqa: BLE001
    try:
        _f.write(_json.dumps({"ts": _time.time(), "kind": "install_error",
                              "err": repr(_e)}) + "\n")
    except Exception:
        pass
    print(f"[v28 r{_rank}] INSTALL FAILED: {_e!r}", flush=True)
