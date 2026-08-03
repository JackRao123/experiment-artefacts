# LPS-1003 adjudicator: triple-exec majority vote + diff structure dump.
# Installed at runtime on every rank via harness7 pycode lever:
#   {"seq": N, "actions": [], "pycode": "exec(open('/root/.cache/user_artifacts/lps1003/rearm/adjudicator.py').read())"}
# Replaces harness6's double-exec wrapper on the indexer with a triple-exec:
# out1 is returned (faithful); disagreements are classified by majority
# (run1_outlier / run2_outlier / run3_outlier / all_differ) and the diff's
# tile structure (128-col / 128-row marginals) is logged. Idempotent.
import json as _json
import os as _os
import time as _time

import cudnn.deepseek_sparse_attention.indexer_forward.api as _api

if not getattr(_api, "_adjudicator_installed", False):
    _api._adjudicator_installed = True
    import torch as _torch

    _rank = _os.environ.get("RANK", "x")
    _dir = _os.environ.get("BT_DSA_DOUBLE_DIR", "/tmp/dsa_double")
    _os.makedirs(_dir, exist_ok=True)
    _adjf = open(f"{_dir}/adjudicate_rank{_rank}.jsonl", "a", buffering=1)

    # Recover the TRUE kernel from harness6's wrapper closure (its cell
    # holding a function named indexer_forward_wrapper), else use current.
    _cur = _api.indexer_forward_wrapper
    _true = None
    for _cell in (_cur.__closure__ or ()):
        try:
            _v = _cell.cell_contents
        except ValueError:
            continue
        if callable(_v) and getattr(_v, "__name__", "") == "indexer_forward_wrapper":
            _true = _v
            break
    if _true is None:
        _true = _cur  # no h6 wrapper found; wrap whatever is installed

    _state = {"n": 0}

    def _neq(a, b):
        m = a != b
        both_nan = _torch.isnan(a) & _torch.isnan(b)
        return m & ~both_nan

    def _adjudicated(*a, **k):
        out1 = _true(*a, **k)
        _state["n"] += 1
        try:
            out2 = _true(*a, **k)
            out3 = _true(*a, **k)
            s1 = out1["scores"] if isinstance(out1, dict) else out1
            s2 = out2["scores"] if isinstance(out2, dict) else out2
            s3 = out3["scores"] if isinstance(out3, dict) else out3
            with _torch.no_grad():
                d12 = _neq(s1, s2)
                d13 = _neq(s1, s3)
                d23 = _neq(s2, s3)
                n12 = int(d12.sum())
                n13 = int(d13.sum())
                n23 = int(d23.sum())
                if n12 or n13 or n23:
                    if n23 == 0:
                        verdict = "run1_outlier"      # model consumed the bad one
                        mask = d12
                    elif n13 == 0:
                        verdict = "run2_outlier"
                        mask = d12
                    elif n12 == 0:
                        verdict = "run3_outlier"
                        mask = d23
                    else:
                        verdict = "all_differ"
                        mask = d12 | d13 | d23
                    rows = mask.any(dim=-1)
                    cols = mask.any(dim=-2)
                    # 128-tile marginals, run-length compressed as index lists
                    rt = rows.view(-1)[: rows.numel() // 128 * 128].view(-1, 128).any(-1)
                    ct = cols.view(-1)[: cols.numel() // 128 * 128].view(-1, 128).any(-1)
                    rti = rt.nonzero().flatten().tolist()
                    cti = ct.nonzero().flatten().tolist()
                    # sample 3 differing coords with values
                    idx = mask.nonzero()
                    samp = []
                    for j in range(0, min(3, idx.shape[0])):
                        r_, c_ = idx[j * max(1, idx.shape[0] // 3)].tolist()
                        samp.append([r_, c_, float(s1[r_, c_]), float(s2[r_, c_]),
                                     float(s3[r_, c_])])
                    rec = {"ts": _time.time(), "call": _state["n"], "verdict": verdict,
                           "n12": n12, "n13": n13, "n23": n23,
                           "shape": list(s1.shape),
                           "row_tiles_bad": len(rti), "row_tiles_tot": int(rt.numel()),
                           "col_tiles_bad": len(cti), "col_tiles_tot": int(ct.numel()),
                           "row_tile_first_last": [rti[0], rti[-1]] if rti else None,
                           "col_tile_idx_head": cti[:40], "samples": samp}
                else:
                    rec = {"ts": _time.time(), "call": _state["n"], "verdict": "agree"}
            _adjf.write(_json.dumps(rec) + "\n")
            if rec["verdict"] not in ("agree",):
                print(f"[adjud r{_rank}] call={_state['n']} {rec['verdict']} "
                      f"n12={n12} n13={n13} n23={n23} rowtiles={rec['row_tiles_bad']}/"
                      f"{rec['row_tiles_tot']} coltiles={rec['col_tiles_bad']}/"
                      f"{rec['col_tiles_tot']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[adjud r{_rank}] failed: {exc}", flush=True)
        return out1

    _api.indexer_forward_wrapper = _adjudicated
    try:
        import cudnn as _cudnn

        _ns = getattr(_cudnn, "DSA", None)
        if _ns is not None:
            try:
                _ns.indexer_forward_wrapper = _adjudicated
            except Exception:
                pass
    except Exception:
        pass
    print(f"[adjud r{_rank}] installed (true kernel recovered: {_true is not _cur})",
          flush=True)
