"""LPS-1003 Issue-2 white-box harness: DSA top-k OUTPUT-INDICES under-write hunt.

Deploy as <dir>/sitecustomize.py and prepend <dir> to PYTHONPATH of the trainer
ranks (start_trainer.sh's srun uses --export=ALL). Inert unless BT_DSA_AUDIT=1.

Targets the mechanism in experiment_handoff.md (rewritten 2026-07-30 evening):
the wheel allocates `output_indices_torch = torch.empty(num_rows, top_k,
int32)` (indexer_top_k_decode_varlen.py:684) and never pre-fills it, despite
its own api.py:51-54 promising an "initial (-1) state"; the long-row radix
branch may leave rows/slots unwritten, so recycled allocator garbage shows
through as valid-looking indices. The scores-read-past hypothesis is dead
(code-refuted in the wheel; independently confirmed at runtime by this
harness's boot-1 run: invalid region 100% -inf over 1.6e12 elements).

Experiments:
  E0  branch/shape audit    which score branch runs; are launches prod-shaped
                            (num_rows > 148 -> large_occupancy variant, the only
                            variant that compiles the gmem spill path)
  E1  output-index audit    per call: indices >= row window, duplicates within a
                            row (the expected allocator-garbage signature:
                            previous-layer echoes), negative sentinel counts
  E2  staged-block poison   THE detector. Fill an (num_rows, top_k) int32 block
                            with a sentinel and free it immediately before the
                            wheel's torch.empty of exactly that shape, so the
                            caching allocator hands the kernel a known-poisoned
                            output buffer. Any slot still holding the sentinel
                            afterwards is an UNWRITTEN slot — a caught
                            under-write. Replaces the whole-GPU poison
                            (poison_gpus.py) with a deterministic, targeted one.
                            Includes a mandatory positive control per shape: if
                            the allocator does not actually hand the staged
                            block back, results are inconclusive and say so.
  E4  fix arm               BT_DSA_FIX=1 rewrites sentinel slots to -1 after the
                            call — behaviourally the proposed fill_(-1) glue
                            hardening. Run STAGE alone vs STAGE+FIX: destruction
                            then clean is the complete proof pair.

Env:
  BT_DSA_AUDIT=1              enable (required for everything below)
  BT_DSA_STAGE=1              E2 staged-block poison + unwritten-slot detection
  BT_DSA_STAGE_VAL=200003     sentinel; must exceed any sk so it is unambiguous,
                              and stay < 262144 so it also mimics the in-range
                              garbage the whole-GPU poison would have produced
  BT_DSA_FIX=1                E4 fix arm (implies STAGE)
  BT_DSA_DUP_EVERY=10         duplicate scan every Nth call (0=off; adds a sort)
  BT_DSA_SCORES_EVERY=50      sampled invalid-region content check (regression
                              watch on the now-refuted scores path; 0=off)
  BT_DSA_AUDIT_DIR=/path      per-rank JSONL, one record per call
  BT_DSA_HB=25                stderr heartbeat every N calls
  BT_DSA_ECHO_CALLS=40        always print the first N calls

Anomalies (unwritten slots, duplicates, out-of-range selections, failed staging
control) ALWAYS print regardless of sampling. Adds GPU->CPU syncs on every
call: diagnostic boots only, and the batch-0 baseline gate (0.760-0.771 band)
must pass with the harness on before any result here is trusted.

Seam: megatron...dsa_cudnn_kernels._indexer_top_k_one_chunk — every selection
path funnels here (call sites 683/831/1300 via _indexer_top_k_wrapper_chunked;
verified on trainer-cuda13-sm103 @ 0e0b65a). Never modify vendored 3rdparty/.
"""

from __future__ import annotations

import atexit
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
import time

_TARGET = "megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels"
_AUDIT = os.environ.get("BT_DSA_AUDIT") == "1"
_FIX = os.environ.get("BT_DSA_FIX") == "1"
_STAGE = os.environ.get("BT_DSA_STAGE") == "1" or _FIX
_DIR = os.environ.get("BT_DSA_AUDIT_DIR")

if _AUDIT:

    def _install(mod) -> None:
        import torch

        rank = os.environ.get("RANK", "?")
        hb = int(os.environ.get("BT_DSA_HB", "25"))
        echo_n = int(os.environ.get("BT_DSA_ECHO_CALLS", "40"))
        dup_every = int(os.environ.get("BT_DSA_DUP_EVERY", "10"))
        scores_every = int(os.environ.get("BT_DSA_SCORES_EVERY", "50"))
        SENT = int(os.environ.get("BT_DSA_STAGE_VAL", "200003"))

        st = {
            "topk": 0,
            "torch_scores": 0,
            "cudnn_fw": 0,
            "tfsc": 0,
            # E2/E4
            "staged_calls": 0,
            "unwritten_slots": 0,
            "unwritten_rows": 0,
            "unwritten_calls": 0,
            "fixed_slots": 0,
            "ctrl_shapes_ok": 0,
            "ctrl_shapes_fail": 0,
            # E1
            "dup_scanned": 0,
            "dup_slots": 0,
            "dup_rows": 0,
            "dup_calls": 0,
            "sel_oor_total": 0,
            "sel_oor_calls": 0,
            "short_rows": 0,
            "radix_rows": 0,
            # regression watch
            "scores_scanned": 0,
            "inv_finite_calls": 0,
            "sk_ge_sentinel_calls": 0,
        }
        ctrl: dict[tuple, float] = {}

        jf = None
        if _DIR:
            try:
                os.makedirs(_DIR, exist_ok=True)
                jf = open(os.path.join(_DIR, f"rank{rank}_pid{os.getpid()}.jsonl"), "a", buffering=1)
            except OSError as e:
                print(f"[dsa_audit r{rank}] cannot open audit dir {_DIR}: {e}", file=sys.stderr, flush=True)

        def say(msg: str) -> None:
            print(f"[dsa_audit r{rank}] {msg}", file=sys.stderr, flush=True)

        def emit(rec: dict) -> None:
            if jf is None:
                return
            try:
                rec["ts"] = round(time.time(), 3)
                jf.write(json.dumps(rec) + "\n")
            except Exception as e:  # noqa: BLE001 — logging must never break training
                print(f"[dsa_audit r{rank}] emit failed: {e}", file=sys.stderr, flush=True)

        # ------------------------------------------------------------------ #
        # E0: score-branch counters
        # ------------------------------------------------------------------ #
        orig_torch_scores = mod._compute_indexer_scores_chunk_with_global_rows

        def _torch_scores(*a, **kw):
            st["torch_scores"] += 1
            c = st["torch_scores"]
            if c == 1 or c % 500 == 0:
                say(f"score branch TORCH global_rows call#{c} (this branch -inf-masks past seq_lens)")
            return orig_torch_scores(*a, **kw)

        mod._compute_indexer_scores_chunk_with_global_rows = _torch_scores

        # mod._cudnn_dsa is None at import (line ~127); it is only bound when
        # _ensure_dsa_namespace() first runs, so wrap from inside that call.
        orig_ensure = mod._ensure_dsa_namespace

        def _ensure():
            orig_ensure()
            ns = getattr(mod, "_cudnn_dsa", None)
            if ns is None or getattr(ns, "_bt_fw_wrapped", False):
                return
            orig_cud_fw = ns.indexer_forward_wrapper

            def _cud_fw(*a, **kw):
                st["cudnn_fw"] += 1
                c = st["cudnn_fw"]
                if c == 1 or c % 500 == 0:
                    varlen = "varlen" if kw.get("cu_seqlens_q") is not None else "dense"
                    offs = "offsets" if kw.get("q_causal_offsets") is not None else "no-offsets"
                    say(f"score branch CUDNN indexer_forward call#{c} ({varlen},{offs})")
                return orig_cud_fw(*a, **kw)

            try:
                ns.indexer_forward_wrapper = _cud_fw
                ns._bt_fw_wrapped = True
                say("cuDNN branch counter attached")
            except Exception as e:  # noqa: BLE001 — read-only ns: counter off, audits unaffected
                say(f"WARN cannot wrap indexer_forward_wrapper ({e}); cuDNN counter off")

        mod._ensure_dsa_namespace = _ensure

        orig_tfsc = mod._indexer_topk_from_score_chunks

        def _tfsc(q_bshd, k_bshd, w_bsh, seq_lens, topk_k, return_topk_scores, **kw):
            st["tfsc"] += 1
            c = st["tfsc"]
            rec = {
                "kind": "tfsc", "call": c, "q": tuple(q_bshd.shape), "sk": int(k_bshd.size(1)),
                "k": int(topk_k), "ret_scores": bool(return_topk_scores),
                "starts": kw.get("starts") is not None, "ends": kw.get("ends") is not None,
                "score_seq_lens": kw.get("score_seq_lens") is not None,
                "brks": kw.get("bottom_right_key_start"),
            }
            emit(rec)
            if c <= 5 or c % 200 == 0:
                say(f"tfsc#{c} q={rec['q']} sk={rec['sk']} k={rec['k']} brks={rec['brks']}")
            return orig_tfsc(q_bshd, k_bshd, w_bsh, seq_lens, topk_k, return_topk_scores, **kw)

        mod._indexer_topk_from_score_chunks = _tfsc

        # ------------------------------------------------------------------ #
        # E2: staged-block poison of the wheel's output_indices allocation
        # ------------------------------------------------------------------ #
        def _verify_reuse(rows: int, k: int, device) -> float:
            """Fraction of a fresh torch.empty(rows,k) that carries the sentinel.

            The wheel's first real allocation after our stage is
            torch.empty(num_rows, top_k, int32) of exactly this shape, so this
            mimics it. Without this control a zero unwritten-slot count is
            ambiguous: correct kernel, or staged block never handed over?
            """
            key = (rows, k)
            if key in ctrl:
                return ctrl[key]
            frac = -1.0
            try:
                block = torch.full((rows, k), SENT, dtype=torch.int32, device=device)
                del block
                probe = torch.empty(rows, k, dtype=torch.int32, device=device)
                frac = float((probe == SENT).float().mean())
                del probe
            except Exception as e:  # noqa: BLE001
                say(f"staging control failed for {key}: {type(e).__name__}: {e}")
            ctrl[key] = frac
            if frac > 0.99:
                st["ctrl_shapes_ok"] += 1
                say(f"staging control OK rows={rows} k={k}: fresh empty() is {frac:.4f} sentinel")
            else:
                st["ctrl_shapes_fail"] += 1
                say(
                    f"STAGING CONTROL FAILED rows={rows} k={k}: fresh empty() only {frac:.4f} "
                    "sentinel -> allocator did not hand the staged block back; "
                    "unwritten-slot counts for this shape are INCONCLUSIVE"
                )
            return frac

        orig_topk = mod._indexer_top_k_one_chunk

        def _topk(scores_flat, seq_lens, topk_k, return_topk_scores):
            st["topk"] += 1
            n = st["topk"]
            rows, sk = scores_flat.shape
            k = int(topk_k)
            rec = {"kind": "topk", "call": n, "rows": int(rows), "sk": int(sk), "k": k,
                   "staged": False, "fix": _FIX}
            notes = []
            sl = None
            dev = scores_flat.device

            try:
                with torch.no_grad():
                    sl = seq_lens.to(device=dev, dtype=torch.long).clamp(min=0, max=sk)
                    rec.update(sl_min=int(sl.min()), sl_max=int(sl.max()))
                    # Which output branch: trivial full-write (window <= k) vs radix (window > k).
                    n_short = int((sl <= k).sum())
                    st["short_rows"] += n_short
                    st["radix_rows"] += int(rows) - n_short
                    rec.update(short_rows=n_short, radix_rows=int(rows) - n_short)
                    if sk >= SENT:
                        st["sk_ge_sentinel_calls"] += 1
                        notes.append(f"SENTINEL-AMBIGUOUS sk={sk} >= sentinel {SENT}")
                    # Regression watch on the refuted scores path (sampled).
                    if scores_every and n % scores_every == 0:
                        st["scores_scanned"] += 1
                        inv = torch.arange(sk, device=dev).unsqueeze(0) >= sl.unsqueeze(1)
                        n_inv = int(inv.sum())
                        if n_inv:
                            fin = torch.isfinite(scores_flat) & inv
                            n_fin = int(fin.sum())
                            rec.update(inv_elems=n_inv, inv_finite=n_fin,
                                       inv_nan=int((torch.isnan(scores_flat) & inv).sum()))
                            if n_fin:
                                st["inv_finite_calls"] += 1
                                notes.append(f"INVALID-REGION-FINITE {n_fin}/{n_inv}")
                        del inv
            except Exception as e:  # noqa: BLE001
                say(f"topk#{n} pre-audit failed: {type(e).__name__}: {e}")
                rec["pre_audit_error"] = f"{type(e).__name__}: {e}"

            # Stage the poisoned output block LAST — no allocation may occur
            # between the free and the wheel's torch.empty, or a different
            # block gets handed over.
            if _STAGE:
                try:
                    frac = _verify_reuse(int(rows), k, dev)
                    rec["ctrl_frac"] = frac
                    block = torch.full((int(rows), k), SENT, dtype=torch.int32, device=dev)
                    del block
                    st["staged_calls"] += 1
                    rec["staged"] = True
                except Exception as e:  # noqa: BLE001
                    say(f"topk#{n} staging failed: {type(e).__name__}: {e}")
                    rec["stage_error"] = f"{type(e).__name__}: {e}"

            out = orig_topk(scores_flat, seq_lens, topk_k, return_topk_scores)

            try:
                with torch.no_grad():
                    idx = out.get("indices") if isinstance(out, dict) else None
                    if idx is not None and sl is not None:
                        neg = idx < 0
                        rec["sel_neg"] = int(neg.sum())
                        # E2: unwritten slots = still carrying the sentinel.
                        if _STAGE:
                            unwritten = idx == SENT
                            n_un = int(unwritten.sum())
                            rec["unwritten_slots"] = n_un
                            if n_un:
                                bad = unwritten.any(dim=1)
                                n_rows_un = int(bad.sum())
                                st["unwritten_slots"] += n_un
                                st["unwritten_rows"] += n_rows_un
                                st["unwritten_calls"] += 1
                                rows_i = bad.nonzero().flatten()
                                det = []
                                for r in rows_i[:8].tolist():
                                    w = int(sl[r])
                                    det.append({
                                        "row": r, "window": w, "expected": min(k, w),
                                        "written": int((idx[r] != SENT).sum()),
                                        "unwritten": int(unwritten[r].sum()),
                                        "branch": "trivial" if w <= k else "radix",
                                    })
                                rec.update(unwritten_rows=n_rows_un, unwritten_detail=det)
                                notes.append(
                                    f"UNDER-WRITE {n_un} slots in {n_rows_un} rows "
                                    f"e.g. row {det[0]['row']} window={det[0]['window']} "
                                    f"written={det[0]['written']}/{det[0]['expected']} "
                                    f"branch={det[0]['branch']}"
                                )
                                if _FIX:
                                    idx.masked_fill_(unwritten, -1)
                                    st["fixed_slots"] += n_un
                                    rec["fixed_slots"] = n_un
                                    notes.append(f"FIX rewrote {n_un} slots to -1")
                            del unwritten
                        # E1: out-of-window selections (sentinel excluded — that is
                        # our own poison, already counted as an under-write).
                        oor = (idx >= sl.unsqueeze(1)) & ~neg
                        if _STAGE:
                            oor = oor & (idx != SENT)
                        n_oor = int(oor.sum())
                        rec["sel_oor"] = n_oor
                        if n_oor:
                            st["sel_oor_total"] += n_oor
                            st["sel_oor_calls"] += 1
                            bad_rows = oor.any(dim=1).nonzero().flatten()
                            det = [{"row": int(r), "window": int(sl[r]),
                                    "sample": idx[r][oor[r]][:8].tolist()}
                                   for r in bad_rows[:8].tolist()]
                            rec.update(rows_oor=int(bad_rows.numel()), oor_detail=det)
                            notes.append(
                                f"SELECTED-OUT-OF-RANGE {n_oor} in {int(bad_rows.numel())} rows "
                                f"e.g. row {det[0]['row']} window={det[0]['window']} "
                                f"sample={det[0]['sample'][:4]}"
                            )
                        del oor, neg
                        # E1: duplicates within a row — the allocator-garbage
                        # signature (previous-layer echoes repeat indices).
                        if dup_every and (n % dup_every == 0 or notes):
                            st["dup_scanned"] += 1
                            valid = idx >= 0
                            if _STAGE:
                                valid = valid & (idx != SENT)
                            # push invalid slots to the top so they cannot pair up
                            big = torch.iinfo(torch.int32).max
                            sorted_idx, _ = torch.where(valid, idx, torch.full_like(idx, big)).sort(dim=-1)
                            same = (sorted_idx[:, 1:] == sorted_idx[:, :-1]) & (sorted_idx[:, :-1] != big)
                            n_dup = int(same.sum())
                            rec["dup_slots"] = n_dup
                            if n_dup:
                                dup_rows = int(same.any(dim=1).sum())
                                st["dup_slots"] += n_dup
                                st["dup_rows"] += dup_rows
                                st["dup_calls"] += 1
                                rec["dup_rows"] = dup_rows
                                notes.append(f"DUPLICATE-INDICES {n_dup} slots in {dup_rows} rows")
                            del sorted_idx, same, valid
            except Exception as e:  # noqa: BLE001
                say(f"topk#{n} post-audit failed: {type(e).__name__}: {e}")
                rec["post_audit_error"] = f"{type(e).__name__}: {e}"

            if hb and n % hb == 0:
                rec["ctr"] = dict(st)
            emit(rec)
            if notes or n <= echo_n or (hb and n % hb == 0):
                tag = " | ".join(notes) if notes else ("early" if n <= echo_n else "hb")
                say(
                    f"topk#{n} rows={rows} sk={sk} k={k} sl=[{rec.get('sl_min')},{rec.get('sl_max')}] "
                    f"short/radix={rec.get('short_rows')}/{rec.get('radix_rows')} "
                    f"staged={rec['staged']} unwritten={rec.get('unwritten_slots', '-')} "
                    f"dup={rec.get('dup_slots', '-')} oor={rec.get('sel_oor', '-')} "
                    f"neg={rec.get('sel_neg', '-')} :: {tag}"
                )
            return out

        mod._indexer_top_k_one_chunk = _topk

        def _exit_summary() -> None:
            emit({"kind": "exit", **st, "ctrl": {f"{r}x{k}": v for (r, k), v in ctrl.items()}})
            say(f"EXIT counters {st}")

        atexit.register(_exit_summary)
        say(
            f"installed stage={_STAGE} fix={_FIX} sentinel={SENT} dup_every={dup_every} "
            f"scores_every={scores_every} dir={_DIR or '-'} (seam=_indexer_top_k_one_chunk)"
        )

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != _TARGET:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                try:
                    _install(module)
                except Exception as e:  # noqa: BLE001
                    print(f"[dsa_audit] install failed: {e}", file=sys.stderr, flush=True)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
