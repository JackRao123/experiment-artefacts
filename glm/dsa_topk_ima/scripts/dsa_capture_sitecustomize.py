"""Env-gated DSA top-k instrumentation, injected via PYTHONPATH sitecustomize.

Deploy as <dir>/sitecustomize.py and prepend <dir> to PYTHONPATH of the trainer
ranks. Does nothing unless one of the env vars below is set.

Modes (combinable):

  BT_DSA_PROBE=1
      Per _indexer_top_k_one_chunk call, compute the per-row count of scores
      above the fp16 minimum (-65504). Log a line whenever any row's count is
      below topk+BT_DSA_PROBE_MARGIN (default 2048+2048), i.e. the row is
      within `margin` of the candidate-flood regime of the 1.26.0 OOB bug.
      Adds one GPU->CPU sync per call: acceptable for diagnostic runs only.

  BT_DSA_CAPTURE_DIR=/path
      Keep a per-process ring buffer (BT_DSA_RING, default 48) of CPU copies
      of every top-k call's inputs. On torch.AcceleratorError (or any
      exception) inside the call, dump the ring + the failing call to
      $BT_DSA_CAPTURE_DIR/rank<R>_pid<P>/ before re-raising. The pre-call CPU
      copy forces a stream sync; record that reproduction still occurs with
      capture enabled before trusting captured artifacts.

The hook targets the vendored Megatron module
megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels
and wraps _indexer_top_k_one_chunk via an import-time meta-path hook so it
works regardless of import order.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
import time

_TARGET = "megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels"
_PROBE = os.environ.get("BT_DSA_PROBE") == "1"
_CAP_DIR = os.environ.get("BT_DSA_CAPTURE_DIR")

if _PROBE or _CAP_DIR:

    def _install(mod) -> None:
        import collections

        import torch

        orig = mod._indexer_top_k_one_chunk
        rank = os.environ.get("RANK", "?")
        margin = int(os.environ.get("BT_DSA_PROBE_MARGIN", "2048"))
        heartbeat = int(os.environ.get("BT_DSA_PROBE_HEARTBEAT", "200"))
        ring_n = int(os.environ.get("BT_DSA_RING", "48"))
        ring: collections.deque = collections.deque(maxlen=ring_n)
        state = {"n": 0}
        FP16_MIN = -65504.0

        def _dump(tag: str, current=None) -> None:
            if not _CAP_DIR:
                return
            d = os.path.join(_CAP_DIR, f"rank{rank}_pid{os.getpid()}")
            os.makedirs(d, exist_ok=True)
            try:
                if current is not None:
                    torch.save(current, os.path.join(d, f"{tag}_failing_call.pt"))
                for i, entry in enumerate(ring):
                    torch.save(entry, os.path.join(d, f"{tag}_ring{i:03d}.pt"))
                print(f"[dsa_capture r{rank}] dumped {len(ring)} ring entries + failing call -> {d}",
                      file=sys.stderr, flush=True)
            except Exception as e:  # noqa: BLE001 — dumping must never mask the real error
                print(f"[dsa_capture r{rank}] dump failed: {e}", file=sys.stderr, flush=True)

        def wrapper(scores_flat, seq_lens, topk_k, return_topk_scores):
            state["n"] += 1
            n = state["n"]
            entry = None
            if _CAP_DIR:
                entry = {
                    "call": n,
                    "scores_flat": scores_flat.detach().to("cpu", non_blocking=False),
                    "seq_lens": seq_lens.detach().to("cpu", non_blocking=False),
                    "topk_k": int(topk_k),
                    "return_topk_scores": bool(return_topk_scores),
                    "ts": time.time(),
                }
                ring.append(entry)
            if _PROBE:
                try:
                    with torch.no_grad():
                        L = seq_lens.to(scores_flat.device)
                        valid = torch.arange(scores_flat.shape[-1], device=scores_flat.device)[None, :] < L[:, None]
                        above = ((scores_flat > FP16_MIN) & valid).sum(dim=-1)
                        eff_k = torch.minimum(L, torch.full_like(L, int(topk_k)))
                        slack_t = above - eff_k
                        # Only rows with eff_k == topk_k can flood (short rows
                        # trivially select everything); track them separately.
                        full_rows = L >= int(topk_k)
                        real_flood = bool((slack_t[full_rows] < 0).any().item()) if full_rows.any() else False
                        slack_full = int(slack_t[full_rows].min().item()) if full_rows.any() else 10**9
                        slack = slack_t.min().item()
                        if real_flood or slack_full < margin or (heartbeat and n % heartbeat == 0):
                            masked = torch.where(valid, scores_flat, scores_flat.new_tensor(float("inf")))
                            real_min = float(masked.min().item())
                            below_total = int(((scores_flat <= FP16_MIN) & valid).sum().item())
                            q01 = float(torch.quantile(
                                scores_flat[valid].float()[:: max(1, int(valid.sum().item()) // 100000)], 0.01
                            ).item())
                            print(
                                f"[dsa_probe r{rank}] call {n} shape={tuple(scores_flat.shape)} "
                                f"k={int(topk_k)} slack_full={slack_full} slack_any={slack} "
                                f"below_fp16min_total={below_total} real_min={real_min:.4e} q01={q01:.4e} "
                                f"{'REAL_FLOOD' if real_flood else ('NEAR' if slack_full < margin else 'hb')}",
                                file=sys.stderr, flush=True,
                            )
                except Exception as e:  # noqa: BLE001
                    print(f"[dsa_probe r{rank}] probe failed: {e}", file=sys.stderr, flush=True)
            try:
                return orig(scores_flat, seq_lens, topk_k, return_topk_scores)
            except BaseException:
                _dump(f"call{n}", entry)
                raise

        mod._indexer_top_k_one_chunk = wrapper
        print(f"[dsa_capture r{rank}] installed on {_TARGET} "
              f"(probe={_PROBE} capture={'on' if _CAP_DIR else 'off'} ring={ring_n})",
              file=sys.stderr, flush=True)

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
                    print(f"[dsa_capture] install failed: {e}", file=sys.stderr, flush=True)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
