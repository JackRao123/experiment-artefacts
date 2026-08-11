#!/usr/bin/env python3
"""W5 — two-rank straggler analysis (LPS-1062 overlap_design).

Compares the per-layer-pass MoE all-to-all timing of TWO ranks' traces of the
SAME training step (e.g. rank 0 and rank 8) and answers:

  1. How large is the cross-rank skew? Per call index (900 dispatch + 900
     combine token a2a per step = 75 MoE layers x 4 datums x 3 passes), the
     start/end time difference between the two ranks after aligning each
     trace to its own step start. Reported p50/p90 (signed and absolute).
  2. Does the hot-expert rank predict the straggler? The hot-expert rank for
     a combine call is the rank whose combine **In msg nelems** is larger
     (it received more tokens to expert-process). We test whether the rank
     that is hotter per call tends to START/END its combine later, via
     Spearman/Pearson correlation between (B_in - A_in) and (B_end - A_end),
     plus the hit rate P(hotter rank ends later), plus the same test on the
     dispatch side (hot = larger dispatch Out, i.e. more tokens to receive).

Inputs: two kineto .pt.trace.json files (one per rank), steady window only
(never window 1). Clock bases differ across ranks, so each trace is aligned
to its own step start (first ProfilerStep#0 slice if present, else first
SendRecv kernel); intra-step clock drift over ~47 s is sub-ms against
millisecond skew.

Usage:
    python straggler_analysis.py RANK_A.pt.trace.json RANK_B.pt.trace.json \
        [--label-a rank0] [--label-b rank8] [--out SUMMARY.md]

    # mechanical self-test (skew must be ~0, verdict inconclusive):
    python straggler_analysis.py TRACE TRACE --self-test

Dependencies (all three required; the script exits with a clear message if
any is missing):
    pip install "perfetto==56.1" pandas numpy
trace_processor v56.1 is checker-authoritative for this project (>=v57 keeps
~5.4k overlapping slices that v56.1 drops — see dispatcher_opt/
check_acceptance.py's header note). pandas is required by the perfetto
python API's as_pandas_dataframe(); numpy is used directly.
"""

from __future__ import annotations

import argparse
import sys

_MISSING = []
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
    _MISSING.append("numpy")
try:
    import pandas  # noqa: F401  (required by perfetto's as_pandas_dataframe)
except ImportError:  # pragma: no cover
    _MISSING.append("pandas")
try:
    from perfetto.trace_processor import TraceProcessor
except ImportError:  # pragma: no cover
    TraceProcessor = None
    _MISSING.append("perfetto")
if _MISSING:  # pragma: no cover
    sys.stderr.write(
        "ERROR: missing python package(s): "
        + ", ".join(_MISSING)
        + '\n  pip install "perfetto==56.1" pandas numpy\n'
    )
    sys.exit(2)


def load_token_a2a(trace_path: str) -> dict:
    """Extract the EP-group token (BFloat16) SendRecv kernels, time-ordered.

    Returns dict with ts, dur, in_nelems, out_nelems (numpy arrays), and
    step_start (int, ns) for alignment.
    """
    tp = TraceProcessor(trace=trace_path)

    step = tp.query("SELECT ts FROM slice WHERE name='ProfilerStep#0' LIMIT 1").as_pandas_dataframe()
    step_start = int(step.ts.iloc[0]) if len(step) else None

    sr = tp.query("""
        SELECT s.ts, s.dur,
               ine.display_value AS in_nelems,
               one.display_value AS out_nelems,
               pg.display_value  AS pgroup,
               dt.display_value  AS dtype
        FROM slice s
        JOIN args ine ON ine.arg_set_id = s.arg_set_id AND ine.key = 'args.In msg nelems'
        JOIN args one ON one.arg_set_id = s.arg_set_id AND one.key = 'args.Out msg nelems'
        JOIN args pg  ON pg.arg_set_id  = s.arg_set_id AND pg.key  = 'args.Process Group Description'
        JOIN args dt  ON dt.arg_set_id  = s.arg_set_id AND dt.key  = 'args.dtype'
        WHERE s.category='kernel' AND s.name LIKE '%SendRecv%'
          AND dt.display_value = 'BFloat16'
          AND pg.display_value LIKE '%EXPERT%'
        ORDER BY s.ts
    """).as_pandas_dataframe()
    if len(sr) == 0:
        raise RuntimeError(f"{trace_path}: no expert-group BFloat16 SendRecv kernels found")
    if step_start is None:
        step_start = int(sr.ts.iloc[0])
    return {
        "ts": sr.ts.to_numpy(dtype=np.int64) - step_start,
        "dur": sr.dur.to_numpy(dtype=np.int64),
        "in_nelems": sr.in_nelems.astype(float).to_numpy(),
        "out_nelems": sr.out_nelems.astype(float).to_numpy(),
        "path": trace_path,
    }


def split_dispatch_combine(a: dict) -> tuple[dict, dict]:
    """Consecutive-pair split (validated against gated-v2-4mb-steady: token
    kernels strictly alternate dispatch/combine on the comm stream; pairing
    reproduces laplace's 10.85/16.64 ms gaps exactly)."""
    idx_d = np.arange(0, len(a["ts"]) - 1, 2)
    idx_c = np.arange(1, len(a["ts"]), 2)
    dispatch = {k: v[idx_d] for k, v in a.items() if k != "path"}
    combine = {k: v[idx_c] for k, v in a.items() if k != "path"}
    return dispatch, combine


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def skew_stats(delta_ms: np.ndarray) -> dict:
    """Skew percentiles. The PRIMARY numbers are median-centered: a constant
    offset across all calls is a cross-host clock / capture-window artifact
    (the step markers are CPU-side and not clock-comparable across hosts), not
    per-call straggler signal — proven by the duration-floor sanity check (a
    genuine X ms posting skew would show as >=X ms a2a durations on the
    waiting rank). The raw mean offset is kept for the footnote only.
    """
    centered = delta_ms - np.median(delta_ms) if len(delta_ms) else delta_ms
    return {
        "n": int(len(delta_ms)),
        # median-centered (PRIMARY)
        "c_p10": pct(centered, 10),
        "c_p50": pct(centered, 50),
        "c_p90": pct(centered, 90),
        "c_abs_p50": pct(np.abs(centered), 50),
        "c_abs_p90": pct(np.abs(centered), 90),
        # raw (footnote only — clock artifact lives here)
        "raw_mean": float(np.mean(delta_ms)) if len(delta_ms) else float("nan"),
        "raw_p50": pct(delta_ms, 50),
    }


def duration_stats(dur_ms: np.ndarray) -> dict:
    return {
        "min": float(dur_ms.min()) if len(dur_ms) else float("nan"),
        "p50": pct(dur_ms, 50),
        "p90": pct(dur_ms, 90),
    }


def hot_rank_test(size_delta: np.ndarray, end_delta_ms: np.ndarray):
    """Does the hotter rank (larger msg) finish later?

    size_delta: B_size - A_size per call (elems). end_delta_ms: B_end - A_end.
    End deltas are median-centered first: a constant offset across all calls
    is a clock/capture-window artifact (or, ambiguously, a uniform rank lag —
    reported separately in the skew table), not per-call straggler signal.
    Correlations are shift-invariant; the hit rate is not, hence centering.
    """
    if len(size_delta) == 0:
        return None
    mask = size_delta != 0
    n = int(mask.sum())
    if n < 30:
        return {"n": n, "note": "too few asymmetric calls (<30)"}
    sd = size_delta[mask]
    ed = end_delta_ms[mask] - np.median(end_delta_ms[mask])
    # Spearman via rank transform (robust to heavy tails in both axes)
    sr = np.argsort(np.argsort(sd))
    er = np.argsort(np.argsort(ed))
    spearman = float(np.corrcoef(sr, er)[0, 1])
    pearson = float(np.corrcoef(sd, ed)[0, 1])
    hit = float((np.sign(sd) == np.sign(ed)).mean())
    # weighted: count big-skew calls more (the ones that matter for the wall)
    big = np.abs(ed) >= np.percentile(np.abs(ed), 75)
    hit_big = float((np.sign(sd[big]) == np.sign(ed[big])).mean()) if big.sum() >= 10 else float("nan")
    return {
        "n": n,
        "spearman": spearman,
        "pearson": pearson,
        "hit_rate": hit,
        "hit_rate_top_quartile_skew": hit_big,
    }


def fmt_skew(name: str, s: dict) -> str:
    return (
        f"| {name} | {s['n']} | {s['c_p10']:+.2f} | {s['c_p50']:+.2f} | {s['c_p90']:+.2f} | "
        f"{s['c_abs_p50']:.2f} | {s['c_abs_p90']:.2f} |"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace_a", help="rank A trace (.pt.trace.json)")
    ap.add_argument("trace_b", help="rank B trace (.pt.trace.json)")
    ap.add_argument("--label-a", default="rankA")
    ap.add_argument("--label-b", default="rankB")
    ap.add_argument("--out", help="write the markdown summary here (default: stdout)")
    ap.add_argument("--self-test", action="store_true",
                    help="mechanical check: same trace twice — skew must be ~0")
    args = ap.parse_args()

    a = load_token_a2a(args.trace_a)
    b = load_token_a2a(args.trace_b)
    la, lb = args.label_a, args.label_b

    lines = []
    lines.append(f"# W5 two-rank straggler analysis — {la} vs {lb}")
    lines.append("")
    lines.append(f"- A: `{args.trace_a}`")
    lines.append(f"- B: `{args.trace_b}`")
    lines.append(
        "- Alignment: each trace zeroed at its own step start (ProfilerStep#0 or first "
        "SendRecv). Token a2a = BFloat16 SendRecv on the expert group; dispatch/combine "
        "split by consecutive pairing (validated on gated-v2-4mb-steady)."
    )
    lines.append("")

    if len(a["ts"]) != len(b["ts"]):
        lines.append(
            f"**WARNING: call-count mismatch — {la} {len(a['ts'])} vs {lb} {len(b['ts'])} "
            "token a2a calls. Traces may cover different windows; truncating to the "
            "common prefix.**"
        )
    n = min(len(a["ts"]), len(b["ts"]))
    for d in (a, b):
        for k in ("ts", "dur", "in_nelems", "out_nelems"):
            d[k] = d[k][:n]

    a_d, a_c = split_dispatch_combine(a)
    b_d, b_c = split_dispatch_combine(b)
    m = min(len(a_d["ts"]), len(b_d["ts"]))
    for d in (a_d, b_d, a_c, b_c):
        for k in ("ts", "dur", "in_nelems", "out_nelems"):
            d[k] = d[k][:m]

    # --- 1. skew distributions (median-centered is PRIMARY) ------------------
    rows = []
    for name, x, y in (
        ("dispatch start", a_d["ts"], b_d["ts"]),
        ("dispatch end", a_d["ts"] + a_d["dur"], b_d["ts"] + b_d["dur"]),
        ("combine start", a_c["ts"], b_c["ts"]),
        ("combine end", a_c["ts"] + a_c["dur"], b_c["ts"] + b_c["dur"]),
    ):
        rows.append((name, skew_stats((y - x) / 1e6)))

    lines.append(f"## 1. Cross-rank skew ({lb} − {la}, per call index, MEDIAN-CENTERED)")
    lines.append("")
    lines.append(
        "Primary table. Each series is centered on its own median before percentiling: "
        "a constant offset across all calls is a cross-host clock / capture-window "
        "artifact (the step markers are CPU-side and not clock-comparable across "
        "hosts), not per-call straggler signal — see the duration-floor sanity check "
        "below. The raw signed offsets are demoted to the footnote."
    )
    lines.append("")
    lines.append("| series | n | p10 (ms) | p50 (ms) | p90 (ms) | p50 |skew| (ms) | p90 |skew| (ms) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, s in rows:
        lines.append(fmt_skew(name, s))
    lines.append("")
    worst = max(rows, key=lambda r: r[1]["c_abs_p90"])
    lines.append(
        f"Largest centered p90 |skew|: **{worst[0]} = {worst[1]['c_abs_p90']:.2f} ms**. "
        + (
            "SELF-TEST: values should all be ~0 (same trace twice)."
            if args.self_test
            else ""
        )
    )
    lines.append("")

    # --- 1b. duration-floor sanity check ---------------------------------------
    dur_rows = []
    for label, d in ((la, a), (lb, b)):
        d_d, d_c = split_dispatch_combine(d)
        dur_rows.append((label, "dispatch", duration_stats(d_d["dur"] / 1e6)))
        dur_rows.append((label, "combine", duration_stats(d_c["dur"] / 1e6)))
    lines.append("## 1b. Duration-floor sanity check (per-rank a2a durations)")
    lines.append("")
    lines.append(
        "Collectives are synchronizing: a genuine posting skew of X ms would appear "
        "as >=X ms a2a DURATIONS on the waiting rank. Both ranks' duration floors "
        "therefore bound the real skew from below-zero — a large constant offset in "
        "the raw skew table with ~ms-scale duration floors on both ranks is a clock "
        "artifact, not real skew."
    )
    lines.append("")
    lines.append("| rank | series | min (ms) | p50 (ms) | p90 (ms) |")
    lines.append("|---|---|---:|---:|---:|")
    for label, series, s in dur_rows:
        lines.append(f"| {label} | {series} | {s['min']:.2f} | {s['p50']:.2f} | {s['p90']:.2f} |")
    lines.append("")

    # --- 1c. footnote: raw signed offsets (clock artifact) ----------------------
    lines.append("## 1c. Footnote — raw signed offsets (NOT per-call skew)")
    lines.append("")
    lines.append(
        "Raw mean signed offsets ("
        + "; ".join(f"{name} {s['raw_mean']:+.2f} ms" for name, s in rows)
        + "). A ~constant value across all four series is the cross-host clock / "
        "capture-window artifact — the step markers are CPU-side and not "
        "clock-comparable across hosts (proven by the duration floors in 1b: a real "
        "posting skew of that size would inflate the waiting rank's durations to the "
        "same size). Use only the median-centered rows in §1 for per-call skew."
    )
    lines.append("")

    # --- 2. hot-expert rank vs straggler --------------------------------------
    lines.append("## 2. Does the hot-expert rank predict the straggler?")
    lines.append("")
    lines.append(
        "Per call, `size delta` = B − A message size (elems); `end delta` = B − A end "
        "time (ms), **median-centered** (a constant offset across all calls is a "
        "clock/capture-window artifact, not per-call straggler signal — see the §1c "
        "footnote). Positive end delta = B finishes later. Hit rate = fraction of "
        "asymmetric calls where the hotter rank finishes later."
    )
    lines.append("")

    # combine: the receiver's In size = tokens it must expert-process (hot experts)
    comb = hot_rank_test(b_c["in_nelems"] - a_c["in_nelems"], ((b_c["ts"] + b_c["dur"]) - (a_c["ts"] + a_c["dur"])) / 1e6)
    # dispatch: the receiver's Out size = tokens it will receive (pre-positioned heat)
    disp = hot_rank_test(b_d["out_nelems"] - a_d["out_nelems"], ((b_d["ts"] + b_d["dur"]) - (a_d["ts"] + a_d["dur"])) / 1e6)

    lines.append("| test | n asymmetric | Spearman | Pearson | hit rate | hit rate (top-quartile skew) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, r in (("combine: In-size vs end delay", comb), ("dispatch: Out-size vs end delay", disp)):
        if r is None:
            lines.append(f"| {name} | 0 | — | — | — | — |")
        elif "note" in r:
            lines.append(f"| {name} | {r['n']} | {r['note']} | | | |")
        else:
            lines.append(
                f"| {name} | {r['n']} | {r['spearman']:+.3f} | {r['pearson']:+.3f} | "
                f"{r['hit_rate']:.1%} | {r['hit_rate_top_quartile_skew']:.1%} |"
            )
    lines.append("")

    # --- 3. verdict -------------------------------------------------------------
    lines.append("## 3. Verdict")
    lines.append("")
    if args.self_test:
        ok = all(
            abs(r[1]["raw_mean"]) < 0.01 and r[1]["c_abs_p90"] < 0.01 for r in rows
        )
        lines.append(
            f"Self-test {'PASS' if ok else 'FAIL'}: skew on identical traces is "
            f"{'~0 as required' if ok else 'NONZERO — the alignment/pairing is broken'}."
        )
    else:
        c = comb if comb and "spearman" in comb else None
        d = disp if disp and "spearman" in disp else None
        if c is None and d is None:
            lines.append("Insufficient asymmetric calls for a verdict.")
        else:
            best_name, best = ("combine In-size", c) if (c and (not d or abs(c["spearman"]) >= abs(d["spearman"]))) else ("dispatch Out-size", d)
            strength = (
                "STRONG" if abs(best["spearman"]) >= 0.5 else
                "MODERATE" if abs(best["spearman"]) >= 0.3 else
                "WEAK"
            )
            direction = "hotter rank finishes LATER (hot-expert predicts the straggler)" if best["spearman"] > 0 else "hotter rank finishes EARLIER (hot-expert does NOT predict the straggler)"
            lines.append(
                f"**{strength} signal via {best_name}:** Spearman {best['spearman']:+.3f}, "
                f"hit rate {best['hit_rate']:.1%} over {best['n']} asymmetric calls — {direction}."
            )
            lines.append("")
            lines.append(
                "Sizing the de-skew upside: the median-centered p90 |skew| rows in §1 "
                "bound what perfect cross-rank balance could recover per call; multiply "
                "by 900 calls/step for a first-order ceiling (overlap with comm slack "
                "reduces the realized fraction). Note this pairwise view covers only 2 "
                "of 16 ranks — a weak pairwise result does not bound effects involving "
                "the other 14 ranks or within-rank queueing."
            )
    lines.append("")

    text = "\n".join(lines)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
