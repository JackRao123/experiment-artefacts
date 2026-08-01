# /// script
# requires-python = ">=3.10"
# dependencies = ["wandb"]
# ///
# LPS-1003: regenerate the GLM loss-curve comparison chart from live W&B data.
#
#   uv run plot_loss_compare.py                  # default: broken vs patched GLM runs
#   uv run plot_loss_compare.py --runs A,B,C     # any runs in the project, by name
#   uv run plot_loss_compare.py --no-open        # just write the HTML
#
# Needs WANDB_API_KEY in the environment (read access to baseten-training).
import argparse, datetime, json, os, subprocess, sys

DEFAULT_RUNS = [
    ("GLM-5.2-FP8_r1_b32_rank32", "broken image (5a4ae4d)", 1),
    ("GLM-5.2-FP8_r1_b32_rank32_dsatopk1", "patched image (0e0b65a)", 0),
]
# Validated categorical slots (light, dark) — see dataviz reference palette.
SLOTS = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"),
         ("#1baf7a", "#199e70"), ("#eda100", "#c98500")]

ap = argparse.ArgumentParser()
ap.add_argument("--project", default="baseten-training/oe-grader-sft")
ap.add_argument("--runs", help="comma-separated run names (default: the two GLM runs)")
ap.add_argument("--metric", default="train_mean_nll")
ap.add_argument("--marker", default="70:first IMA crash →",
                help="vertical marker as STEP:LABEL, or 'none'")
ap.add_argument("--out", default="glm_loss_compare.html")
ap.add_argument("--no-open", action="store_true")
a = ap.parse_args()

if a.runs:
    series = [(n.strip(), n.strip(), i % len(SLOTS)) for i, n in enumerate(a.runs.split(","))]
else:
    series = DEFAULT_RUNS

import wandb  # deferred: slow import

api = wandb.Api()
by_name = {}
for r in api.runs(a.project):
    by_name.setdefault(r.name, r)
data, states = {}, {}
for name, label, slot in series:
    r = by_name.get(name)
    if r is None:
        sys.exit(f"run {name!r} not found in {a.project}")
    pts = [(row.get("train_step"), row.get(a.metric))
           for row in r.scan_history(keys=["train_step", a.metric])]
    data[name] = [[s, round(v, 4)] for s, v in pts if v is not None and s is not None]
    states[name] = r.state
    print(f"{name}: {len(data[name])} points, state={r.state}")

now = datetime.datetime.now().strftime("%-I:%M %p, %b %-d")
mk_step, mk_label = (None, "") if a.marker == "none" else \
    (int(a.marker.split(":", 1)[0]), a.marker.split(":", 1)[1])
meta = {"crash_step": mk_step, "crash_label": mk_label, "metric": a.metric,
        "note": f"Generated {now}. " + "; ".join(
            f"{lbl}: {len(data[n])} steps ({states[n]})" for n, lbl, _ in series)}
sjs = [{"key": n, "label": lbl, "slot": sl} for n, lbl, sl in series]

light_vars = "\n".join(f"    --s{i}: {SLOTS[i][0]};" for i in range(len(SLOTS)))
dark_vars = "\n".join(f"      --s{i}: {SLOTS[i][1]};" for i in range(len(SLOTS)))
legend = "\n".join(
    f'      <span class="key"><span class="swatch" style="background:var(--s{sl})"></span>{lbl}</span>'
    for _, lbl, sl in series)

HTML = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GLM-5.2 SFT loss — {a.metric}</title>
<style>
  :root {{
    color-scheme: light dark;
  }}
  .viz-root {{
    color-scheme: light;
    --surface-1: #fcfcfb; --page: #f9f9f7;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
    --grid: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
{light_vars}
  }}
  @media (prefers-color-scheme: dark) {{
    .viz-root {{
      color-scheme: dark;
      --surface-1: #1a1a19; --page: #0d0d0d;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
{dark_vars}
    }}
  }}
  body {{ margin: 0; }}
  .viz-root {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--page); color: var(--text-primary); padding: 24px; min-height: 100vh; }}
  .card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
    padding: 20px 22px; max-width: 1060px; margin: 0 auto; }}
  h1 {{ font-size: 17px; font-weight: 650; margin: 0 0 2px; }}
  .sub {{ font-size: 13px; color: var(--text-secondary); margin: 0 0 14px; }}
  .legend {{ display: flex; gap: 18px; font-size: 12.5px; color: var(--text-secondary);
    margin-bottom: 10px; flex-wrap: wrap; }}
  .legend .key {{ display: inline-flex; align-items: center; gap: 7px; }}
  .legend .swatch {{ width: 14px; height: 3px; border-radius: 2px; display: inline-block; }}
  .chartwrap {{ position: relative; }}
  svg {{ width: 100%; height: auto; display: block; }}
  .tip {{ position: absolute; pointer-events: none; background: var(--surface-1);
    border: 1px solid var(--border); border-radius: 7px; padding: 7px 10px; font-size: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,.12); display: none; min-width: 170px; z-index: 3; }}
  .tip .t-step {{ color: var(--muted); margin-bottom: 3px; }}
  .tip .row {{ display: flex; justify-content: space-between; gap: 14px; }}
  .tip .row .name {{ display: inline-flex; align-items: center; gap: 6px; color: var(--text-secondary); }}
  .tip .row .val {{ font-variant-numeric: tabular-nums; color: var(--text-primary); }}
  .tip .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
  .note {{ font-size: 12px; color: var(--muted); margin-top: 10px; }}
</style></head><body>
<div class="viz-root">
  <div class="card">
    <h1>GLM-5.2-FP8 LoRA SFT — {a.metric} by step</h1>
    <p class="sub">Same script, data, and shuffle seed — identical batch order across runs.</p>
    <div class="legend">
{legend}
    </div>
    <div class="chartwrap" id="wrap">
      <svg id="chart" viewBox="0 0 1000 430" role="img" aria-label="Loss curves"></svg>
      <div class="tip" id="tip"></div>
    </div>
    <p class="note" id="note"></p>
  </div>
</div>
<script>
const DATA = {json.dumps(data)};
const META = {json.dumps(meta)};
const S = {json.dumps(sjs)};
const W = 1000, H = 430, M = {{ t: 16, r: 150, b: 34, l: 46 }};
const iw = W - M.l - M.r, ih = H - M.t - M.b;
const xmax = Math.max(...S.map(s => DATA[s.key].length ? DATA[s.key][DATA[s.key].length-1][0] : 0)) + 5;
const ymax = Math.ceil(Math.max(...S.flatMap(s => DATA[s.key].map(p => p[1]))) * 1.06 * 10) / 10;
const X = v => M.l + (v / xmax) * iw, Y = v => M.t + ih - (v / ymax) * ih;
const svg = document.getElementById("chart"), NS = "http://www.w3.org/2000/svg";
function el(n, at, parent) {{ const e = document.createElementNS(NS, n);
  for (const k in at) e.setAttribute(k, at[k]); (parent || svg).appendChild(e); return e; }}
for (let gy = 0; gy <= ymax + 1e-9; gy += 0.5) {{
  el("line", {{ x1: M.l, x2: M.l + iw, y1: Y(gy), y2: Y(gy), stroke: "var(--grid)", "stroke-width": 1 }});
  el("text", {{ x: M.l - 8, y: Y(gy) + 4, "text-anchor": "end", "font-size": 11.5,
    fill: "var(--muted)", "font-variant-numeric": "tabular-nums" }}).textContent = gy.toFixed(1);
}}
const xstep = xmax > 800 ? 100 : 50;
for (let gx = 0; gx <= xmax; gx += xstep) {{
  el("text", {{ x: X(gx), y: H - 12, "text-anchor": "middle", "font-size": 11.5,
    fill: "var(--muted)", "font-variant-numeric": "tabular-nums" }}).textContent = gx;
}}
el("text", {{ x: M.l + iw / 2, y: H - 0.5, "text-anchor": "middle", "font-size": 11.5, fill: "var(--muted)" }}).textContent = "train step";
el("line", {{ x1: M.l, x2: M.l + iw, y1: Y(0), y2: Y(0), stroke: "var(--baseline)", "stroke-width": 1 }});
if (META.crash_step !== null && META.crash_step <= xmax) {{
  el("line", {{ x1: X(META.crash_step), x2: X(META.crash_step), y1: M.t, y2: M.t + ih,
    stroke: "var(--baseline)", "stroke-width": 1, "stroke-dasharray": "4 4" }});
  el("text", {{ x: X(META.crash_step) + 6, y: M.t + 14, "font-size": 11, fill: "var(--muted)" }}).textContent = META.crash_label;
}}
for (const s of S) {{
  const pts = DATA[s.key];
  if (!pts.length) continue;
  const d = pts.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join("");
  el("path", {{ d, fill: "none", stroke: `var(--s${{s.slot}})`, "stroke-width": 2,
    "stroke-linejoin": "round", "stroke-linecap": "round" }});
  const last = pts[pts.length - 1];
  el("circle", {{ cx: X(last[0]), cy: Y(last[1]), r: 3.5, fill: `var(--s${{s.slot}})` }});
  el("text", {{ x: X(last[0]) + 8, y: Y(last[1]) + 4, "font-size": 12, "font-weight": 600,
    fill: "var(--text-primary)" }}).textContent = s.label;
}}
const cross = el("line", {{ y1: M.t, y2: M.t + ih, stroke: "var(--baseline)", "stroke-width": 1, visibility: "hidden" }});
const hdots = S.map(s => el("circle", {{ r: 4.5, fill: `var(--s${{s.slot}})`, stroke: "var(--surface-1)", "stroke-width": 2, visibility: "hidden" }}));
const tip = document.getElementById("tip"), wrap = document.getElementById("wrap");
const bykey = {{}}; S.forEach(s => {{ bykey[s.key] = new Map(DATA[s.key].map(p => [p[0], p[1]])); }});
wrap.addEventListener("mousemove", ev => {{
  const r = svg.getBoundingClientRect();
  const step = Math.round(((ev.clientX - r.left) / r.width * W - M.l) / iw * xmax);
  if (step < 0 || step > xmax) {{ hide(); return; }}
  cross.setAttribute("x1", X(step)); cross.setAttribute("x2", X(step)); cross.setAttribute("visibility", "visible");
  let rows = `<div class="t-step">step ${{step}}</div>`;
  S.forEach((s, i) => {{
    const v = bykey[s.key].get(step);
    if (v === undefined) {{ hdots[i].setAttribute("visibility", "hidden"); return; }}
    hdots[i].setAttribute("cx", X(step)); hdots[i].setAttribute("cy", Y(v)); hdots[i].setAttribute("visibility", "visible");
    rows += `<div class="row"><span class="name"><span class="dot" style="background:var(--s${{s.slot}})"></span>${{s.label}}</span><span class="val">${{v.toFixed(3)}}</span></div>`;
  }});
  tip.innerHTML = rows; tip.style.display = "block";
  const tw = tip.offsetWidth, px = (X(step) / W) * r.width;
  tip.style.left = Math.min(Math.max(px + 14, 4), r.width - tw - 4) + "px";
  tip.style.top = (ev.clientY - r.top + 18) + "px";
}});
wrap.addEventListener("mouseleave", hide);
function hide() {{ cross.setAttribute("visibility", "hidden"); hdots.forEach(d => d.setAttribute("visibility", "hidden")); tip.style.display = "none"; }}
document.getElementById("note").textContent = META.note;
</script>
</body></html>"""

open(a.out, "w").write(HTML)
print(f"wrote {a.out}")
if not a.no_open and sys.platform == "darwin":
    subprocess.run(["open", a.out])
