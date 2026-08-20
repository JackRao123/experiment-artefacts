# BLOCKED — lebesgue, 2026-08-13 ~23:5x CDT (Mac date)

**Fold into NOTEBOOK.md on recovery (append was blocked at write time).**

## 1-node smoke (hatch a) — build stalled by a Mac system episode; W2 UNAFFECTED

fermi overruled the not-worth-it (box-window economics) and ordered hatch (a).

LANDED + PUSHED before the episode (safe on remotes):
- `jackrao/lps-1062-overlap-contract-shim` @ a3da1223 — adds the DSA-legal
  `(12,2,2)` layout entry + test. N=12 is the smallest layer count fielding 4
  VPP chunks with legal starts (chunk starts ≤3 or ≡3 mod 4; N=8 only offers
  layers 3,7). Chunks 1-2 / 3-6 / 7-10 / 11-12, starts 1/3/7/11.
- The shim branch @ e13de4d7 + the mcore landmine-fix chain (b37c01f2e) — the
  W2 gate itself is unaffected.

WRITTEN to disk (intact; this episode is access-control, NOT data loss):
- `pp2cp8ep8/configs/trainer_shim_smoke_1node.json` — PP2/VPP2/CP2/EP2 @8192,
  LoRA32, flag on; passes the shim guards.
- `runs/overnight_20260813_overlap_campaign/SHIM_SMOKE_1NODE.md` — the 5-line
  protocol + honest random-init PASS bar (CE ≈ ln(154880) ≈ 11.95 DECREASING =
  computes+trains; proves contract/boot/landmine + DSA×executor, NOT the W2
  parity gate) + dial-probe caveats.
- `pp2cp8ep8/tools/build_smoke_snapshot.py` — the snapshot builder, rewritten
  to meta-device + incremental per-shard materialization (peak RAM ~4 GiB)
  after a 28 GiB whole-model build thrashed the campaign Mac.

NOT done: the snapshot was never built end-to-end (the Mac thrash + cascade
interrupted it). The builder is written but UNTESTED end-to-end.

## The episode (why lebesgue went dark)

Heavy memory pressure (the 28 GiB model build + campaign trace servers)
cascaded into: (1) per-user tccd crashed → TCC EPERM on ALL of
~/Documents, ~/Desktop, ~/Downloads — reads of existing files blocked, only
new-file creation works; user tccd won't restart (`killall` found none,
`launchctl kickstart gui/<uid>/com.apple.tccd` refused: "reentrancy
avoided"); (2) Tailscale MagicDNS (100.100.100.100) hanging → git/curl can't
resolve hosts (direct resolver 8.8.8.8 works). Cross-session messaging is also
down (send-message.sh uses osascript→iTerm injection, TCC-Automation-gated).

**Recovery needs Jack: re-grant the terminal's access in System Settings →
Privacy & Security (Full Disk Access / Files & Folders for iTerm2), or reboot
(restarts the user tccd and likely fixes Tailscale DNS too).**

## On recovery, lebesgue (or whoever picks this up):

1. Re-run the snapshot build end-to-end to verify:
   `python3 tools/build_smoke_snapshot.py --output-dir /root/lps1062/glm52_smoke`
   (~4 min on an uncontended host; on the Mac, expect contention).
2. Verify the snapshot loads + is detected as GLM-5.2 (config.json carries
   architectures=GlmMoeDsaForCausalLM, model_type=glm_moe_dsa).
3. Hand snapshot + config + protocol to bohr (after bohr's dial rung 2).
4. Delete the stale `tools/build_shim_smoke_snapshot.py` (kernel-locked at
   write time; the canonical builder is `tools/build_smoke_snapshot.py`).
5. Restore the parked untracked scratch dirs from
   `$TMPDIR/opencode/trainers-scratch-park/` (mudith_openevidence_training/,
   ep8prompt.md) back to the trainers repo root — they were moved aside to get
   the pre-push `make check` green and never moved back.
