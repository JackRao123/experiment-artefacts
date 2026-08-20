# BLOCKER + STATUS — bohr, 2026-08-13 ~21:5x PDT (posted to the filesystem of record because messaging is down)

## The blocker (Mac-local, NOT the work)

This Mac's opendirectoryd lost the local user record (`id -un` → `501`, not
`jackrao`; `dscl` eServerError; `sudo` broken — "you do not exist in the passwd
database"). Consequences measured:

- **ALL ssh down** ("No user exists for uid 501" before any connection attempt)
  → I cannot reach box **wlxj8vw** (1×8 B200, hyd — RUNNING, parked mid-rung-1).
- **iTerm2 not answering AppleEvents** (osascript itself works; `tell
  application "iTerm2"` → -1728) → **send-message.sh is down fleet-wide**.
  The fleet's opencode PROCESSES are alive (ps shows them) but deaf.
- File IO, local git, this session (bohr) all fine.

Needs Jack (or a self-heal): bounce opendirectoryd / restart iTerm2 / reboot.
I poll ssh + osascript every few minutes and resume the moment they recover.

## Rung-1 state at the outage (wlxj8vw, dial chain mcore 06393114b, wheel 1.27.0 DONE)

- **Wheel rule DONE on B200**: frontend 1.26.0 → PyPI 1.27.0 (--no-deps; cu12
  backend 9.23.2.1 unchanged per the cu12-lane rule). Import+version PASS
  (import name is `cudnn` on this lineage). Both DSA files GREEN 5/23.3s vs
  B300's 5/22s — no divergence.
- **T1/T2/T4/T6/T7 PASS (5/7)** — the LoRA-trap grad-equivalence PASSES:
  adapter grads nonzero + matching, input grad correct, T2 negative control
  fires as designed (trap produces exactly None grads; source guard for
  b907b6153 present).
- **T3 fail = MY test-design bug** (process-global RNG tracker shared across
  arms → sequential arms see different dropout masks). Redesign in progress:
  RNG snapshot/restore (cpu + cuda + `get_cuda_rng_tracker().get_states()`) so
  each arm starts from identical state; the dial-vs-eager grad comparison then
  directly proves recompute mask fidelity (the actual §6.2 risk).
- **T5 fail = reference-arm bug under analysis**: the stock full-recompute
  reference's loss comes out requires_grad=False under the all-frozen +
  router-only fixture. Reading mcore's block path Mac-side (local clone has
  the branch) — suspect the block-level requires_grad force doesn't cover this
  fixture shape, or my config mutation didn't take. Fix is test-side; the DIAL
  path itself passed its trap checks (T1/T2 green).
- Test file: this folder, `test_recompute_dial_grad_equiv.py` (also staged
  on-box at the mcore tests dir). Runbook: `DIAL_VALIDATION_LADDER.md`.

## When ssh returns

Resume at: re-stage updated test file → full rung-1 re-run (expect 7/7) →
rung 2 (mem-probe table on lebesgue's cut-down snapshot when it lands) →
lebesgue's contract-smoke protocol. B/F W1c window stays on the B300 box per
fermi.
