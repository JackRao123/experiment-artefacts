# UPSTREAM PROPOSAL — per-layer recompute dial to unlock MoE all-to-all overlap at long sequence length

Drafted by serre, 2026-08-13 (night shift, LPS-1062). For Jack to forward
upstream or hand to the team. One page; measured numbers cited, unmeasured
slots marked **TO-FILL**. Companion docs:
`results/OVERLAP_PER_LAYER_RECOMPUTE_DESIGN.md` (the upstream dial: mechanism,
citations, risks, verification plan) and `results/EXECUTOR_CONTRACT_SCOPING.md`
(the trainer-side executor contract, our work).

## 1. Problem

On GLM-5.2-FP8 LoRA at 131k tokens (PP2/CP8/EP8, 2×8 B300), the MoE expert
all-to-all communication is ~24% of step time at our largest step size (31.4s
of the 131.4s d16 step, traced) and runs with almost no compute overlap (~8%
of its duration overlaps any compute today). Megatron's built-in fix,
`overlap_moe_expert_parallel_comm`, overlaps that communication with compute —
but at long sequence length it is blocked twice over:

1. **Memory (upstream's wall):** the flag hard-rejects whole-layer (full)
   recompute (transformer_config.py:2632-2642) because its fine-grained
   scheduler drives each layer as five explicit sub-module nodes and cannot
   see inside an opaque checkpoint. The only legal partial setting (selective
   recompute of attention) leaves the MoE intermediates saved: we measured
   258.8 GiB allocated (of a 275 GiB budget) with even one microbatch in
   flight. There is **no per-layer dial** — recompute is configurable by
   module *type*, not by layer *index*.
2. **Executor contract (our wall, found tonight):** the flag's combined-1F1B
   executor requires the training loop's forward step to return a schedule
   plan (`forward_step_func(..., return_schedule_plan=True)`,
   combined_1f1b.py:390-395). Our trainer's forward-step closures implement
   no such protocol — the probe died on a deterministic TypeError before
   memory was ever reached. Any consumer with a custom loss/data path (RL
   losses, chunked LM heads, packed-sequence iterators) hits the same wall.

## 2. The three-leg path

**(a) Executor-contract adoption — OUR trainer, our work.** Teach the
trainer's forward-step closures the schedule-plan protocol. Scoped tonight:
wrapper-level, not a re-architecture — the existing chunked-CE/RL loss code
runs verbatim inside the executor's loss node; ~100-150 LoC + tests, 0.5-2
days. Details: `EXECUTOR_CONTRACT_SCOPING.md`. This is the prerequisite for
(b) and (c) — and for A/B-ing any future overlap feature upstream ships.

**(b) Near-term memory path: fine-grained activation offloading (works
today).** mcore's module-level offload (`fine_grained_activation_offloading`)
is CI-tested upstream *with* the overlap flag + selective recompute + PP2 +
EP. It buys the overlap by shrinking the eager-activation bill over PCIe
instead of by recomputing. Costs: PCIe traffic, some TE-fuser disengagement,
LoRA-unvalidated. Probe result (fit + canary): **TO-FILL** (memory-only
scoping boot running tonight; the full flag-on probe waits on (a)).

**(c) Structural path (the upstream ask): per-layer recompute dial under the
overlap flag.** First K layers/stage run as opaque whole-layer checkpoints
(no overlap, minimal memory); the remaining L−K run the existing five-node
overlapped decomposition. The scheduler already supports heterogeneous layer
shapes (dense layers carry no-op comm nodes today) and unequal
forward/backward layer counts. Patch: one new config field (the four existing
recompute asserts stay untouched), one new per-layer callable builder (~100
lines), one branch in the plan builder (~10 lines). Effort: **~1 day
prototype, 2-3 days upstream-grade.** Full mechanism, the LoRA
silent-zero-grad trap (designed around), and the verification plan:
`OVERLAP_PER_LAYER_RECOMPUTE_DESIGN.md`.

## 3. The prize, quantified

Measured tonight on the mission config (131k, tok/s/GPU):

| lever | mechanism | est. step-time win |
|---|---|---|
| Full a2a overlap (needs (a)+(b) or (a)+(c)) | hide expert all-to-all behind paired microbatch compute | measured at d16 (L3 trace decomposition, `A2A_EXPOSURE_DECOMPOSITION.md`): 31.4s of the 131.4s step exposed; **schedule-hideable ceiling 25-28s/step (~19-21%)** — the program prize |
| Recompute-tax refund (comes free with (c)) | tonight every layer is recomputed; hybrid recomputes only K/40 | the recompute *replay* is the largest single block in the exposed a2a (11.7s, 37%) — the dial deletes that replay's a2a outright on overlapped layers, in addition to enabling the overlap |
| 16k-seqlen datapoint (customer regime) | the flag is *legal* at 16k — selective recompute fits there | **TO-FILL** if probe L4 runs tonight |

**What the L3 trace decomposition measured (d16, one traced step, gauss):** of
the 131.4-second step, 31.4 seconds are expert all-to-all with no compute
running alongside — and only ~30% of that is real data movement (10.2s at the
achieved 611 GB/s wire rate); the other ~70% is wait (straggler/imbalance and
schedule exposure). The largest single block is the all-to-all inside the
recompute *replay* (11.7s, 37% of the exposed total) — a cost that exists only
because every layer is recomputed today, and that only recompute-reduction or
overlap can touch. That strengthens leg (c) twice over: each layer the dial
moves out of recompute both gains the overlap and loses its replay a2a
outright. The schedule-hideable ceiling — what microbatch-interleaving can in
principle hide — is 25-28s/step (~19-21% of step), which is the prize this
proposal is buying; for scale, tonight's L1 flag covers only its ~2.2s
dispatch-backward slice, consistent with its small (+1.5-3%) expected value.

For calibration: tonight's headline fix (schedule repair) took us from 550 to
1052 tok/s/GPU; a further ~10-20% from this work would be ~1150-1250+.

## 4. Ask

1. **Upstream:** endorse (or take) the per-layer recompute dial (leg c) —
   ~2-3 days, design and risks already scoped. Without it, the overlap
   feature is marketing-only above ~16k tokens; we'd like that on the record
   as a roadmap gap either way.
2. **Upstream:** confirm the offloading path (leg b) is the intended
   near-term answer at long seqlen, and share any known LoRA+offload caveats
   (**TO-FILL** after our probe).
3. **Ours:** land the executor-contract adoption (leg a) — scoped, no
   upstream dependency.
