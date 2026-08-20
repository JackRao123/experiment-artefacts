# BRIEF — volta: PP+CP microbatch semantics + pad-to-131k packing (2026-08-10 night)

You are **volta**, working for **pauli** (manager, this Mac). Reply/report via:
`~/.agents/scripts/send-message.sh pauli "volta: <message>"`.
Report on milestones/blockers; don't go silent >30 min while active. Peer
`gibbs` runs the box bring-up and depends on your Step-1 memo — message them
directly too (`send-message.sh gibbs "volta: ..."`).

## Mission

Make PP>1 work with our CP/THD-packed data path. Two parts: (1) URGENT
analysis memo on how microbatches are formed and what PP requires of them;
(2) implement pad-to-131k partition padding so every microbatch is the same
size. Program context: we're bringing up GLM-5.2 PP2/CP8/EP8 @131k from tip
of trainers; see `experiment_artefacts/glm/lps_1062_perf/pp2cp8ep8/GOAL.md`.

Jack's reasoning (constraint 2): "Same size microbatch required between PP
ranks. This is required per call — not globally, but we can just fix it
globally to 131k because we use CP/THD packing. Pack every partition to 131k;
pad the remainder. Low-single-digit % waste is acceptable."

## Your workspace

- Mac worktree: `~/Documents/wt-pp2-packing`, branch
  `jackrao/lps-1062-pp2-packing` (off origin/main df831501). Jack's
  `~/Documents/trainers` checkout is read-only reference.
- Artefacts: `experiment_artefacts/glm/lps_1062_perf/pp2cp8ep8/` (memo goes
  here as `PACKING_MEMO.md`).

## Part 1 — analysis memo (do FIRST, gibbs is blocked on landmines you find)

Trace the path in `server/src/trainers_server/dp_worker/` (megatron_bridge
backend) from `/forward_backward` request datums → THD packing/partitioning →
Megatron fwd-bwd call. Answer precisely, with file:line:

1. How does one partition map to a microbatch? How is `num_microbatches`
   derived/communicated under PP? (Megatron's pipeline schedules need a
   consistent count across ranks.)
2. Where are tensor shapes for PP p2p communication determined? With THD
   varlen, do stages exchange/assume shapes (`seq_length`,
   micro_batch_size, hidden) — i.e., do UNEQUAL partition lengths break p2p
   (shape mismatch/hang) or are var-shapes supported?
3. Do all PP ranks receive the same partition list / datum stream? (Under
   PP, only the first stage embeds and the last computes loss, but every rank
   needs cu_seqlens for attention + the packed layout for CP. How does the
   non-first stage get its THD metadata?)
4. Loss/token-weighting: with padded tokens, where is the loss mask formed —
   will pad tokens contribute to loss/grad or token counts (TPS accounting,
   loss normalization)?
5. Anything PP-specific in the backend that's only exercised at PP>1
   (assertions, unimplemented branches, LoRA + PP interactions in
   fwd-bwd)? Note: PP16/PP8 golden configs ran with CP=1 historically; CP>1
   ran with PP=1. The INTERSECTION is virgin code paths.
6. Confirm tail-padded THD under CP is sound at tip: main has ef4ea4a8
   ("explicit pad_between_seqs for tail-padded THD under CP", LPS-1063 fix,
   PR #994). Our pad-to-131k plan leans on exactly this; check the pad
   mechanism you'd use hits that path (TE `pad_between_seqs=True`).

Deliver: `pp2cp8ep8/PACKING_MEMO.md` + message pauli AND gibbs with the
3-sentence version (esp. anything that changes the bring-up plan). Target:
within ~60-90 min.

## Part 2 — pad-to-131k implementation

Design (adjust to what you find in Part 1):
- When PP>1 (or behind an env/config flag, e.g. `BT_PACK_PAD_TO_MAX`), pad
  every packed partition to exactly `max_seq_len` tokens with pad tokens that
  are excluded from loss (mask/weight 0) and sit as a tail segment in the THD
  layout (`pad_between_seqs` handling per LPS-1063 — tail padding must be
  DECLARED, not inferred; the TE detect heuristic ignores tail padding and
  silently mis-attends).
- Pad token id: whatever the model/tokenizer designates (or any id with loss
  weight 0 — check how existing padding paths do it).
- CP interaction: padded region must still split evenly across CP ranks
  (131072 divisible by 2×CP=16: yes, 8192 per chunk) — confirm the CP
  sharding requires that and that a partial tail doesn't break the
  load-balanced (zigzag/contiguous) split.
- Tests (Mac-runnable if imports allow, else mark for on-box):
  (a) packer unit test: partitions all come out exactly 131072 with correct
  cu_seqlens + loss mask; (b) parity design: loss on [datums padded to 131k]
  == loss on same datums unpadded (CP1/PP1 reference) within tolerance —
  gibbs can execute on-box later.
- Commit to your branch; when green, tell pauli — we'll merge into gibbs's
  `jackrao/lps-1062-pp2cp8ep8` for the real-data phase. (Bring-up itself uses
  synthetic exactly-131k datums, so your code is NOT on the bring-up critical
  path — correctness and cleanliness over speed.)

## Part 3 (stretch) — waste estimate

Customer histogram (70% ≤32K / 28.5% 32-64K / 1.4% ≤131K): estimate padding
waste % under pack-to-131k with the current packer's bin-packing. Just a
number + method note in the memo.

## Reporting

Append to `pp2cp8ep8/NOTEBOOK.md` (timestamped). Message pauli on: memo done,
implementation done, tests green, blockers. Papercuts for friction
(`papercuts add ... --tag lps1062`).
