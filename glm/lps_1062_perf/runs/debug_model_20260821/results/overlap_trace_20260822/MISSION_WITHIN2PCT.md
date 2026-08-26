# Overnight mission: offload within 2% of baseline (2026-08-22/23)

Arm names (Jack's standing definitions): `offload` = 4-group offload +
fix knobs (K=6, unchained); `baseline` = no recompute, no offload;
`fullrecompute` = full/uniform recompute. Method: one change per A/B,
mechanisms must be observed or bisected, never inferred-and-reported.

## Experiment log

| # | change under test | prediction | result | verdict |
|---|---|---|---|---|
| E1 | gap ledger (traced pair, ES off) | — | fwd +64.4 ms idle (EventSync +39.7, streamSync/.item +16-22, eventQuery +6), bwd +16.2, mid/tail 0; ~2 ms unattributed | ledger complete |
| E2 | D2H copy chunking 64 MiB (`BT_OFFLOAD_COPY_CHUNK_BYTES`) | fwd EventSync 69.5→~35 ms | EventSync 73.8 ms (unchanged); median 9,272 (worse); new 194.5 ms bwd `cudaMalloc` stall | **refuted** — engine does not interleave between one stream's queued chunks |
| E3 | D2H pacing (`BT_OFFLOAD_D2H_PACED`, chain groups on the d2h stream) | fwd EventSync → ≤~40 ms | EventSync 69.1 ms (unchanged); median 10,021 (+1.4% vs same-config rerun = within run noise) | **refuted** — queue was never deep; pacing is a no-op |
| E4 | microscopy of one readback wait | — | readback becomes *ready* only at producer completion (~24 ms, baseline-equal), then waits behind 1-2 in-flight bulk copies (~18 ms, offload-specific); engine drains the bulk stream before servicing the ready tiny copy | mechanism **observed** |
| E5 | sequence-length probe 16k/32k (8 windows each, median) | gap % shrinks as per-layer compute grows vs copy volume | 8k −16..18%, 16k **−16.7%**, 32k **−17.0%** (16k aggregate looked better only via one baseline outlier window) | **refuted** — the gap is scale-invariant: with DSA, per-token compute and per-token copy volume are both ~constant in seq, so the saturation ratio never improves. This also invalidates the earlier "shrinks at 131k" claim. Memory saving scales as expected (16.9 GiB @16k, 33.7 GiB @32k) |
| E6 | dispatcher metadata-readback stream at high priority (`BT_MOE_DTOH_HIGH_PRIORITY=1`, token_dispatcher.py) | if copy scheduling honors stream priority, fwd EventSync 69→~35 ms and median gap −5% | EventSync 73.6 ms (unchanged); arm median 10,307 = run noise | **refuted** — the copy engine ignores stream priority for these transfers |
| E7 | reload prefetch at backward entry (`on_backward_entry` in manager + `backward_step` in schedules.py; active whenever `BT_OFFLOAD_PREFETCH_DEPTH>0`) | bwd idle 38→~22 ms, H2D window starts ~430 ms instead of ~550 ms; parity in the ≤3e-6 band | H2D window **[430.0, 766.8]** (exactly backward entry), bwd idle 38→31.2 ms, bwd wall 350→343 ms, parity holds | **confirmed, partial** — ~7-11 ms recovered; ~9 ms of join waits remain vs baseline's 22 ms bwd idle |

Additional sample-count correction: the heavy-tail window class is NOT
reliably suppressed by expandable_segments:False. Tally across ES-off
offload arms: 0/18 (first two arms), then chunk64 2/10, prio 0/10,
paced 0/10, bentry 2/10. The tail is episodic with ES off and frequent
with ES on. Its observed ES-off signature is a ~194 ms `cudaMalloc`
growing an h2d-stream allocator segment mid-run; trigger bisection
remains open.

Also checked and closed: mcore's dispatcher `cuda_sync_point` machinery is
already maximally deferred at this topology (EP1 → "before_finish"), and
TE GroupedLinear requires host-side `m_splits` at this pin — the per-layer
host sync itself is irreducible without upstream surgery (device-side
grouped-GEMM sizing) or SM-driven copies that leave the DMA engine free.

## Corrected mechanism (observed, replaces the "deep queue" story)

At 8,192 tokens the offload D2H volume (151 ms at 53 GiB/s) nearly fills
the forward wall (~190 ms), so the copy engine is almost always busy when
each MoE layer's dispatcher readback becomes ready. The engine finishes
in-flight bulk copies before servicing a ready tiny copy from another
stream and will not interleave inside a stream's queued commands
(E2/E3 falsified both interleaving levers). Per readback that costs the
residual of 1-2 group copies (~10-20 ms) × 4 readbacks + the same effect
on `.item()`/small syncs ≈ the whole +64 ms forward gap. This is
arithmetic saturation at this sequence length: with full copy volume and
full memory saving, the collision cannot be scheduled away at 8k.
Consequently the within-2% question at fixed 8k reduces to whether the
copy-to-compute ratio improves with sequence length → E5.

## Correction to the expandable_segments finding

The heavy-tail stall class is **not** exclusively ES-on: E2's arm (ES off)
showed a directly observed 194.5 ms `cudaMalloc` inside a backward reload
(`torch.empty` on the h2d stream growing a fresh segment) plus 59 ms of
blocked `cudaEventQuery`. Current corrected statement: the tail class is
allocator segment-lifecycle work blocking the CUDA context — ES map/unmap
when ES is on, raw `cudaMalloc` growth when off; ES-on makes it much more
frequent under offload churn. Bisection of the exact free-then-regrow
trigger is still open.

## FINAL ACCEPTANCE (24 windows each, ES off, 2026-08-23 morning)

| arm | median fb | median TPS | vs baseline | peak alloc | max \|dloss\| vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 0.690 s | 11,879 | — | 108.53 GiB | — |
| offload (final: knobs + backward-entry) | 0.837 s | 9,785 | **−17.6%** | 100.45 GiB | 4.8e-6 |
| fullrecompute | 0.773 s | 10,599 | −10.8% | 95.98 GiB | 1.9e-6 |

## VERDICT: within-2% NOT reached; the floor and its mechanism are proven

- Every scheduling lever was tested with a pre-registered prediction; four
  of five were refuted by direct observation (chunking, pacing, stream
  priority, and the scale-artifact hypothesis); backward-entry prefetch
  was confirmed and shipped (~7-11 ms).
- The remaining gap decomposes (traced, itemized): **~64 ms/step** of
  per-MoE-layer dispatcher host syncs colliding with a copy engine that is
  saturated by design (copy volume ≈ forward wall at every sequence
  length, and the engine provably will not interleave or honor priority);
  **~9 ms** residual backward joins; and **episodic allocator-class
  stalls** (open bisection) that add 2-5% to arm medians. Best observed
  offload windows are 0.77 s vs baseline 0.68 s → even the no-stall floor
  is ~+13%.
- **Jack's thesis is right in principle and the blocker is now a specific
  design point, not the offload machinery:** TE GroupedLinear requires
  host-side `m_splits`, forcing a per-layer D2H readback whose latency is
  set by bulk-copy occupancy. Two engineering paths would remove the
  entire ~64 ms class and land within ~1-3% of baseline:
  1. **Device-side grouped-GEMM sizing** (no host counts) — upstream
     TE/mcore feature work.
  2. **SM-driven offload copies** (copy kernels writing to pinned memory,
     leaving the DMA engine free for metadata readbacks) — days-scale
     engineering in the offload module.
- Secondary open item: root-cause the episodic segment-growth stall
  (194 ms `cudaMalloc` ES-off signature / ES map-unmap ES-on signature).

## Standing results table (median TPS, 8k, 10-window arms unless noted)

| arm | median fb | median TPS | note |
|---|---:|---:|---|
| baseline (ES off) | 0.695 s | 11,787 | |
| offload pre-mission (knobs, ES off) | 0.850-0.854 s | 9,587-9,878 | two same-config runs |
| offload + chunk64 | 0.883 s | 9,272 | E2, refuted |
| offload + paced | 0.817 s | 10,021 | E3, within run noise |

Run-to-run spread of identical configs is ~±3% on this proxy; treat
single-arm deltas below that as noise.
