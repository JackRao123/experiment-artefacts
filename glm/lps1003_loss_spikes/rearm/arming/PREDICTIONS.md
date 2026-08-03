# LPS-1003 arming-condition bisection — predictions (written BEFORE runs)

Date: 2026-08-02. Substrate: devbox q8x5ky3 node0 (B300, rescheduled node
b300-1-z7sxjdpi-0020, all GPUs idle), single GPU, single process.
Wheel: UNFIXED +dsatopk1 content via PYTHONPATH shadow
(sha256 `_interface.py` = 5d0e429e…, no `_get_kernel_stream`).
Geometry: the PR #875 CI-repro case — out = [8192, 73728] fp32 = 2.25 GiB
> 2^31 B → fill_ splits at row 4096; F2 = rows [4096, 8192).
Env baseline: CUDA_DEVICE_MAX_CONNECTIONS unset,
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, CUDA_MODULE_LOADING
default (LAZY).

Detector: per-exec full scan — fully-erased rows (isneginf count == 73728
where reference expects finite cols) + partial rows + bitwise double-exec
compare. Both execs of each rep scanned independently (the CI test only
checked disagreement; knowing WHICH exec fired is itself arming evidence:
exec2 also fresh-maps its out block after empty_cache, since exec1's out is
still held).

| run | lever / env | prediction | why |
|---|---|---|---|
| ec | empty_cache before each rep | fires ≥5/8 on exec1 | CI test 6/6 at this geometry on B300 |
| warm | no lever | 0/8 | rep1 healing, 0/20 plain reps on trainer |
| ec_premap | empty_cache, then map 2 same-size dummies (fill+free) so both execs reuse warm blocks | **clean 0/8 if** arming = driver mem-ops feeding the call's own alloc; fires if arming = residual state from recent unmap/map | sharpest split of candidate A |
| ec_prealloc_out | out= passed in, allocated once at boot; empty_cache ≈ no-op | 0/8 | if this fires, arming isn't allocator-driven at all |
| ec_map_elsewhere | out prealloc'd; lever = map+unmap a big UNRELATED buffer | fires ⇒ mapping churn anywhere arms (global channel/invalidate state); clean ⇒ per-buffer (out's own fresh pages) | decisive A-refinement |
| noexp + ec | expandable_segments:False | fires | armH8 boot fired without expandable; fresh cudaMalloc should arm the same |
| eager + ec | CUDA_MODULE_LOADING=EAGER | fires | ec re-armed a fully-warm trainer 14/14 — modules were long loaded |
| warmup_small boot | sub-threshold call (single-launch fill, same compile_key → same cubins, same tvm-ffi path) first, then big rep0 | big rep0 STILL fires | same reasoning: warm modules/tvm-ffi don't protect the warm trainer |
| conn1 + ec | CUDA_DEVICE_MAX_CONNECTIONS=1 | 0/5 | established masking |

Decision rules (pre-committed):
- ec fires + ec_premap clean + ec_map_elsewhere clean → arming = fresh
  physical mapping OF THE OUT BUFFER consumed inside the call window;
  candidate A confirmed in its narrow per-buffer form.
- ec_premap clean + ec_map_elsewhere fires → arming = global post-remap
  driver state (channel invalidation), not per-buffer.
- ec_premap fires → alloc-time mapping is NOT the mechanism; suspect the
  empty_cache unmap side; then add run ec_unmap_only (alloc/free dummy,
  empty_cache, prealloc'd out).
- warmup_small boot clean while plain boot fires → module-load or tvm-ffi
  first-call IS a separate boot trigger (candidates B/D live); eager run
  then splits B from D.
- If exec2 fires at comparable rate to exec1 under ec → "fresh mapping of
  the written buffer" is sufficient regardless of position in the window.
  If exec2 never fires → only the FIRST post-empty_cache call is armed;
  suspect the unmap/free ops (they precede exec1 only).

Boot-rep0 note: every fresh process contributes one boot observation
(recorded process_fresh=true). armH8 evidence says boot has a trigger
besides page mapping; the lab's noexp boot observations test whether that
survives standalone.

---

## REVISION 1 (2026-08-02, after runs `ec`, `conn1`) — written BEFORE the
## remaining matrix

Observed (unpredicted): in the standalone lab the race fires on EVERY exec
after the first — warm or cold, lever or no lever (`ec` run: rep0 exec1
clean 0 erased/0 partial; all 15 subsequent execs erased=1824 rows
[6368,8191] + partial fringe ~1180). conn=1: 4/4 fully clean (same race,
masked). The trainer pattern (cold-only) is NOT reproduced by allocator
state; the lab syncs before every exec.

Revised hypothesis (idle-queue arming): the tvm-ffi raw launch has NO
ordering vs torch's F2 whenever the trio is submitted to an IDLE stream-0
channel. Protection comes from either (a) pending predecessor work in the
channel at submission time, or (b) first-ever-launch module-load
serialization (lab exec1). On the trainer, warm steady-state keeps stream
0 hundreds of kernels deep at indexer_fwd time → protected; cold windows
(boot, empty_cache, rebuild) are exactly the moments the queue DRAINS
(synchronous alloc/unmap driver ops) → armed. Allocator activity is a
correlate, not the mechanism.

New predictions (pre-committed):
| run | prediction under idle-queue | prediction under alloc/mapping |
|---|---|---|
| warm (no lever, cached blocks, synced between reps) | FIRES all execs (queue drained by sync) | clean 0/8 |
| busy (enqueue ~50ms matmuls on stream 0, no sync, then call) | CLEAN | fires (out still cached/warm either way) |
| busy_ec (junk enqueued, then empty_cache, then call) | CLEAN (queue trumps lever) | fires (fresh mapping) |
| ec_premap | fires (queue drained) | clean |
| ec_prealloc | fires | clean |
| ec_map_elsewhere | fires | clean-or-fires per its sub-case |
| eager + ec (CUDA_MODULE_LOADING=EAGER) | boot exec1 NOW FIRES TOO (lazy-load serialization was exec1's only protection) | exec1 clean, rest fire |
| warmup_small + ec | big exec1 FIRES (cubins already loaded by small call) | exec1 clean |

Decision rule: busy/busy_ec clean + warm fires + eager kills exec1's
immunity → arming condition = idle hardware channel at submission, cold
starts only correlate. The remaining "why does an idle channel invert two
same-stream launches across the two launch APIs" is the NVIDIA-boundary
question; package with witness traces.

---

## SCORECARD (updated as results land; see results/*.jsonl)

REVISION 1 was itself superseded mid-session by the witness traces + the
ExternalStream probe: there is no same-stream inversion at all — the
unfixed wheel's `_torch_stream_context(ExternalStream(int(handle)))` puts
the fills on a FRESH TORCH POOL STREAM every call (torch treats
stream_ptr=0 as absent → getStreamFromPool; 32 handles cycling), while S
goes to the legacy stream. No ordering ever exists; manifestation is pure
timing (does S start before F2 finishes?).

- warm: FIRED 15/16 execs (only process-first exec clean) — REVISION 1 ✓,
  original alloc-hypothesis ✗
- ec / ec_premap / ec_prealloc / ec_map_elsewhere: all fired (allocator
  state irrelevant) — REVISION 1 ✓
- conn1: 0/8 clean ✓ (single connection restores submission-order execution)
- fixed wheel: 0/8 clean ✓
- eager + ec: rep0 exec1 STILL clean — REVISION 1 sub-prediction ✗:
  first-exec protection is NOT (only) lazy cubin load; boot trace shows
  ~868 µs host gap between F2 enqueue and S submission on first call
  (tvm-ffi/cute first-call setup dominates). Protection detail only.
- wsmall + ec: big exec1 still clean (rep0 AND rep1), fired from rep2 —
  same conclusion as eager; first-call host latency not consumed by the
  small call. ✗ on the sub-prediction, mechanism unaffected.
- busy / busy_ec: pending — prediction CLEAN (pending default-stream work
  delays S past F2; reproduces trainer-warm immunity).
