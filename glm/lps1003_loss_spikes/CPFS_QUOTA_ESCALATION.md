# CPFS file-count quota — escalation draft

**Post to:** `#internal-alicloud-setup` (C0BEXCMJTQR) — tag `@Justin` (Justin Schmitt).
**Alternate:** `#ask-sre` → Dinesh Mudrakola (SRE oncall as of 2026-08-01 18:00 PDT).
**Also relevant:** John Thorpe (filed the 07-23 capacity resize on this fileset), Zhang Lu (provisioned the `alicloud-bmcpfs` storage class, Asia TZ).

---

## Message (copy-paste)

Heads up — the Parsed org's CPFS training-cache fileset on `ali-apse7-prod-1` has hit its **file-count** quota, and it's been blocking all training jobs and devboxes org-wide since ~16:30 PDT Sat Aug 1.

**It's inodes, not capacity.** Capacity is fine — 4.6T used of 200T. Three probes inside the fileset:

| probe | consumes | result |
|---|---|---|
| append 64 MB of real data to an existing file | blocks, no inode | **OK** |
| create a 0-byte **new** file | one inode | **`EDQUOT` (Disk quota exceeded)** |
| hard-link an existing file | dir entry, no inode | **OK** |

Creating an empty file fails while writing 64 MB succeeds — so the binding limit is the file count, not bytes.

**Measured usage: 1,013,270 files across 192 project dirs** — i.e. sitting on the CPFS default 1,000,000 file-quantity limit.

- Filesystem: `bmcpfs-3800206s6bsbnaa52jwuc`
- Fileset: `fset-38fdb0874929aac4`
- PVC: `org-99340d71961343c28c5c567d705ab0c0-training-provider-sfs-pvc`

**User-visible symptom:** `truss train` jobs and workstations/devboxes fail almost immediately as `TRAINING_JOB_FAILED` with **no error message** — the bootstrap can't create `.baseten-internal/node-0` on the shared cache. Per bot telemetry in `#training-events-internal` this hit ~10 jobs tonight across **Charles**, **Xiaohan** and **me**, independently. Loops/RL jobs are *unaffected* because they don't mount the cache PVC — so the cluster looks healthy at a glance and nothing alerted.

**@Justin — could you flip the file quantity limit to unlimited on `fset-38fdb0874929aac4`**, the same change you made on the other fileset on 07-21 ("the file quantity limit defaults to 1M. Going to change it to unlimited")? The 07-23 resize on this fileset raised capacity 250Gi → 20Ti but, as far as I can tell, left the file cap at the 1M default.

**Not a page — this can wait for normal hours.** I bought temporary headroom by deleting 13,242 `.pyc` files from my own project tree (`__pycache__` only, regenerates on import), which restored writes immediately and let a devbox provision succeed. But that's a band-aid on a shared pool: one `uv sync` venv build is ~100k files, so the org can re-exhaust the cap at any moment and everyone gets the silent-failure symptom again.

**Separately, two stranded nodes:** `e02-sg-e1n4vn65z0k` and `e02-sg-e1n4vn65z0r` have been `NotReady` since ~2026-08-01 23:37 UTC after I rebooted them to clear driver-orphaned GPU contexts (`z0k` reports `NvidiaPersistencedOffline`). They haven't come back on their own — 16 B300 GPUs of stranded capacity if someone can look at them via the Alibaba console.

Two notes for afterwards:

1. **This will recur.** Since the Alibaba CSI driver can't resize filesets, every CSI-created fileset ships with the 1M default. The other three org filesets on this cluster are still on the default config and will hit the same wall as they fill up. Worth a sweep + a default at creation time.
2. **Alerting gap.** A full inode quota produces `TRAINING_JOB_FAILED` with an empty `error_message`, so users can't self-diagnose and nothing pages. A cheap probe (create + delete a file in each fileset, alert on `EDQUOT`) would have caught this hours earlier.

*Cleanup note:* while diagnosing I patched the PVC request 250Gi → 10Ti to try to force a resize. It's stuck on `ExternalExpanding` ("waiting for an external controller") because there's no bmcpfs provisioner/resizer in-cluster — only the `bmcpfs-csi-node` DaemonSet; `csi-provisioner` handles disk/nas/oss only. Feel free to revert it; it had no effect and capacity was never the constraint.

---

## Backing detail (for questions, not for the post)

**Timeline.** Last successful write to the fileset: `2026-08-01 23:32:06 UTC` (16:32 PDT). First observed failure: devbox job `qzpj0kw`, `2026-08-02 00:32 UTC` (17:32 PDT). Enforcement began inside that one-hour window. The 1M cap has existed since fileset creation; the org simply grew into it.

**Where the 1,013,270 files are** (top consumers, `find | wc -l` per top-level dir):

| files | dir | note |
|---:|---|---|
| 459,339 | `team_qzr5p83` | shared team dir — 45% of the org total |
| 193,452 | `2qj2rj3` | kl-exp* experiment trees |
| 191,721 | `dq47r1q` | my project — 185,696 of it is `trainers_main` (server venv 101,097 + sampler venv 75,798) |
| 83,000 | `lqzj5xw` | |
| 56,155 | `7qkk4lq` | |
| ~30,000 | ~187 other project dirs | most hold only 4–5 files |

The pattern is Python venvs and git checkouts on shared storage: one `uv sync` venv is ~100k files, so a handful of devboxes exhausts a 1M cap regardless of how little disk they use. My own tree alone holds 13,242 `.pyc` files in 2,072 `__pycache__` dirs.

**Why deletion is a poor remedy.** You can't overshoot an inode quota — writes stop exactly at the limit — so freeing N files buys exactly N slots, shared across everyone who's blocked and retrying. Since a single venv rebuild consumes ~100k slots, any cleanup small enough to be safe is also small enough to evaporate immediately. Raising the cap is the only durable fix.
