> **RETRACTED 2026-08-20 ~13:05 PDT — the causal claim below is WRONG.**
>
> Real cause: **the ali region is out of B300 capacity.** `kubectl describe` on the
> Pending pods reports `FailedScheduling ... 17 Insufficient nvidia.com/gpu` and
> `GOATScaler NotTriggerScaleUp: missing matching nodepool: 1 NodePool ResourceNotMatch`
> (the autoscaler cannot add nodes). Jobs sit Pending, get torn down, and the
> interactive-session configmap is garbage-collected on teardown — so the
> "configmap not found" FailedMount recorded below is a **downstream symptom of a
> job already dying, not the cause.** 2-node jobs fail more often only because they
> need two 8-GPU nodes at once.
>
> What survives: the observations are accurate, and the ASK is unchanged — job
> status should surface the scheduler's real reason instead of
> `TRAINING_JOB_STOPPED: (no error_message)`. The scheduler had the right answer
> the whole time; it never reached the operator. See papercut correction.
>
> Kept as a record of a wrong inference and how it was caught.

# Multinode worker-pod regression — evidence chain

Collected by banach 2026-08-20 ~12:55 PDT. Org namespace `org-99340d71961343c28c5c567d705ab0c0`, cluster `ali-apse7-prod-1`.

## Symptom
Multinode (2+ node) training jobs die mid-provision. The LEADER pod `<jobid>-multinode-0`
comes up healthy; the WORKER pod `<jobid>-multinode-0-1` never does. The job then reports
`TRAINING_JOB_STOPPED` with an **empty error_message**, which is indistinguishable from
infra flake and invites pointless retries.

## Root cause
The worker pod cannot mount volume `bt-interactive-session` because configmap
`bt-interactive-session-<jobid>` is never created. The LeaderWorkerSet then deletes the
worker every ~30s (`DeletedForFailedMount`).

## Killed our provisions
- `qj0ov2w` (devbox-up 16 b300 ali) — died at the ssh-reachability step, 6 retries rc=255.
- `wn2m92w` (devbox-up 16 b300 ali) — passed steps 4-11 incl. all verification checks, then
  "left RUNNING mid-provision — now TRAINING_JOB_STOPPED: (no error_message)".

## Configmaps present (note the absence)
    bt-interactive-session-31gdxj3         1      24h
    bt-interactive-session-wgomnj3         1      10d
    bt-interactive-session-wn2mz4w         1      103m
    bt-interactive-session-woloenw         1      5h36m
    -> bt-interactive-session-wn2m92w and -qj0ov2w are ABSENT.

## No multinode job in the namespace has a running worker
    baseten-training-job-32lpv0w-multinode-0-1                        2/2     Terminating         0          26h
    baseten-training-job-q8epr8q-multinode-0-1                        0/2     Init:0/5            0          26h
    baseten-training-job-q975593-multinode-0-1                        2/2     Terminating         0          27h
    baseten-training-job-wxg425q-multinode-0-1                        2/2     Terminating         0          25h
    -> last healthy workers were 25-27h ago; q8epr8q stuck Init:0/5 for 26h (another team's job).

## Representative events
    117s        Warning   DeletedForFailedMount   leaderworkerset/baseten-training-job-q8epr8q-multinode                                        LeaderWorkerSet deleted; object baseten-training-job-q8epr8q-multinode-0-1: reason: FailedMount, message: MountVolume.SetUp failed for volume "bt-interactive-session" : configmap "bt-interactive-session-q8epr8q" not found
    87s         Warning   DeletedForFailedMount   leaderworkerset/baseten-training-job-q8epr8q-multinode                                        LeaderWorkerSet deleted; object baseten-training-job-q8epr8q-multinode-0-1: reason: FailedMount, message: MountVolume.SetUp failed for volume "bt-interactive-session" : configmap "bt-interactive-session-q8epr8q" not found
    57s         Warning   DeletedForFailedMount   leaderworkerset/baseten-training-job-q8epr8q-multinode                                        LeaderWorkerSet deleted; object baseten-training-job-q8epr8q-multinode-0-1: reason: FailedMount, message: MountVolume.SetUp failed for volume "bt-interactive-session" : configmap "bt-interactive-session-q8epr8q" not found
    27s         Warning   DeletedForFailedMount   leaderworkerset/baseten-training-job-q8epr8q-multinode                                        LeaderWorkerSet deleted; object baseten-training-job-q8epr8q-multinode-0-1: reason: FailedMount, message: MountVolume.SetUp failed for volume "bt-interactive-session" : configmap "bt-interactive-session-q8epr8q" not found

## Ruled out
- CPFS cache quota (the Aug 2026 38-hour outage signature): no `resourcequota` objects in the
  namespace; failures are mount-level not write-level; a fresh 1TiB bmcpfs PVC bound
  successfully 13 min before the second death.
- Anything we changed: another org's job fails identically and predates our work by ~26h.

## Ask
Platform-side fix for configmap creation on multinode jobs, and — independently —
surface the FailedMount reason in job status instead of an empty error_message.

Papercut: pc_6b94e6a88159 (tags: platform, training-jobs).
