# GLM-5.2 B300 DSA IMA: Overnight Exact-Replay A/B Investigation

Investigate and conclusively fix the GLM-5.2 B300 cuDNN DSA illegal-memory-access incident. Do not assume PR #814, the current `dsatopk1` wheel, or any other existing patch is causal. The task is complete only after the same frozen workload fails without the necessary fix and succeeds with it, and the causal patch has been isolated.

## Non-Negotiable Evidence Standard

The existing production smoke test is insufficient. It sent one synthetic 84,173-token datum to a patched image and passed, but it did not replay the original customer data, model state, request ordering, or queueing behavior.

Do not claim that sequence length alone reproduces this incident. Historical logs show that an 84,173-token request failed once and an identically shaped retry succeeded on the old image. A later request with maximum sequence length 63,956 then crashed the retry. The failure may be data-dependent, state-dependent, stream-order-dependent, or a nondeterministic kernel race.

Completion requires all of the following:

1. Recover or faithfully recapture the original workload and freeze it with SHA-256 hashes.
2. Reproduce the CUDA IMA on the historical baseline code/image.
3. Replay the identical frozen artifact against candidate fixes.
4. Isolate the one patch, or smallest necessary patch set, that changes the outcome.
5. Demonstrate repeated fixed-arm success under the same procedure that fails on baseline.
6. Build and validate the resulting CUDA13/B300 image.
7. Open a minimal source/wheel PR and, only if needed, a narrow GLM image-pin PR.

## Environment Strategy: Dev Box First

Do almost all investigation on one persistent two-node B300 dev environment: 2 nodes with 8 B300 GPUs each, matching the historical trainer allocation. Use `/dev-box-up` in its supported multinode mode and keep the environment alive overnight. Use the connection, checkout, sync, and command helper scripts produced by `/dev-box-up`; do not invent parallel SSH, rsync, port-forward, lifecycle, or bootstrap scripts. Provision enough local disk for the source checkout, compiled kernel caches, captures, logs, and the approximately 756 GB GLM checkpoint.

Do not begin by repeatedly provisioning Loops deployments or changing Billip. Defer the Training Project, image override, and production Loops launches until the causal fix is isolated locally.

Use one checkout and one environment at a time. Start from exact source commit `5a4ae4d`, prepare the repository with the standard Make targets, and run the baseline:

```bash
make server-venv
make sampler-venv
```

These targets are expected to run quickly and are the source of truth for constructing the development environments. Do not hand-roll dependency installation. Record the resulting package versions and hashes. After the baseline reproduces, apply exactly one candidate patch to the same checkout, rerun `make server-venv` and `make sampler-venv`, and replay the identical artifact. Revert to `5a4ae4d` before testing a different patch. Only one candidate environment or built image is needed at a time.

Do not build trainer images during kernel diagnosis. The historical image identifies the baseline source/environment, but the local A/B should iterate directly in the standard venvs. Build one candidate image only after the causal patch is isolated.

The ideal local A/B is:

```text
Same B300 hardware
Same frozen input artifact
Same replay command
Same source base and dependency versions
Same operation ordering and stream behavior

Arm A: 5a4ae4d                         -> reproduces CUDA IMA
Arm B: 5a4ae4d + one isolated patch    -> repeatedly succeeds
```

Start with the full two-node trainer because no reduced reproducer exists yet. After the failing kernel input has been captured and reduced, run the reduced replay on one GPU within the existing environment when possible; do not relaunch a smaller dev box merely to change GPU count.

## Repository and Incident Identity

Local repository:

- `/Users/jackrao/Documents/trainers`

Historical incident:

- User: `q6xkJK0`
- Email: `jack.rao+parsed@baseten.co`
- Organization: `BL15KQ0`
- Team: `qzr5p83`, Parsed
- Loops run: `owp7glw`
- Trainer deployment: `rwn24dw`
- SamplingServer ID supplied with the incident: `6wggeow`
- Model: `zai-org/GLM-5.2-FP8`
- Cluster: `ali-apse7-prod-1`
- Namespace: `org-99340d71961343c28c5c567d705ab0c0`
- Kubeconfig: `~/.kube/ali-apse7-prod-1.yaml`
- Loki datasource: `LOKI-ali-apse7-prod-1`
- Loki datasource UID: `PBC8D421AD474DD61`
- Historical image: `baseten/trainers-server:trainer-cuda13-sm103-5a4ae4d`
- Historical LoRA: rank 32, alpha 32, forward scale 1
- Historical attention backend: flash
- Historical allocation: two B300 nodes with 8 GPUs per node
- Historical tensor parallelism: TP1
- Historical pipeline parallelism: PP1

Resolve and record the actual EP, CP, ETP, DP, world size, seed, trainer config, and package versions before treating the topology as exact. Do not rely on remembered values.

Production log query starting point:

```logql
{namespace="org-99340d71961343c28c5c567d705ab0c0", app=~"baseten-trainer-rwn24dw.*", container="trainer-container"}
```

## Current Candidate Fixes and PRs

Historical baseline:

- `baseten/trainers-server:trainer-cuda13-sm103-5a4ae4d`

Top-k OOB candidate:

- `baseten/trainers-server:trainer-cuda13-sm103-98cd395`
- Contains `nvidia-cudnn-frontend==1.26.0+dsatopk1`
- Carries the #814 indexer top-k OOB patch.

Release/config descendant:

- `baseten/trainers-server:trainer-cuda13-sm103-0e0b65a`
- `0e0b65a` is a config-only commit on top of `98cd395`; its trainer runtime code and cuDNN wheel are intended to be unchanged.

Open source/wheel candidate:

- Trainers PR #821
- Contains the cudnn-frontend#396 TMEM WAR race backport.
- Also contains empty-top-k-row and backward index-bounds fixes.
- Do not use its combined wheel as the first diagnostic arm. Test candidate patches independently.

Current unproven config PR:

- https://github.com/basetenlabs/trainers/pull/829
- It pins only GLM-5.2 B300 to the `0e0b65a` image.
- Keep it marked draft until the A/B proves that `dsatopk1` is causal.
- Do not merge it from the existing smoke evidence.
- Update it only if the `dsatopk1` image is proven causal.
- Close or supersede it if another fix is required.

Relevant upstream issues/PRs:

- cudnn-frontend#396: hd576 backward TMEM WAR/barrier race
- cudnn-frontend#410: top-k OOB-lane fix, still open during the original audit
- cudnn-frontend#439: empty top-k row fix

## Historical Operation Timeline

### First Failing Attempt on the Old Image

Operation `6d8548d7addc412ebdbcbce32313b68a`:

- Submitted at `2026-07-28T21:35:20.258546Z`
- 11 datums
- 508,992 total tokens
- Sequence min/max: 28,355/70,912
- Completed successfully in 51.16 seconds

Operation `08cee77a089d480a861b18c8343dbadd`:

- Submitted at `2026-07-28T21:35:21.373630Z`
- 26 datums
- 1,248,558 total tokens
- Sequence min/max: 33,621/68,011
- Completed successfully in 122.01 seconds

Operation `a369ffe9f665445b94439d340a164aa8`:

- Submitted at `2026-07-28T21:35:22.616617Z`
- 39 datums
- 1,329,907 total tokens
- Sequence min/max: 18,129/84,173
- Queued behind the first two requests
- Began after `08ce...` completed at approximately `21:38:13Z`
- Never completed
- CUDA IMA surfaced at `2026-07-28T21:38:37.989644Z`
- Rank 3 surfaced an asynchronous `torch.AcceleratorError`
- The visible stack reached a later RoPE `.tolist()` synchronization point; this is not necessarily the faulting kernel

### Retry on the Same Old Image

Operation `99fb4aa7824d42ea9516f14aa811d76a`:

- Same shape as `6d85...`
- Completed successfully in 131.10 seconds

Operation `bd1abd6f9cdb47e0a8247b4897e1290c`:

- Same shape as `08ce...`
- Completed successfully in 135.91 seconds

Operation `1357625853834f7c9744b80426233572`:

- Same 39-datum, 1,329,907-token, max-84,173 shape as `a369...`
- Completed successfully in 144.75 seconds
- Successfully retrieved with HTTP 200
- This disproves the claim that the 84,173-token shape deterministically triggers the incident

After those requests:

- Optimizer step 1 completed successfully
- Learning rate was `1e-05`
- `train_mean_nll=0.1161306576102976`
- 3,087,457 accumulated tokens
- 448,460 loss tokens
- Checkpoint `glm52-bolt-mt-async-5`, version 1, was written

Operation `339c67dde6aa4f7699daf46ab97a5c66`:

- Submitted at `2026-07-28T22:00:03.482924Z`
- 26 datums
- 843,999 total tokens
- Sequence min/max: 17,630/63,956
- NCCL watchdog reported CUDA IMA at `2026-07-28T22:00:36.262211Z`
- The operation never completed

The second crash is associated with the later `339c...` request, not the successfully completed 84,173-token retry. Preserve both request sequences when recovering the original workload.

## Objective 1: Recover an Exact Replay

The original researcher did not provide their client repro or payloads. Spend a bounded initial effort looking for accessible private artifacts, but do not block the overnight investigation indefinitely waiting for data that is not available. The primary path is to run a faithful workload on the historical baseline until it fails, capture that failing request and its lower-level kernel inputs, and freeze the captured failure for the A/B.

Recover the original `Datum` payloads if they are accessible, or faithfully recapture a new failing baseline workload containing:

- `model_input` token IDs
- Every `loss_fn_inputs` tensor
- Tensor names, dtypes, and shapes
- Datum ordering
- Request ordering
- Asynchronous submission and queueing behavior
- LoRA rank, alpha, adapter state, and checkpoint
- Optimizer state and step
- RNG seeds and client-side sampling/shuffling state
- SDK and client source revisions

If the original artifacts are found, recover at minimum the first three-request sequence ending in `a369...`; prefer the retry sequence through `339c...`, including optimizer and checkpoint operations between requests. If they are not found, use those sequences as the target shape/order for recapturing a new baseline failure.

The server logs do not contain request bodies. First search available private artifacts by run/deployment IDs, operation IDs, checkpoint names, token totals, and the `glm52-bolt-mt-async-*` prefix. If the researcher's source is not present, recreate the known request shapes, ordering, pipelining, rank, optimizer sequence, and model state on the old baseline and stress it until the same class of CUDA IMA occurs.

If the original client can be rerun, add a temporary client-side capture wrapper around `forward_backward` that serializes each `Datum` before submission. Capture before network serialization so the replay is independent of server availability.

Raw token IDs and loss tensors may be captured because this is an internal researcher workload. Store them only in a private dev-box or private S3 experiment-artifact location. Do not commit token payloads, print them in logs, attach them to a PR, or place them in a public artifact.

Create a manifest containing:

- SHA-256 of every captured request artifact
- Ordered operation list
- Datum counts
- Total token counts
- Every sequence length
- Tensor names, dtypes, and shapes
- Model and LoRA configuration
- Checkpoint and optimizer state
- Source commit
- Client SDK version
- Capture timestamp and provenance

If the exact historical payload cannot be recovered, a newly captured workload is acceptable only after it reproduces the same CUDA IMA and the investigation confirms the same lower-level mechanism. Freeze the first useful failing request and kernel inputs immediately; from that point onward, both A and B must consume the identical captured artifact.

## Objective 2: Reproduce on the Dev Box Baseline

Begin without `CUDA_LAUNCH_BLOCKING`; synchronization can mask a stream or barrier race.

Replay the frozen artifacts against the `5a4ae4d` baseline environment without modification and in their original order. Preserve request pipelining and queue depth if the original client submitted requests asynchronously.

Record for every attempt:

- Git commit
- Container/image digest or environment manifest
- cuDNN-frontend version
- Torch and CUDA versions
- GPU model and driver
- Kernel cache location and whether it was cold or warm
- Artifact SHA-256
- Operation ordering
- Pass/failure result
- First rank showing an error
- First observed synchronization point
- Any pod/process restart or NCCL fallout

Because the incident may be a race, run multiple cycles and cold starts. Do not proceed to the fixed arm until the baseline has produced at least two failures attributable to the same lower-level mechanism. Once a failing request or reduced kernel artifact is frozen, confirm that baseline can fail on that exact artifact rather than continually changing inputs.

If the baseline does not reproduce, investigate:

- Exact adapter and optimizer checkpoint
- Preceding forward/backward operations
- Rank 32 versus rank 16
- Request pipelining and queue depth
- Backward versus forward phase
- RNG state
- Kernel JIT/cache state
- Stream scheduling
- Warm versus cold process
- `optim_step`, `save_state`, and weight-sync ordering
- Full two-node CP/EP topology

Do not proceed to a claimed fix until Arm A fails.

## Objective 3: Pinpoint the Faulting Kernel

After reproducing naturally, add narrowly scoped capture instrumentation. Capture exact inputs immediately before:

- DSA indexer top-k
- Sparse attention forward
- Sparse attention backward
- `reduce_dKV`

Relevant repository artifacts:

- `server/scripts/repro_cudnn_dsa_indexer_topk_oob.py`
- `experiment_artefacts/glm/dsa_topk_ima/scripts/capture_topk_crash.py`
- `server/patches/cudnn-frontend-1.26.0-dsa-indexer-topk-oob.patch`
- Trainers PR #821 and its three backward patches

Use `CUDA_LAUNCH_BLOCKING=1` or explicit synchronization only in secondary diagnostic runs. Record whether synchronization changes reproducibility. Compute Sanitizer was previously found not to support B300; verify current support before spending time on it.

Test hypotheses independently:

1. #814 top-k OOB only
2. cudnn-frontend#396 TMEM WAR/barrier race only
3. Empty-top-k-row fix only
4. Backward index-bounds hardening only
5. Any newly discovered issue

Test candidate patches sequentially in the same checkout. For each patch, reset to `5a4ae4d`, apply only that patch, rerun `make server-venv` and `make sampler-venv`, and execute the frozen replay. Preserve the logs and environment manifest before resetting for the next patch. Do not first test a wheel combining all fixes. A combined wheel can be used only after individual arms establish the necessary patch set.

Reduce the captured failure to a standalone GPU reproducer. The reduced reproducer should consume a frozen tensor artifact, print environment and artifact hashes, fail on baseline, and pass after the isolated fix.

## Objective 4: Run the Paired Dev-Box A/B

Freeze the replay artifact and SHA-256 before starting the formal A/B.

Arm A requirements:

- Baseline `5a4ae4d`
- No candidate fix
- Diagnostic capture is allowed only if it does not alter reproduction
- Must reproduce the same class of CUDA IMA

Arm B requirements:

- Same hardware
- Same baseline source
- Same dependency versions except the isolated patch/wheel
- Same artifact hash
- Same operation order
- Same process/cache conditions
- Must complete with finite, structurally valid outputs

After at least two baseline failures, require ten consecutive fixed replay passes using the identical frozen artifact and procedure. Include cold-process/cache runs. “Did not crash once” is not sufficient.

Produce an A/B table containing:

- Source commit
- Patch set
- Environment/image digest
- cuDNN-frontend version
- Artifact hash
- Attempts
- Passes
- CUDA IMA failures
- First failing rank and kernel
- Numerical result checks
- Timing

## Objective 5: Build the Minimal Fixed Image

Only after the dev-box A/B isolates the cause:

1. Create the smallest source/wheel change needed.
2. Add a focused regression using the reduced non-customer kernel artifact or a safely synthetic equivalent proven to trigger the same mechanism.
3. Update the vendored wheel, `uv.lock`, and wheel documentation only as required.
4. Build a CUDA13/B300 trainer image using `build-sampler-trainer-images`.
5. Verify the built image contains the expected commit, exact wheel version, and patch set.
6. Run the same frozen reduced reproducer inside the built image before production deployment.

Do not add unrelated refactors, broad B300 pin changes, speculative compatibility code, or multiple unproven fixes.

## Objective 6: Final Fixed-Image Production Validation

Create a new dedicated Training Project only after the local cause is isolated and the candidate image exists.

Suggested project name:

```text
glm52-dsa-fixed-image-validation-<UTC-DATE>
```

Record the generated project ID in the final manifest. The causal A/B has already happened on the dev environment, so production needs only to validate packaging and integration of the fixed image.

Do not reuse existing project `4q9klxw`. It already exists as `glm52-dsa-ab-verification` and contains stopped B200 job `qzp998w`, so it is not a clean experiment container.

The agent is authorized to create this project, temporarily apply the narrowest user/project-scoped image override, launch the fixed-image validation, deactivate its owned resources, and restore the exact prior override value without waiting for additional approval. Do not alter a shared global image pin.

Run the newly built fixed image with the same frozen workload used by the local A/B. It must complete without CUDA IMA. Verify finite losses/results, no Xid, no restart, no NCCL watchdog failure, and expected sampler health.

Verify actual pod image/digest with Kubernetes or `kube_pod_container_info`; do not trust only control-plane configuration. Verify the installed cuDNN-frontend version from inside rank 0.

## Objective 7: Ship the Minimal PR

If #814-only is proven causal:

- Keep PR #829 GLM-scoped and minimal.
- Update its description with exact dev-box and production A/B evidence.

If another cudnn-frontend patch is required:

- Open the minimal source/wheel PR containing only the proven fix.
- Build and validate its CUDA13/B300 image.
- Open or update a separate narrow GLM image-pin PR only if configuration must change.
- Close or supersede PR #829 if `dsatopk1` is insufficient.

The final PR description must include:

- Baseline failure command and artifact hash
- Fixed success command and same artifact hash
- Attempt counts
- Exact images and digests
- Root cause at the relevant kernel source line
- Regression test
- Production run/deployment IDs

## Cleanup and Deliverables

Deactivate only deployments created by this investigation. Restore temporary Billip/user/project overrides exactly. Preserve the dev box until artifacts and logs have been copied to durable storage, then stop it.

Leave an audit directory containing:

- Exact replay scripts
- Private artifact locations and SHA-256 hashes
- Environment manifests
- Baseline logs
- Candidate logs
- Reduced kernel reproducer
- A/B result table
- Root-cause analysis
- Image build links
- PR links
- Production run/deployment IDs
- Cleanup confirmation

Do not declare success merely because time expired, an image built, or one fixed run passed. Completion requires a frozen replay, a failing baseline, a repeatedly passing fixed arm, an isolated causal patch, and a minimal PR with inspectable evidence.
