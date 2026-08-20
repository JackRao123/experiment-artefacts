# Disposable-subagent review — f2407a10 (offload plumbing)

VERDICT: CHANGES REQUIRED for the morning PR; **cleared to boot tonight**
(no runtime-correctness defect; field names, placement, and splat isolation
all verified clean; the L0b probe uses hand-written trainer JSON so finding
4 does not bite tonight).

## Findings (fix before PR, not before boot)

1. control.py:246-251 — MEDIUM, dead code: the fused_group_mlp branch is
   unreachable ("fused_group_mlp" is not in the `allowed` set, so the
   invalid-modules check at :234-239 raises first). Users never see the
   op-fuser explanation. Either add it to `allowed` so the dedicated check
   fires, or delete the branch and fold the explanation into the
   invalid-modules message.
2. test_fused_group_mlp_rejected — MEDIUM: passes via the generic
   invalid-modules message (wrong reason); would keep passing if the branch
   were deleted. Fix together with (1).
3. control.py:186-189, :219-220, :561-563 — MEDIUM, inverted rationale in
   comments: the claim "mcore's checks only run at provider construction —
   before the write" is FALSE here: the bridge provider defers post-init
   and `provider.finalize()` (megatron_config.py:288) re-runs mcore's
   asserts AFTER the write (bridge transformer_config.py:108, :130). The
   trainer validators are still worth having (fail at config-parse time,
   not model-build time) — fix the stated justification so the next person
   isn't misled.
4. types.py:117 — MEDIUM, scope question: the model-registry config path
   (trainer_reconcile.py:53-90) does not carry activation_offload — the
   feature is reachable only via hand-written trainer JSON. If intentional
   (probe-only for now), say so in the commit body.
5. control.py:571 — LOW: fused_group_mlp in the `offending` set is
   unreachable (already rejected upstream). Harmless mirror; note or drop.
6. megatron_config.py:282 — LOW: aliases the live pydantic list onto the
   provider; mcore mutates recompute_modules elsewhere, so `list(...)` is
   cheap insurance.
7. megatron_config.py:281-287 — LOW: when enabled=False an unvalidated
   modules list is still written. Currently harmless; write `[]` when
   disabled.
8. Test gaps — LOW: min_tensor_size<0, delta_bytes<0, extra="forbid"
   rejection, legal attn_proj+core_attn pair; and nothing pins the
   name→value MAPPING (swapping min_offloaded_tensor_size and
   activation_offload_fraction RHS would pass the suite).
9. control.py:209 — NIT: `modules: list[str] = []` vs the file's
   default_factory convention.

## Verified clean (for the PR description)
Five field names exact vs mcore; defaults equal mcore defaults (unset
config leaves provider pristine); writes land before finalize() and nothing
downstream overwrites them; splat safety confirmed; fused_group_mlp
op-fuser claim TRUE (transformer_config.py:1859-1861, flag never set in
trainer); cross-config rule matches mcore :1838-1847 exactly incl. the
unset-modules case; NVTE_CPU_OFFLOAD_V1 docstring accurate (TE>=2.10 path
via cfg.validate()); inapplicable mcore constraints correctly not
re-encoded; no debug leftovers.
