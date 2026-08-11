# Box 3 (wxlg05w) arm log — grothendieck, 2026-08-10 UTC

## Anchors (golden, shared tree, ship+B/F): 131k-d4 = 691 tok/s/GPU (steady 719/706), 16k-d32 = 718 (717s). Canary vs r3anchor golden: w0 +2.4e-3, m0 +1.8e-3, m1 +0.9e-3, m2 +1.4e-3 (curie: accepted-with-caveat, box-3-local refs for verdict numerics).

## W2 arm — REFERENCE boot (READY 11:56, gate OFF) — PRE-DRIVE ARM-CHECK (A2.1)
- [sitecustomize w2box3] megatron_core re-pointed -> mcore_wxlg05w_r3stack (@15d5679eb, byte-copy of the T2-validated branch clone)
- BT_DSA_CP_LAYOUT_CACHE=1 ACTIVE / BT_THD_ROPE_HOST_CACHE=1 ACTIVE (B/F)
- BT_MOE_ROUTING_REPLAY_FORCE=1 (C-prime) ACTIVE
- BT_MOE_PROBS_A2A_COMM=1 (W1) ACTIVE
- BT_MOE_A2A_PIPELINE present but DISABLED (W2 gate OFF, proven)
- BT_MOE_LOOKAHEAD_RECOMPUTE present but DISABLED (W3-v2 bytes in tree, gate OFF, proven)
- Fresh boot, history-symmetric (A3): window 0 = first op on this boot.

## W2 arm — ARM boot (READY 12:39, gate K=2) — PRE-DRIVE ARM-CHECK (A2.1)
- re-point -> mcore_wxlg05w_r3stack (same clone as reference)
- B/F ACTIVE; C-prime ACTIVE; W1 ACTIVE
- BT_MOE_A2A_PIPELINE=2: chunked MoE A2A pipeline ACTIVE + armed (K=2, L=8 experts/group)
- W3 LOOKAHEAD_RECOMPUTE explicitly DISABLED (proven)
- Fresh boot, history-symmetric with the reference arm.

## HYBRID-GATE NOTE (13:45): the W2 reference + arm boots ran FORCE-without-CACHE (BT_MOE_DISPATCH_REPLAY_CACHE never set — same operator omission as box 1). Reference wall (701) carries un-cached replay cost. The arm hang is NOT explained by cache state (reference ran clean with identical flags; the hang is the 8-rank PG 18 collective — fermi owns).
