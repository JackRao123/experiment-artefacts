# Stacked-outlook recompute logic — pre-draft for the 05:00 scorecard fill

(fermi, background task per helmholtz. Drop-in replacement for the
SCORECARD_MORNING.md "Stacked outlook" section once the W2 arm / A-v3 arm /
W3-v3 canary verdicts land. Every quantity is either a measured anchor or a
labeled model term; the arithmetic is written out so the fill is mechanical.)

## Constants (do not re-derive at 05:00)

- Base: **46.8 s/step @131k×d4, 715 tok/s/GPU** (frontier at nightfall: golden
  + NCCL env + TF32 head + B/F; C′-on profile per baseline discipline).
- Conversion: 524,288 tok/step ÷ 16 GPUs = **32,768 tok/GPU/step**;
  tok/s/GPU = 32768 / step_seconds. (House fuzz: 32768/46.8 = 700.2 vs the
  recorded 715 — ~2 %, from the rounded step/token figures. Keep the house
  convention: model in STEP SECONDS, convert with 32768, and quote the
  recorded-style tok/s; never mix the two mid-calculation.)
- Two-tier rule: box-1 walls are tail-bound-informational; the prod-fabric
  model column is the ship-magnitude estimate. Both are shown; never quote
  the box column alone.

## Model terms (step seconds @131k×d4; [M]=model, [B]=box-measured)

| term | value | additivity rule |
|---|---|---|
| W1 | −3.4…−5.1 [M prod fabric] / −0.46 [B] | fully additive in all variants (memo §8 arithmetic: W1 keeps full value under W3) |
| W2 v1 | −3.0 [M] standalone | −1.5 [M] when W3-v3 is IN (bwd-phase value subsumed; fwd-phase survives, memo §8) |
| W3-v3 | −8…−12 [M] at 60–80 % capture | IN only on canary PASS; capture % from the frame's T1 read |
| C′ | 0 [B, wait-conserved] … −0…−3 [M compounding] | the compounding band applies only when comm is hidden (W2/W3 IN) |
| A-v3 | 0 (tempered expectation) | any measured win is pure upside — fill from the arm, else 0 |
| option-6 | −1…−1.5 [M] | ONLY in W3-v3-OUT variants (non-additive where W3 captures); unbuilt |
| F2 | 0 @2-node | excluded from the 2-node stack (scale-out enabler; see §4-node section) |

## Variants (base 46.8 s; low/high = optimistic/pessimistic term ends)

| # | stack | step model | tok/s/GPU | vs 715 |
|---|---|---|---|---|
| V0 | proven tonight (B/F∘ + C′ + W1) | 46.8−0.46 = **46.3 [B]**; 46.8−3.4…−5.1 = **41.7–43.4 [M]** | ~708 [B]; 755–786 [M] | +5–10 % [M] |
| V1 | V0 + W2 (arm PASS) | −3.0 more ⇒ **38.7–40.4 [M]** | 811–847 [M] | +13–18 % [M] |
| V2 | V0 + W2 + W3-v3 (canary PASS) | W2 marginal −1.5; W3-v3 −8…−12 ⇒ **27.9–33.9 [M]** | 966–1175 [M] | +35–64 % [M] |
| V3 | V2 + option-6 | option-6 excluded (subsumed) | = V2 | = V2 |
| V4 | V1 + option-6, W3-v3 OUT (canary FAIL world) | −1…−1.5 more ⇒ **37.2–39.4 [M]** | 832–881 [M] | +16–23 % [M] |
| V5 | W2 arm FAIL world: V0 alone | = V0 | 755–786 [M] | +5–10 % [M] |

∘ B/F already inside the 715 frontier anchor.

A-v3 moves any variant by its measured amount (tempered: ~0). C′ compounding
band (−0…−3) is included in the V1/V2/V4 low ends only.

## Fill-in protocol at 05:00 (mechanical)

1. W2 timed arm verdict: PASS → V1 line activates (replace [M] with the
   arm's mechanism read if it revises −3.0); FAIL → V5.
2. W3-v3 canary verdict: PASS → V2 activates with the frame's T1 capture %
   (recompute the W3-v3 term as capture×10.8 s if capture lands outside
   60–80 %); FAIL → V4 (option-6 note: build decision re-opens per the
   stub's disposition).
3. A-v3: fill its measured delta (or 0) and shift the active variant's low
   end accordingly.
4. Strike the inactive variants; keep the table's term rules visible (Jack
   reads the assumptions, not just the numbers).
5. 16k×d32 column: deliberately ABSENT — the lever models are 131k-shaped
   (W3 16k is HARD OFF; W1's 16k share is larger but unmeasured tonight).
   Do not extrapolate; say "16k model needs its own measurements".
