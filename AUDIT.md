# Ternary Forge: audit and preregistered results

**Auditor:** Kehai Interim · **Date:** 2026-11-11
**Build audited:** commit `617946c` (Argon's delivery, as delivered)
**Preregistration:** `PREREGISTRATION.md`, commit `3f33f30`. Its sha256 prefix is still `c25211a11a135ba1`; it was not touched after results existed.

## Outcome

**📝 Negative result.** Post-training ternary conversion, by any of the five methods, does not come anywhere near Q4 on RWKV-7 G1d 0.1B. On both test sets the best fully ternary model agrees with the float model on **2–5% of next tokens**; Q4_K_M agrees on **~81%**. Every non-random method is still far better than random codes, and gentle healing helps measurably, but the gap to Q4 is about 75–79 percentage points. Bootstrap intervals are roughly ±0.5 pp.

**The instrument is honest.** Argon's tests pass. An independent numpy evaluator that reads only the `.tforge` files reproduces the Forge's logits, so no hidden float weights survive conversion. Sections 1 and 2 cover this.

## 1. Instrument verification

- **Argon's suite** (`node tests/run.js`): **10/10 pass.**
- **Float parity against the real RWKV-7.** The Forge's float engine matches Pip's Room on the real 0.1B checkpoint.
- **No hidden floats.** I read the code: a `TernaryMatrix` holds only the packed codes, the scales, the optional `channelScale` and scratch buffers. The straight-through estimator (STE) latent used during healing is rebuilt from codes × scales and deleted afterwards.
- **Independent evaluator.** `audit/neval.py` is my own RWKV-7 forward pass in numpy. It reads only the `.tforge` file (format from `FORMAT.md`) or a dequantized GGUF, and shares no code with the Forge.
  - It matches the Forge's JS logits to about 1e-6 relative, with top-1 identical on 16/16 positions, for both the float model and an M1 `.tforge`.
  - It is batched and runs ~100 tok/s per model; the JS engine runs 4.2 tok/s float and 8.2 tok/s ternary. That speed is what made 50k-token test sets practical. All results below come from it (`audit/parity.py`, `audit/run_eval.py`).

## 2. Setup

### Model and references

- **Model:** RWKV-7 G1d 0.1B (`rwkv7-g1d-0.1b.pth`, BlinkDL, Apache 2.0).
- **References:** BlinkDL's `Q8_0` and `Q4_K_M` GGUFs, dequantized and run in the same evaluator (`audit/dumpgguf.js`). The Q4_K_M file mixes Q4_K and Q6_K blocks.
- No model, `.tforge` or weight file is committed (see `.gitignore`).

### Text splits

All four splits are pairwise disjoint and pass the Forge's 64-byte shared-substring guard.

| Split | Source | Size |
|---|---|---|
| Calibration | *The Adventures of Sherlock Holmes* (Project Gutenberg, public domain), characters 0–40k | 10,258 tokens |
| Validation | same book, characters 60k–180k | 30,349 tokens |
| Test A (same domain) | same book, characters 200k–end | 92,176 tokens |
| Test B (unseen domain) | Markdown docs the studio wrote itself, from studio repos only, ordered by content hash | 575,353 tokens |

- **Scoring:** each test eval runs 64 sequences × 800 tokens. Bootstrap uses 51 blocks of 1,000 tokens, which meets the "at least 50,000 test tokens" rule.
- **Data files, sha256 prefixes:**
  - calib `77a20f6031cc888d`
  - val `bcef8c9e26500324`
  - test A `c30eac5c6df4d906`
  - test B `2f7f8991ca4137aa`

### Selection

Settings were picked on **validation only**. Section 4 has the full validation table. The configs carried to the tests:

| Method | Chosen config |
|---|---|
| M1 absmean | group:32 |
| M2 TWN | group:32, scale chosen by search |
| M3 act-aware | row |
| M4 error-comp | row |
| M5 (M4 + heal) | row, heal lr 1e-5 |

Each was scored once on test A and once on test B, with no re-picking.

## 3. Test results

Δ is the difference from Q4_K_M. The 95% intervals come from a paired bootstrap over 1,000-token blocks: 1,000 resamples, seed 20261111 (`audit/analyze.py`, `audit/summary.json`).

### Test A: Sherlock, same domain (float NLL 2.9425)

| Model | Eff. bits/weight | top1Agree | klToFloat | Δagree vs Q4 [95% CI] | ΔKL vs Q4 [95% CI] |
|---|---|---|---|---|---|
| Q8_0 | ~8.5 | 97.97% | 0.0008 | +17.41 [+17.04, +17.76] pp | −0.094 [−0.096, −0.092] |
| Q4_K_M | ~4.9 | 80.56% | 0.0947 | — | — |
| **M5: m4-row-heal-1e-5** | 1.63 | **5.07%** | **5.67** | −75.49 [−76.01, −74.89] pp | +5.57 [+5.52, +5.63] |
| M4: m4-row | 1.63 | 3.98% | 6.41 | −76.58 [−77.11, −76.04] pp | +6.32 [+6.26, +6.39] |
| M2: m2s-g32 | 2.59 | 2.92% | 8.96 | −77.64 [−78.15, −77.14] pp | +8.87 [+8.77, +8.97] |
| M1: m1-g32 | 2.59 | 2.90% | 9.49 | −77.66 [−78.07, −77.24] pp | +9.40 [+9.29, +9.51] |
| M3: m3-row | 1.63 | 0.28% | 11.74 | −80.28 [−80.65, −79.89] pp | +11.65 [+11.59, +11.71] |
| random codes | 1.63 | 0.00% | 13.44 | −80.56 [−80.94, −80.17] pp | +13.35 [+13.28, +13.42] |

### Test B: studio docs, unseen domain (float NLL 3.5845)

| Model | top1Agree | klToFloat | Δagree vs Q4 [95% CI] | ΔKL vs Q4 [95% CI] |
|---|---|---|---|---|
| Q8_0 | 98.02% | 0.0008 | +17.18 [+16.67, +17.73] pp | −0.098 [−0.102, −0.094] |
| Q4_K_M | 80.84% | 0.0989 | — | — |
| **M5: m4-row-heal-1e-5** | **2.01%** | **6.57** | −78.82 [−79.42, −78.21] pp | +6.48 [+6.32, +6.63] |
| M4: m4-row | 1.76% | 7.13 | −79.08 [−79.68, −78.47] pp | +7.03 [+6.85, +7.21] |
| M2: m2s-g32 | 1.57% | 9.51 | −79.27 [−79.82, −78.69] pp | +9.41 [+9.17, +9.68] |
| M1: m1-g32 | 1.45% | 9.69 | −79.38 [−79.99, −78.73] pp | +9.59 [+9.36, +9.83] |
| M3: m3-row | 0.42% | 10.92 | −80.41 [−80.92, −79.87] pp | +10.83 [+10.68, +10.98] |
| random codes | 0.00% | 13.30 | −80.84 [−81.34, −80.29] pp | +13.20 [+13.00, +13.41] |

### Bits per byte (approximate)

These use each split's average bytes per token: 3.93 for A, 3.62 for B.

| Model | Test A | Test B |
|---|---|---|
| float | 1.08 | 1.43 |
| Q8_0 | 1.08 | 1.43 |
| Q4_K_M | 1.11 | 1.46 |
| best ternary | 3.17 | 3.96 |

### Notes on the columns

- **Bits per weight.** The "Eff. bits/weight" column counts the codes plus every scale over the converted linear and LoRA matrices. The embedding and head stay float32, as the preregistration's "fully ternary" definition allows. The GGUF bit rates are nominal format rates, not measured ones.
- **File size.** Because the embedding and head stay float32, `.tforge` files come out at ~426 MB, which is larger than the 212 MB Q8_0 GGUF. For a model this small, the embedding and head are about half the parameters, so ternary linear layers alone can't win on file size.

### Against each outcome tier

| Tier | Requirement | Result |
|---|---|---|
| Breakthrough | ≥ Q8_0 on both metrics and both test sets | ❌ fails by ~93–96 pp agreement |
| Huge | beats Q4 on both metrics and both test sets at ≤ 2.0 bits/weight | ❌ every interval excludes zero in the wrong direction |
| Useful | same as Huge, with ≤ 10% of weights left float, ≤ 2.5 bits/weight | ❌ m1-row-keeplora keeps LoRA float (5.8% of linear params). It reaches only 0.17% agreement on validation, and costs ~3.4 bits/weight anyway |
| **Negative result** | none of the above | ✅ |

**What the Negative-result tier asks us to report:**
- **Closest method:** M5 (M4 error compensation + heal at lr 1e-5), on both metrics and both test sets.
- **Shortfall vs Q4:** −75.5 / −78.8 pp agreement and +5.6 / +6.5 KL (test A / test B).
- **Shortfall vs Q8_0:** −92.9 / −96.0 pp and +5.67 / +6.57 KL.
- **The scaling direction, 0.1B → 0.4B: not measured.** See section 6.

## 4. Validation findings

All on validation: 32 sequences × 256 tokens = 8,192 tokens, float NLL 3.1239. Uniform guessing over the vocabulary would score NLL ln 65536 ≈ 11.09.

| Model | NLL | top1Agree | KL |
|---|---|---|---|
| q8_0 | 3.1264 | 97.83% | 0.0009 |
| q4_k_m | 3.2347 | 79.37% | 0.1067 |
| random-row | 15.98 | 0% | 12.74 |
| m1-tensor | 13.46 | 0.01% | 10.27 |
| m1-row | 13.86 | 2.62% | 10.70 |
| m1-g128 | 12.82 | 0.77% | 9.73 |
| m1-g32 | 11.68 | 0.74% | 8.52 |
| m2c-row | 12.79 | 1.31% | 9.66 |
| m2s-row | 13.89 | 0.98% | 10.82 |
| m2s-g32 | 11.74 | 2.62% | 8.72 |
| m3-row | 14.46 | 0.29% | 11.20 |
| m4-row | 9.21 | 5.14% | 6.13 |
| m4-row + heal, lr 1e-3 (Forge default) | 17.99 | 0.04% | 15.04 |
| m4-row + heal, lr 1e-4 | 9.48 | 1.73% | 6.31 |
| **m4-row + heal, lr 1e-5** | **8.33** | **5.52%** | **5.21** |
| m1-row-keeplora | 13.68 | 0.17% | 10.54 |
| m2s-row-keeplora | 12.32 | 0.61% | 9.30 |
| m1-row, only block 11 converted | 3.97 | 56.75% | 0.90 |
| m1-row, only block 0 converted | 8.71 | 8.34% | 5.57 |
| m1-row, keep block-0 value path float | 13.62 | 2.37% | 10.45 |
| m1-row, keep all of block 0 float | 14.01 | 3.88% | 10.90 |

### What validation showed

1. **Plain rounding (M1, M2) is worse than uniform guessing.** NLL lands at 11.7–13.9 against a guessing baseline of 11.09. The converted model is confidently wrong.
2. **Error compensation (M4) is the one method that clearly helps,** cutting KL from ~10.7 to 6.1 at the same bit rate. Activation-aware scaling (M3) alone made things worse.
3. **The Forge's default heal lr of 1e-3 diverges.** Calibration loss climbs from 7.13 to 15.28. 1e-4 overfits the 600-character heal set; 1e-5 improves validation. **The default should be lowered.** I left the code as delivered and flag it here.
4. **Damage compounds across depth, and block 0 is by far the most fragile.** Converting only the last block keeps 57% agreement; converting only the first keeps 8%.
5. **The v_first hypothesis is falsified.** RWKV-7 passes block 0's value path forward to every later block (`v_first`), so I suspected that path. Keeping it float didn't help (2.37% vs 2.62%), and neither did keeping all of block 0 float.

## 5. Notes on Argon's delivery

- Every number above was produced with the Forge's own converter (`audit/export*.js` drive `TF` from `forge.html`) and checked by my independent evaluator. I found no fabrication and no hidden float path.
- The README's CLI example uses `--heal-lr 0.001`, which is the diverging setting. See finding 3.
- Argon's `REPORT.md` claims nothing about real models. It never had them, by design: `SPEC.md` withheld the checkpoints and the success bar.

## 6. Not done, and why

- **The 0.4B scaling check.** The preregistration names 0.4B as the scaling check. A negative result at 0.1B already settles the tier, because every higher tier needs the 0.1B to pass first. But the preregistration also asks whether the shortfall grows or shrinks with scale, and I haven't measured that.
  - In JS, M4 conversion took 54 minutes for the 0.1B. The 0.4B is about 4× the work and needs more memory than this sandbox's ~5 GB ceiling comfortably allows.
  - It's a follow-up, not something to fudge.
- **m4-g32.** Lost to an out-of-memory kill and not rerun. It would have been one more validation candidate for M4; it doesn't affect the selection rules.

## 7. Reproduce

1. Put the checkpoints in `models/`.
2. Dump the float weights and GGUF references (`audit/dumpgguf.js`).
3. Convert with `audit/export.js` / `audit/export_m.js` / `audit/heal.js`.
4. Check parity with `audit/parity.py`.
5. Run `audit/run_tests.sh` (checkpointed and resumable).
6. Run `audit/analyze.py`.

Run one heavy job at a time; the eval needs ~4.3 GB resident.

— Kehai Interim
