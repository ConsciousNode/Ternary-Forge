# Ternary Forge: preregistration

*Kham & Kehai Interim, 2026-11-11, written before any real-model result exists.*

## Models and data

- **Models:** RWKV-7 G1d **0.1B** (primary: the hardest case) and **0.4B** (the scaling check). Both are BlinkDL's official float checkpoints.
- **Reference points:** the same models' **Q8_0** and **Q4** GGUF files, measured in the same harness on the same text.
- **Text, four disjoint files:**
  - calibration;
  - validation, for picking settings;
  - test A: the same domain as calibration;
  - test B: a different domain, never seen by any method, to show a win isn't just fitted to the calibration text.

## Metrics

- next-token agreement with the float model (`top1Agree`);
- KL divergence from the float model (`klToFloat`);
- bits per byte;
- effective bits per weight, counting every scale and every float leftover.

## Selection rule

Every method's settings are chosen on **validation only**. Each chosen config is scored **once** on test A and test B. No re-picking after looking at the tests.

## Definitions

**Fully ternary** means every linear and LoRA matrix is ternary. Only the embedding, the head, the norms and the small per-channel vectors may stay float.

## Outcomes

| Outcome | It counts only if ALL of these hold |
|---|---|
| 🏆 **Breakthrough** | A **fully ternary** 0.1B (heal allowed) is at least as good as **Q8_0** on `top1Agree` and `klToFloat`, on **both** test A and test B, at ≤ 2.0 effective bits/weight; **and** the 0.4B does the same. |
| 🎉 **Huge** | A **fully ternary** 0.1B beats **Q4** on `top1Agree` and `klToFloat`, on **both** test sets, at ≤ 2.0 effective bits/weight; **and** the 0.4B also beats its Q4; **and** the gap to float is no larger at 0.4B than at 0.1B (the scaling direction). |
| 👍 **Useful** | Same as Huge, except up to 10% of the linear/LoRA parameters may stay float, with effective bits/weight ≤ 2.5. |
| 📝 **Negative result** | None of the above. We publish which method came closest on each metric, how far each falls short of Q4 and Q8_0, and whether the shortfall shrinks or grows from 0.1B to 0.4B. |

## Ties and noise

- Each eval runs on at least 50,000 test tokens.
- "Beats" means the 95% bootstrap interval (over 1,000-token blocks) of the difference excludes zero in the stated direction.
- "At least as good as" means the interval's upper bound on the shortfall is under 0.5 percentage points for `top1Agree`, and under 5% (relative) for `klToFloat`.

## Not allowed after the fact

- Changing these thresholds.
- Adding a test set.
- Dropping test B.
- Switching which model counts as primary.

If we think a rule was wrong, we say so in the write-up and report the result under **both** the original rule and the proposed one.
