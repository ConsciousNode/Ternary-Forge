# Ternary Forge: a post-training ternary converter and measurement bench for RWKV-7

**ConsciousNode SoftWorks.** Xinu rules: single-file HTML, zero dependencies, offline, no network APIs, deterministic. Inline everything and label every inline: any code lifted from an attached file gets a provenance comment (`// from pips-room.html:LINE-LINE`).

**Attached:** `pips-room.html` (the studio's RWKV-7 engine). It contains:
- the float forward pass;
- loaders for `.pth`, `.safetensors` and `.gguf`;
- the RWKV world tokenizer;
- the `TMAC` / `BitLinear` ternary kernels.

Lift from it, and don't rewrite what already works.

## What to build

An instrument that takes a trained **float** RWKV-7 checkpoint, converts its weights to **ternary** (each weight one of {−1, 0, +1} times a scale) by any of several methods, and measures the result against the float model on held-out text. You build the instrument; we run the experiments. **Do not predict, summarize or rank results anywhere in the code, docs or UI.**

### Deliverables

1. `forge.html`, the browser app:
   - load a checkpoint;
   - choose a method and its settings;
   - convert;
   - evaluate;
   - export the ternary model;
   - load an exported model back and chat with it, using Pip's Room's sampler.
2. Node CLIs:
   - `convert.js` (checkpoint → ternary model file);
   - `eval.js` (model file + text → metrics JSON);
   - `run-batch.js` (a grid of methods × settings → `results.jsonl`);
   - `tables.js` (`results.jsonl` → Markdown tables, every number computed, none typed).
3. `tests/run.js`, the test suite below.
4. `README.md`, `METHODS.md` (every method with its equations and every default with a one-line reason), and `FORMAT.md` (the ternary file format, byte by byte).

## Methods (implement all; each is a switch)

Every method produces, for each converted matrix, a ternary code matrix plus scales. Shared settings:
- **`scope`:** which tensors get converted. Default: all linear and LoRA matrices. Embedding, head, LayerNorm/GroupNorm parameters and the small per-channel vectors stay float unless `scope` says otherwise. A `keep` list may name more tensors to leave float.
- **`granularity`:** `tensor` | `row` | `group:G` (G ∈ {32, 64, 128}), i.e. one scale per tensor, per output row, or per group of G consecutive input weights.

The methods:
- **M1 `absmean`:** BitNet b1.58 rounding. Scale = mean |w| over the scale unit; code = round-clip(w / scale, −1, 1).
- **M2 `twn`:** threshold-then-fit. Pick threshold Δ per scale unit, then the scale as the mean |w| of the weights above it. Either the closed-form approximation Δ = 0.7·mean|w|, or `search`: a grid over Δ minimizing ‖w − ŵ‖².
- **M3 `act-aware`:** before ternarizing, rescale input channels by calibration-activation statistics (AWQ-style). Fold the inverse scale exactly into the preceding operation where the math allows. Where it doesn't, keep it as an explicit float per-channel vector, and count it (see bits/weight).
- **M4 `error-comp`:** GPTQ/OBQ-style sequential rounding using a Hessian built from calibration activations. Round one input column at a time to ternary and push its error onto the not-yet-rounded columns. Settings: block size and damping.
- **M5 `heal` (optional stage after any of M1–M4):** a short distillation fine-tune.
  - **Trains:** the scales (and optionally the float vectors left in `scope`).
  - **Target:** KL to the float model's next-token distribution, on calibration text only.
  - **Codes:** frozen, or updated with a straight-through estimator (a switch).
  - **Steps:** a setting, default small.
  - **BPTT:** exact, lifted from the attached engine. Verify it with a finite-difference check.

## Splits and data (no leaks)

Text comes from files we pass in.
- `--calib`: used **only** by M3, M4 and M5.
- `--val`: used **only** for choosing settings.
- `--test`: used **only** for final metrics.

The CLIs record the hash of each file in every result record. `run-batch.js` refuses to run if any two of the three overlap: test for any shared 64-byte substring, and fail.

## Metrics (every eval record)

- `nll` (nats/token) and `bpb` (bits per byte) on the given text;
- `top1Agree`: the fraction of positions where the ternary model's top token equals the float model's;
- `klToFloat`: mean KL(float ‖ ternary), per token;
- `bitsPerWeight`: the effective bits per converted weight, counting codes at log2(3) plus **every** scale, float vector and fold-in parameter the method adds;
- `fileBytes`: the size of the exported model;
- `tokensPerSec`: decode speed on the TMAC path, measured separately;
- the full config, method settings, input hashes and code hash.

**References in every batch:**
- the float model itself (`method: none`);
- a **random-code control:** the same scales as M1, but codes drawn uniformly at random with a fixed seed, as a floor;
- when a `.gguf` of the same model is supplied, the Q8_0 / Q4 file evaluated in the same harness.

## Forbidden

- **No hidden float weights.** No converted matrix may keep float weights anywhere: not in caches, not in "residual" side tensors, not in fallback paths. The exported file holds codes, scales and only the float tensors `scope`/`keep` declare, and `bitsPerWeight` counts them all.
- **No touching other splits.** No method may read `--val` or `--test` text, or their statistics.
- **No special cases.** No branching on the model name, size or file hash; no constants tuned to a specific checkpoint.
- **No winner.** Never select, rank or hide methods in output. Every configured run lands in `results.jsonl`.

## Tests (`node tests/run.js`, all must pass)

1. **Static scan:** no network or storage APIs; strict CSP in `forge.html`.
2. **Float parity:** on a small randomly initialized RWKV-7 (built by the test), the forge's float forward pass matches the attached engine's logits bit for bit.
3. **Ternary invariant:** decode every converted tensor of an exported file and assert each code is in {−1, 0, +1}. Recompute `bitsPerWeight` independently from the file and match the record.
4. **Round trip:** export → import gives logits identical to the in-memory converted model.
5. **Determinism:** the same inputs and seed give byte-identical exported files and identical metrics, run twice.
6. **Method correctness on toy matrices:**
   - M1 and M2 match hand-computed results on a fixed 4×8 matrix;
   - M2 `search` never has higher reconstruction error than its closed form;
   - M4 has lower or equal output-reconstruction error than M1 on a matrix with correlated synthetic inputs, measured on held-out synthetic inputs.
7. **Heal gradient check:** for M5, the finite-difference gradient matches the analytic one (cosine > 0.999) on a tiny model.
8. **Split leak guard:** overlapping calib/val/test files are refused.
9. **Random-code control:** produces valid files and runs end to end.
10. **Loader coverage:** `.pth`, `.safetensors` and `.gguf` (F32/F16/BF16 and Q8_0) all load into the converter; quantized GGUF inputs are dequantized first and the record says so.

**Test only on small, randomly initialized models generated by the tests.** We'll run the real checkpoints ourselves.

## Report

`REPORT.md` describes what was built, how each test was verified, and the timings for converting and evaluating the test models. No claims about how well ternary conversion works on real models.
