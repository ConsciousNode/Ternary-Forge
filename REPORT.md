# Ternary Forge — Build & Verification Report (`REPORT.md`)

**ConsciousNode SoftWorks.** Engineering report covering what was built, how each test in `tests/run.js` was verified, and execution timings for converting and evaluating the synthetic test models.

---

## 1. What Was Built

1. **`forge.html` (Single-File Browser Application)**:
   - Self-contained offline HTML instrument with strict Content-Security-Policy (`default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; connect-src 'none'; ...`) and zero network or browser storage APIs.
   - Lifts the studio's RWKV-7 (`x070` Goose) float forward/backward pass, `RWKVWorldTokenizer`, `.pth`/`.safetensors`/`.gguf` binary loaders, `TMAC` and `BitLinear` ternary lookup-table kernels, `makeRepTracker`, and `sampleLogits` directly from `pips-room.html` with line-by-line provenance comments (`// from pips-room.html:LINE-LINE`).
   - Implements `TernaryMatrix` (holding only base-3 packed `Uint8Array` ternary codes, `Float32Array` scales, and optional M3 per-channel `Float32Array` vectors — zero float weights) and extends `TMAC` to support `tensor`, `row`, and `group:G` ($G \in \{32, 64, 128\}$) granularities using the 81-entry 4-trit lookup table.
   - Implements all five conversion methods (`M1 absmean`, `M2 twn` closed/search, `M3 act-aware` with exact algebraic fold-in where valid and explicit counted per-channel vectors elsewhere, `M4 error-comp` Cholesky inverse-Hessian sequential rounding, and optional `M5 heal` KL distillation via exact BPTT with frozen codes or STE) plus `none` (float) and `random` (fixed-seed uniform random ternary codes) reference controls.
   - Provides interactive UI panels for checkpoint loading, method/granularity/scope/keep configuration, 64-byte split overlap checking, evaluation (`nll`, `bpb`, `top1Agree`, `klToFloat`, `bitsPerWeight`, `fileBytes`, `tokensPerSec`), `.tforge` binary export/import, `results.jsonl` and Markdown table export, and interactive chat via Pip's Room's sampler.
2. **Node CLIs**:
   - `convert.js`: Converts `.pth`, `.safetensors`, or `.gguf` checkpoints to `.tforge` binary files.
   - `eval.js`: Evaluates `.tforge` or checkpoint files on text and outputs metrics JSON with SHA-256 provenance hashes.
   - `run-batch.js`: Enforces the 64-byte substring overlap guard across `--calib`, `--val`, and `--test`, runs the float reference (`none`), random-code control (`random`), any supplied `.gguf` references, and all grid configurations in exact order without ranking or filtering, writing `results.jsonl`.
   - `tables.js`: Computes Markdown tables directly from `results.jsonl`.
3. **Documentation & Verification Suite**:
   - `README.md`, `METHODS.md`, `FORMAT.md`, `REPORT.md`, and `tests/run.js`.

---

## 2. How Each Test Was Verified (`node tests/run.js`)

All 10 tests run on small, deterministically generated RWKV-7 models (`V=64, C=32, L=2, H=2, N=16` and `V=16, C=8, L=2, H=2, N=4`):

| Test # | Test Name | Verification Method | Time (ms) | Status |
|---:|---|---|---:|---|
| 1 | **Static scan** | Regex and AST-level scan of `forge.html`, `convert.js`, `eval.js`, `run-batch.js`, and `tables.js` verifying presence of strict CSP (`default-src 'none'`, `connect-src 'none'`), 7 provenance labels (`// from pips-room.html:LINE-LINE`), and zero occurrences of `fetch`, `XMLHttpRequest`, `WebSocket`, `EventSource`, `sendBeacon`, `localStorage`, `sessionStorage`, `indexedDB`, `cookie`, or external scripts/stylesheets. | 36.6 | PASS |
| 2 | **Float parity** | Instantiates the unmodified `RWKV7` class extracted from `pips-room.html:1342-1648` in an isolated VM context alongside `TernaryForge.RWKV7` with identical initial tensors (`V=64, C=32, L=2`), steps through a 12-token sequence, and verifies raw `Uint32Array` bit patterns of all 768 output float32 logits match bit-for-bit. | 99.3 | PASS |
| 3 | **Ternary invariant** | Converts a synthetic RWKV-7 model across 8 method/granularity combinations (`M1` tensor/row/group:32, `M2` search, `M3` tensor/row, `M4` row, `M5` heal), verifies converted in-memory tensors hold no `.weight`, `.data`, or `.steLatent` float arrays, exports to `.tforge`, unpacks all 188,416 ternary codes from the binary payload to assert each $\in \{-1, 0, +1\}$, and independently recomputes `bitsPerWeight` from the binary descriptors to match the evaluation record within $10^{-12}$. | 1609.7 | PASS |
| 4 | **Round trip** | Converts a synthetic model in memory across `tensor`, `row`, and `group:32` granularities (`M1`, `M2`, `M3`, `M4`), exports each to `.tforge`, imports a fresh model instance from the buffer, runs an 8-token sequence through both, and asserts all float32 logits match bit-for-bit. | 116.1 | PASS |
| 5 | **Determinism** | Runs conversion, `.tforge` serialization, and evaluation twice with identical inputs and seed (`seed=777`) across `random`, `M1`, `M2`, `M3`, `M4`, and `M5`, verifying `Buffer.compare(bufA, bufB) === 0` and bit-identical metric values (`nll`, `bpb`, `top1Agree`, `klToFloat`, `bitsPerWeight`, `fileBytes`). | 1246.6 | PASS |
| 6 | **Method correctness on toy matrices** | (a) Verifies `M1` and `M2` (`closed`) against hand-calculated scales (`[0.525, 0.425, 0.70, 0.20]` vs `[0.80, 0.60, 2.40, 0.20]`) and codes on a fixed $4 \times 8$ matrix; (b) verifies `M2 search` reconstruction error $\le$ `M2 closed` across `tensor`, `row`, and `group:32`; (c) verifies `M4 error-comp` achieves lower held-out output MSE (`0.14348`) than `M1` (`0.35592`) on correlated synthetic inputs. | 36.8 | PASS |
| 7 | **Heal gradient check** | Builds a 2-layer RWKV-7 model in `Float64Array` (`V=16, C=8, L=2`), converts to ternary (`row` granularity), computes analytical BPTT gradients of the M5 KL loss across all 508 trainable parameters (ternary scales + float per-channel vectors), and compares against central finite differences ($\epsilon = 10^{-6}$), achieving cosine similarity `1.00000000` ($> 0.999$). | 1006.4 | PASS |
| 8 | **Split leak guard** | Verifies `checkSplitOverlap` and `run-batch.js` accept disjoint splits and reject any pair (`calib`/`val`, `calib`/`test`, `val`/`test`) sharing a 64-byte substring. | 15.4 | PASS |
| 9 | **Random-code control** | Verifies `method: "random"` produces identical scales to `M1 absmean` with pseudorandom ternary codes, and runs `convert.js`, `eval.js`, `run-batch.js`, and `tables.js` end-to-end. | 557.6 | PASS |
| 10 | **Loader coverage** | Synthesizes 10 checkpoint files (`.safetensors`, `.pth`, and `.gguf` in `F32`, `F16`, `BF16`, plus `Q8_0` `.gguf`), loads each into the converter, verifies `F32` cross-format logit parity, and confirms `Q8_0` `.gguf` is dequantized prior to conversion with `wasDequantized: true` and `dequantizedFrom: "Q8_0"` recorded in the output metrics. | 795.3 | PASS |

**Total Suite Time:** `5530.6 ms` (`10/10` passed).

---

## 3. Timings for Converting and Evaluating the Synthetic Test Models

Measured on the synthetic 2-layer RWKV-7 test model (`vocabSize = 64, dim = 32, numLayers = 2, headSize = 16, dimFFN = 64`, 69 tensors, 30,272 total parameters, 23,552 converted weights under `scope = "default"`):

| Configuration | Conversion Time (ms) | `.tforge` Export Time (ms) | Evaluation Time (ms) | Exported Size (bytes) | Effective `bitsPerWeight` | TMAC Decode Speed (tokens/s) |
|---|---:|---:|---:|---:|---:|---:|
| `none` (Float reference) | 3.28 | 3.99 | 553.24 | 123,168 | 32.0000 | 3,660.4 |
| `random` (Control, `row`) | 27.00 | 0.35 | 162.67 | 45,840 | 2.6502 | 4,925.2 |
| `M1 absmean` (`tensor`) | 6.65 | 0.28 | 93.53 | 42,900 | 1.6203 | 2,316.6 |
| `M1 absmean` (`row`) | 1.36 | 2.45 | 100.28 | 45,844 | 2.6502 | 3,503.5 |
| `M1 absmean` (`group:32`) | 6.74 | 2.36 | 107.49 | 46,396 | 2.7371 | 2,466.7 |
| `M2 twn` (`closed`, `row`) | 12.36 | 0.29 | 77.21 | 45,868 | 2.6502 | 2,620.7 |
| `M2 twn` (`search`, `row`) | 45.90 | 0.34 | 71.92 | 45,868 | 2.6502 | 3,545.7 |
| `M3 act-aware` (`row`) | 114.46 | 0.32 | 155.57 | 49,004 | 3.6502 | 1,884.6 |
| `M4 error-comp` (`row`) | 156.16 | 0.36 | 33.46 | 45,880 | 2.6502 | 6,505.1 |
| `M1 + M5 heal` (5 steps, frozen) | 300.66 | 0.33 | 67.34 | 45,960 | 2.6502 | 1,950.0 |
| `M1 + M5 heal` (5 steps, STE) | 491.22 | 0.30 | 86.00 | 45,960 | 2.6502 | 6,498.4 |
