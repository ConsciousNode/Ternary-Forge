# Ternary Forge

**ConsciousNode SoftWorks.** A post-training ternary converter (`{-1, 0, +1}` with scales) and measurement bench for **RWKV-7 ("Goose" x070)** checkpoints.

Built under Xinu rules: single-file HTML (`forge.html`), zero external dependencies, offline execution, no network or browser storage APIs, and strict determinism.

## Components

1. **`forge.html`** — Single-file browser application:
   - Loads RWKV-7 checkpoints in `.pth` (PyTorch ZIP), `.safetensors`, and `.gguf` (`F32`, `F16`, `BF16`, and quantized GGUF formats), or previously exported `.tforge` ternary models.
   - Configures and runs conversion methods (`M1 absmean`, `M2 twn`, `M3 act-aware`, `M4 error-comp`, optional `M5 heal`, plus `none` and `random` reference controls) across `tensor`, `row`, and `group:G` (`G ∈ {32, 64, 128}`) granularities.
   - Evaluates models on held-out text (`nll`, `bpb`, `top1Agree`, `klToFloat`, `bitsPerWeight`, `fileBytes`, `tokensPerSec`).
   - Enforces the 64-byte shared-substring split leak guard across calibration and evaluation splits.
   - Exports converted models as `.tforge` binary files and reloads them for interactive chat using Pip's Room's sampler (`sampleLogits` and `makeRepTracker`).
2. **Node CLIs** (zero npm dependencies; all load the canonical engine directly from `forge.html`):
   - `convert.js` — Converts a checkpoint into a `.tforge` ternary model file.
   - `eval.js` — Evaluates a `.tforge` or checkpoint file against held-out text and outputs a metrics JSON record.
   - `run-batch.js` — Evaluates a grid of `methods × settings` along with the float reference (`method: "none"`), random-code control (`method: "random"`), and any supplied `.gguf` reference files, writing one JSON record per line to `results.jsonl`. Refuses to run if `--calib`, `--val`, or `--test` share any 64-byte substring.
   - `tables.js` — Formats `results.jsonl` into Markdown tables with every value computed directly from the input records.
3. **`tests/run.js`** — Self-contained verification suite (Tests 1–10) operating on small, deterministically generated RWKV-7 models.
4. **Documentation**:
   - `METHODS.md` — Mathematical equations and default settings (with one-line reasons) for every conversion method and metric.
   - `FORMAT.md` — Byte-by-byte specification of the `.tforge` binary file format.
   - `REPORT.md` — Engineering summary of what was built, how each of the 10 tests was verified, and execution timings on the synthetic test models.

## CLI Usage

### 1. Convert a Checkpoint (`convert.js`)
```bash
node convert.js \
  --input model.safetensors \
  --output model-m1-row.tforge \
  --method absmean \
  --granularity row \
  --scope default \
  --seed 42
```

With calibration text (`M3 act-aware`, `M4 error-comp`, or `M5 heal`):
```bash
node convert.js \
  --input model.pth \
  --output model-m4-healed.tforge \
  --method error-comp \
  --granularity group:32 \
  --block-size 64 \
  --damping 0.01 \
  --heal \
  --heal-steps 5 \
  --heal-lr 0.001 \
  --calib calib.txt
```

### 2. Evaluate a Model (`eval.js`)
```bash
node eval.js \
  --model model-m1-row.tforge \
  --ref model.safetensors \
  --text test.txt \
  --out metrics.json
```

### 3. Run a Batch Grid (`run-batch.js`)
```bash
node run-batch.js \
  --model model.safetensors \
  --calib calib.txt \
  --val val.txt \
  --test test.txt \
  --gguf model-q8_0.gguf \
  --out results.jsonl
```

### 4. Render Markdown Tables (`tables.js`)
```bash
node tables.js results.jsonl --out tables.md
```

### 5. Run the Verification Suite (`tests/run.js`)
```bash
node tests/run.js
```

## Provenance of Lifted Code

All code lifted from the studio's attached `pips-room.html` is inlined into `forge.html` with explicit line-range provenance comments:
- `// from pips-room.html:1329-1329` — `#rwkv-world-vocab` JSON vocabulary table.
- `// from pips-room.html:1342-1648` — `SIG`, `W_SCALE`, and `class RWKV7` (exact float forward pass, LoRA, and analytical backward pass).
- `// from pips-room.html:1651-1677` — `class RWKVWorldTokenizer` (greedy trie UTF-8 byte tokenizer).
- `// from pips-room.html:1680-2074` — `_halfToF32`, `_toF32`, `readSafetensorsBuffer`, `_zipEntries`, `_unpickleTorch`, `readPthBuffer`, `_GGML`, `_dequantBlocks`, `class QTensor`, `_QDOT`, `readGGUFBuffer`, `rwkv7TensorsFromGGUF`, `rwkv7ConfigFromTensors`, and `loadRWKV7FromBuffer`.
- `// from pips-room.html:3897-3912` — `REP_DEFAULTS` and `makeRepTracker`.
- `// from pips-room.html:3915-3983` — `class TMAC` (81-entry base-3 lookup table ternary matrix-vector kernel).
- `// from pips-room.html:3985-4041` — `class BitLinear`.
- `// from pips-room.html:4342-4392` — `sampleLogits`.
