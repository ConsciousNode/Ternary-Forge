# Ternary Forge Binary File Format (`.tforge` — `FORMAT.md`)

The `.tforge` format is a deterministic, self-contained, little-endian binary container for ternary-converted RWKV-7 models. It holds **only** packed ternary codes, float32 scales, explicit per-channel float32 vectors (when added by M3 `act-aware`), and the unquantized float32 tensors declared by `scope`/`keep`.

## 1. File Layout Overview

```
+-----------------------------------------------------------------------+
| Offset 0..15 (16 bytes): Fixed Binary Header                          |
|   0..3   : uint32 LE  magic        = 0x47524654 ("TFRG" in ASCII)     |
|   4..5   : uint16 LE  version      = 1                                |
|   6..7   : uint16 LE  flags        = 0 (reserved, must be 0)          |
|   8..11  : uint32 LE  headerLen    = byte length of JSON header       |
|   12..15 : uint32 LE  numTensors   = total number of tensor entries   |
+-----------------------------------------------------------------------+
| Offset 16 .. (16 + headerLen - 1): UTF-8 JSON Header                  |
|   Padded with ASCII space (0x20) to a 4-byte boundary so that         |
|   payloadBase = 16 + headerLen is strictly 4-byte aligned.            |
+-----------------------------------------------------------------------+
| Offset payloadBase = 16 + headerLen .. EOF: Binary Payload            |
|   Concatenated tensor buffers in lexicographical tensor name order.   |
|   Every buffer start offset is aligned to a 4-byte boundary (padded   |
|   with 0x00 bytes where packedBytes % 4 != 0).                        |
+-----------------------------------------------------------------------+
```

## 2. Fixed Binary Header (Bytes `0..15`)

| Byte Offset | Type | Field | Value / Description |
|---|---|---|---|
| `0..3` | `uint32` LE | `magic` | `0x47524654` (ASCII bytes `0x54, 0x46, 0x52, 0x47` = `"TFRG"`) |
| `4..5` | `uint16` LE | `version` | `1` (`0x0001`) |
| `6..7` | `uint16` LE | `flags` | `0` (`0x0000`) |
| `8..11` | `uint32` LE | `headerLen` | Byte length of the UTF-8 JSON header (always a multiple of 4) |
| `12..15` | `uint32` LE | `numTensors` | Number of tensor descriptors in `header.tensors` |

## 3. UTF-8 JSON Header (`16 .. 16 + headerLen - 1`)

The JSON object contains three top-level keys:
- **`format`**: `"tforge-v1"`
- **`cfg`**: RWKV-7 architecture configuration object:
  - `vocabSize`, `dim`, `numLayers`, `headSize`, `dimFFN`, `dDecay`, `dAAA`, `dMV`, `dGate`
- **`conversion`**: Conversion provenance and bit-accounting record:
  - `method`, `scope`, `keep`, `granularity`, `seed`, `settings`, `wasDequantized`, `dequantizedFrom`, `bitsPerWeight`, `convertedWeights`, `convertedTensors`, `totalScaleFloats`, `totalExtraFloats`, `codeSha256`
- **`tensors`**: Array of `numTensors` descriptor objects sorted lexicographically by `name`.

### 3.1 Descriptor for a Converted Ternary Tensor (`kind: "ternary"`)
```json
{
  "name": "blocks.0.att.receptance.weight",
  "kind": "ternary",
  "origShape": [32, 32],
  "rows": 32,
  "cols": 32,
  "transposed": false,
  "granularity": "row",
  "groupSize": 0,
  "packedOffset": 0,
  "packedBytes": 256,
  "scalesOffset": 256,
  "numScales": 32,
  "channelScaleOffset": -1,
  "numChannelScales": 0
}
```
- `origShape`: Original checkpoint shape (`[outDim, inDim]` for `_lin` matrices; `[inDim, outDim]` for `_xM` LoRA matrices `w1..g2`).
- `rows`, `cols`: Operator matrix dimensions (`rows = outDim`, `cols = inDim`).
- `transposed`: `true` if `origShape` was `[cols, rows]` (`_xM` LoRA matrices), `false` if `[rows, cols]`.
- `packedOffset`: Byte offset from `payloadBase` (`16 + headerLen`) to the packed ternary code array.
- `packedBytes`: Exact byte length of the packed code array, equal to $\text{rows} \times \lceil \text{cols} / 4 \rceil$.
- `scalesOffset`: Byte offset from `payloadBase` (4-byte aligned) to `numScales` little-endian `float32` scale values (`numScales * 4` bytes).
- `channelScaleOffset`: Byte offset from `payloadBase` (4-byte aligned) to `numChannelScales` little-endian `float32` input-channel scale values (`cols * 4` bytes) when M3 `act-aware` leaves an unfolded per-channel vector, or `-1` when none is present (`numChannelScales = 0`).

### 3.2 Descriptor for an Unquantized Float Tensor (`kind: "float"`)
```json
{
  "name": "blocks.0.ln1.weight",
  "kind": "float",
  "origShape": [32],
  "numFloats": 32,
  "dataOffset": 384
}
```
- `dataOffset`: Byte offset from `payloadBase` (4-byte aligned) to `numFloats` little-endian IEEE-754 `float32` values (`numFloats * 4` bytes).

## 4. Base-3 Ternary Code Packing (`TMAC` Layout)

Within each converted tensor (`kind: "ternary"`), each output row $r \in \{0, \dots, \text{rows}-1\}$ is stored contiguously in $\text{blocksPerRow} = \lceil \text{cols} / 4 \rceil$ bytes.

Each byte $B \in \{0, \dots, 80\}$ packs 4 consecutive ternary codes $(t_0, t_1, t_2, t_3) \in \{-1, 0, +1\}^4$ along the input dimension (`c = 4b + 0 .. 4b + 3`; out-of-bounds tail positions when $\text{cols} \bmod 4 \ne 0$ are padded with $t_k = 0$):
$$B = (t_0 + 1) + 3(t_1 + 1) + 9(t_2 + 1) + 27(t_3 + 1) \in \{0, \dots, 80\}$$

### Unpacking Equation (`TMAC.UNPACK_TABLE`)
For any packed byte $B \in \{0, \dots, 80\}$ and sub-index $k \in \{0, 1, 2, 3\}$:
$$t_k = \left( \left\lfloor \frac{B}{3^k} \right\rfloor \bmod 3 \right) - 1 \in \{-1, 0, +1\}$$
Any byte value $B > 80$ is invalid and rejected by the format verifier.

## 5. Independent `bitsPerWeight` Recomputation from `.tforge`

Given a `.tforge` file, `bitsPerWeight` is defined strictly from the `"ternary"` descriptors in `header.tensors`:
$$N_{\text{conv}} = \sum_{d \in \text{ternary}} (d.\text{rows} \times d.\text{cols}), \qquad S_{\text{total}} = \sum_{d \in \text{ternary}} d.\text{numScales}, \qquad E_{\text{total}} = \sum_{d \in \text{ternary}} d.\text{numChannelScales}$$
$$\text{bitsPerWeight} = \frac{N_{\text{conv}} \cdot \log_2(3) + 32 \cdot S_{\text{total}} + 32 \cdot E_{\text{total}}}{N_{\text{conv}}}$$
