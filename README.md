# Ternary Forge

**A post-training ternary converter and measurement bench for RWKV-7.** It takes a trained float RWKV-7 checkpoint and converts its weights to ternary: each weight becomes −1, 0 or +1 times a scale. Then it measures how much the model changed. The question it exists to answer: can a finished RWKV-7 be made ternary, for cheap, fast edge inference, without being retrained?

Xinu: single-file, zero-dependency, offline.

**Status:** the instrument is being built (Argon, from `SPEC.md`). The Kehai Interim audit and the real-model experiments follow.

- `SPEC.md`: the build spec, exactly as sent. It specifies an instrument, not outcomes: the builder never sees the real checkpoints or the success bar.
- `PREREGISTRATION.md`: **what counts as success, fixed before any real-model result existed.** It was committed on 2026-11-11 (sha256 prefix `c25211a11a135ba1`). It must not change after results exist. Any later disagreement with it gets reported under both rules.

Models measured: BlinkDL's RWKV-7 G1d 0.1B and 0.4B (Apache 2.0). They're fetched separately and never committed.

MIT License · ConsciousNode SoftWorks
