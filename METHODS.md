# Ternary Conversion Methods & Equations (`METHODS.md`)

Every conversion method takes an operator-oriented weight matrix $W \in \mathbb{R}^{R \times C}$ (where $R = \text{outDim}$ is the number of output channels and $C = \text{inDim}$ is the number of input channels; for RWKV-7 `_xM` LoRA matrices stored as $[C, R]$ in checkpoints, $W = M^\top \in \mathbb{R}^{R \times C}$) and produces a ternary code matrix $C_{\text{code}} \in \{-1, 0, +1\}^{R \times C}$ packed in base-3 (`TMAC` format) plus float32 scales $\gamma$.

## Shared Settings & Scale Units

### 1. `scope` and `keep`
- **`scope`** (default: `"default"`): Selects which matrices are converted to ternary.
  - *Default reason:* Converting all per-layer linear (`att.receptance`, `att.key`, `att.value`, `att.output`, `ffn.key`, `ffn.value`) and low-rank LoRA (`w1`, `w2`, `a1`, `a2`, `v1`, `v2`, `g1`, `g2`) matrices covers the bulk of recurrent block parameters while preserving token embedding lookup, LayerNorm/GroupNorm statistics, per-channel time-shift vectors, and the output vocabulary projection unless explicitly requested.
- **`keep`** (default: `[]`): Optional list of tensor names or substrings forced to remain float32.
  - *Default reason:* Empty by default so all tensors selected by `scope` are uniformly converted unless selective layer bypassing is tested.

### 2. `granularity`
Defines the partition of $W \in \mathbb{R}^{R \times C}$ into disjoint scale units $U \subseteq \{0, \dots, R-1\} \times \{0, \dots, C-1\}$, each assigned one float32 scale $\gamma_U$:
- **`tensor`**: One scale unit covering all $R \cdot C$ weights ($N_{\text{scales}} = 1$).
- **`row`** (default in CLI/UI): One scale unit per output row $r \in \{0, \dots, R-1\}$ ($N_{\text{scales}} = R$).
  - *Default reason:* Per-row scaling aligns each scale with a single output channel's dot product in `TMAC` without adding inner-loop group-accumulation overhead.
- **`group:G`** ($G \in \{32, 64, 128\}$): One scale unit per contiguous slice of $G$ input weights within each output row:
  $$U_{r, g} = \{ (r, c) : gG \le c < \min((g+1)G, C) \}, \qquad N_{\text{scales}} = R \cdot \left\lceil \frac{C}{G} \right\rceil$$

## Conversion Methods

### Reference `none` (Float Baseline)
Leaves all tensors in IEEE-754 float32 without quantization:
$$\widehat{W} = W, \qquad \text{bitsPerWeight} = 32.0$$

### Control `random` (Fixed-Seed Uniform Random Ternary Floor)
Computes the exact same scale $\gamma_U$ per scale unit $U$ as **M1 (`absmean`)**, but draws each ternary code $c_{r, c} \in \{-1, 0, +1\}$ independently and uniformly at random ($\mathbb{P}(c = -1) = \mathbb{P}(c = 0) = \mathbb{P}(c = +1) = \tfrac{1}{3}$) using a deterministic Mulberry32 PRNG seeded with `seed`:
$$\gamma_U = \frac{1}{|U|} \sum_{(r, c) \in U} |W_{r, c}|, \qquad c_{r, c} \sim \text{Uniform}(\{-1, 0, +1\}; \text{seed})$$
- **Default `seed`:** `42` — *Reason:* Fixed seed guarantees deterministic reproduction of the random-code baseline across runs.

### M1: `absmean` (BitNet b1.58 Absmean Quantization)
For each scale unit $U$:
$$\gamma_U = \frac{1}{|U|} \sum_{(r, c) \in U} |W_{r, c}|$$
$$c_{r, c} = \begin{cases} \text{clip}\!\left(\text{round}\!\left(\dfrac{W_{r, c}}{\gamma_U}\right), -1, 1\right) & \text{if } \gamma_U > 0 \\ 0 & \text{if } \gamma_U = 0 \end{cases}$$
$$\widehat{W}_{r, c} = \gamma_U \cdot c_{r, c}$$
- **Defaults:** No extra hyperparameters beyond `scope`, `keep`, and `granularity` — *Reason:* Parameter-free closed-form estimator from BitNet b1.58.

### M2: `twn` (Ternary Weight Networks: Threshold-Then-Fit)
For each scale unit $U$ and threshold $\Delta_U \ge 0$, define the active support $I(\Delta_U) = \{ (r, c) \in U : |W_{r, c}| > \Delta_U \}$. The ternary codes and least-squares optimal scale for a given $\Delta_U$ are:
$$c_{r, c}(\Delta_U) = \begin{cases} +1 & \text{if } W_{r, c} > \Delta_U \\ -1 & \text{if } W_{r, c} < -\Delta_U \\ 0 & \text{if } |W_{r, c}| \le \Delta_U \end{cases}$$
$$\gamma_U(\Delta_U) = \begin{cases} \dfrac{1}{|I(\Delta_U)|} \sum_{(r, c) \in I(\Delta_U)} |W_{r, c}| & \text{if } |I(\Delta_U)| > 0 \\ 0 & \text{otherwise} \end{cases}$$
Two threshold selection modes are provided via `twnMode`:
1. **`closed`** (default):
   $$\Delta_U = 0.7 \cdot \frac{1}{|U|} \sum_{(r, c) \in U} |W_{r, c}|$$
2. **`search`**:
   Evaluates $\Delta_0 = 0.7 \cdot \text{mean}_{U}|W|$ and a uniform grid of $G_{\text{steps}}$ points $\Delta_k = \frac{k}{G_{\text{steps}} + 1} \max_{(r,c) \in U} |W_{r,c}|$ for $k = 0, \dots, G_{\text{steps}}$, selecting:
   $$\Delta_U^* = \arg\min_{\Delta \in \{\Delta_0\} \cup \{\Delta_k\}} \sum_{(r, c) \in U} \left( W_{r, c} - \gamma_U(\Delta) c_{r, c}(\Delta) \right)^2$$
   Because $\Delta_0$ is always in the candidate set, `search` is guaranteed never to have higher reconstruction error than `closed`.
- **Default `twnMode`:** `"closed"` — *Reason:* O(1)-pass closed-form approximation from Li et al. (2016) requiring no grid sweep unless requested.
- **Default `gridSteps`:** `100` — *Reason:* Provides 1% threshold resolution across $[0, \max|w|)$ with negligible conversion latency.

### M3: `act-aware` (AWQ-Style Activation-Aware Per-Channel Scaling)
Runs the float model forward on calibration tokens (`--calib` only) to record the mean absolute input activation per input channel $c \in \{0, \dots, C-1\}$ for each converted matrix:
$$\mu_c = \frac{1}{T_{\text{calib}}} \sum_{t=1}^{T_{\text{calib}}} |x_{t, c}|$$
Computes normalized per-channel salience scales with exponent $\alpha \in [0, 1]$:
$$\tilde{s}_c = (\max(\mu_c, 10^{-4}))^\alpha, \qquad s_c = \text{clip}\!\left(\frac{\tilde{s}_c}{\left(\prod_{j=0}^{C-1} \tilde{s}_j\right)^{1/C}}, 10^{-4}, 10^4\right), \qquad d_c = \frac{1}{s_c}$$
Rescales input columns of $W$ prior to quantization:
$$\widetilde{W}_{r, c} = W_{r, c} \cdot s_c$$
and quantizes $\widetilde{W}$ using the base quantizer (`baseMethod`: M1 `absmean` or M2 `twn`) to obtain $(\gamma_U, c_{r, c})$.
To preserve the linear mapping $W x = \widetilde{W} (d \odot x) \approx \widehat{W} (d \odot x)$:
- **Exact Algebraic Fold-In (where the RWKV-7 graph allows)**:
  - For `head.weight` (preceded directly by `ln_out`): $d_c$ is folded into `ln_out.weight` and `ln_out.bias`:
    $$w_{\text{ln\_out}}[c] \leftarrow w_{\text{ln\_out}}[c] \cdot d_c, \qquad b_{\text{ln\_out}}[c] \leftarrow b_{\text{ln\_out}}[c] \cdot d_c$$
  - For `blocks.l.att.a2` and `blocks.l.att.v2` (preceded linearly by `a1` and `v1` with no intervening nonlinearity): when `a1`/`v1` is kept float or converted with `row` or `group:G` granularity, $d_c$ is folded directly into output row $c$ of `a1`/`v1` (or its row/group scales).
- **Explicit Per-Channel Float Vector (where nonlinearity or shared branching prevents folding)**:
  - Stored as `channelScale` ($d \in \mathbb{R}^C$, $C$ float32 values), applied to $x$ before `TMAC` LUT construction ($x'_c = d_c x_c$), written to the `.tforge` file, and counted in `bitsPerWeight` ($+32 C$ bits).
- **Default `actAlpha` ($\alpha$):** `0.5` — *Reason:* Balances weight dynamic range and activation salience (standard AWQ midpoint between $\alpha=0$ and $\alpha=1$).
- **Default `baseMethod`:** `"absmean"` — *Reason:* Pairs activation scaling with parameter-free M1 rounding by default.

### M4: `error-comp` (GPTQ / OBQ Sequential Rounding with Calibration Hessian)
Collects the empirical input Hessian on `--calib` activations $X \in \mathbb{R}^{T_{\text{calib}} \times C}$:
$$H = \frac{2}{T_{\text{calib}}} X^\top X, \qquad H_{\lambda} = H + \lambda \left(\frac{1}{C} \sum_{c=0}^{C-1} H_{c, c}\right) I_C$$
Computes the upper-triangular Cholesky factorization of the inverse Hessian $H_{\lambda}^{-1} = U^\top U$ (via $H_{\lambda} = L L^\top \implies H_{\lambda}^{-1} = L^{-\top} L^{-1} = U^\top U$).
Processes columns in blocks of size $B$ (`blockSize`). For each column $q = 0, \dots, C-1$:
1. Rounds column $q$ of the working matrix $W^{\text{work}}$ to ternary using scale $\gamma_{U(r, q)}$:
   $$c_{r, q} = \text{clip}\!\left(\text{round}\!\left(\frac{W^{\text{work}}_{r, q}}{\gamma_{U(r, q)}}\right), -1, 1\right), \qquad \widehat{W}_{r, q} = \gamma_{U(r, q)} c_{r, q}$$
2. Computes the scaled quantization error:
   $$e_{r, q} = \frac{W^{\text{work}}_{r, q} - \widehat{W}_{r, q}}{U_{q, q}}$$
3. Pushes the error onto unrounded columns $j > q$ (immediately within the current block $q < j < b_1$, and lazily across $j \ge b_1$ at the end of each block):
   $$W^{\text{work}}_{r, j} \leftarrow W^{\text{work}}_{r, j} - e_{r, q} U_{q, j}$$
4. Refines each scale unit's $\gamma_U$ to minimize the quadratic Hessian objective $(w_U - \gamma_U c_U) H_{\lambda, U} (w_U - \gamma_U c_U)^\top$:
   $$\gamma_U^* = \frac{w_U H_{\lambda, U} c_U^\top}{c_U H_{\lambda, U} c_U^\top}$$
- **Default `blockSize` ($B$):** `64` — *Reason:* Matches standard RWKV-7 head size ($N=64$) and balances cache locality with lazy block updates.
- **Default `damping` ($\lambda$):** `0.01` — *Reason:* 1% relative diagonal Tikhonov regularization guarantees positive-definite Cholesky factorization on rank-deficient calibration sequences.

### M5: `heal` (Post-Conversion KL Distillation Stage via Exact BPTT)
Optional stage executable after any of M1–M4 using `--calib` text only.
Given teacher logits $z_t^{\text{float}}$ ($p_t = \text{softmax}(z_t^{\text{float}})$) and student logits $z_t^{\text{stud}}$ ($q_t = \text{softmax}(z_t^{\text{stud}})$) over a calibration sequence of length $T$:
$$\mathcal{L}_{\text{KL}} = \frac{1}{T} \sum_{t=0}^{T-1} \sum_{v=0}^{V-1} p_{t, v} \log \frac{p_{t, v}}{q_{t, v}}, \qquad \frac{\partial \mathcal{L}_{\text{KL}}}{\partial z_{t, v}^{\text{stud}}} = \frac{1}{T}\left(q_{t, v} - p_{t, v}\right)$$
Runs the exact analytical `RWKV7.backward` pass through time ($t = T-1 \dots 0$). For each converted tensor $\widehat{W}_{r, c} = \gamma_{U(r, c)} c_{r, c} d_c$ with backpropagated output-input outer gradient $G_{r, c} = \frac{\partial \mathcal{L}_{\text{KL}}}{\partial \widehat{W}_{r, c}}$:
$$\frac{\partial \mathcal{L}_{\text{KL}}}{\partial \gamma_U} = \sum_{(r, c) \in U} G_{r, c} \cdot c_{r, c} \cdot d_c, \qquad \frac{\partial \mathcal{L}_{\text{KL}}}{\partial d_c} = \sum_{r=0}^{R-1} G_{r, c} \cdot \gamma_{U(r, c)} \cdot c_{r, c}$$
- **Default `healSteps`:** `5` — *Reason:* Short calibration fine-tuning pass to adjust scales without overfitting to the calibration snippet.
- **Default `healLr`:** `0.001` — *Reason:* Stable step size for scale refinement under exact BPTT KL gradients.
- **Default `healSte`:** `false` (frozen codes) — *Reason:* Freezing ternary codes preserves exact analytical differentiability with respect to scales and requires zero latent float weight buffers.
- **Default `healVectors`:** `false` — *Reason:* Restricts training to quantization scales by default so unquantized tensors remain untouched unless enabled.

## Evaluation Metrics

For a sequence of $T$ tokens $(t_0, t_1, \dots, t_{T-1})$ with $N_{\text{pred}} = T - 1$ next-token predictions and $B_{\text{pred}}$ UTF-8 bytes in $(t_1, \dots, t_{T-1})$:
1. **`nll` (nats/token):**
   $$\text{nll} = -\frac{1}{N_{\text{pred}}} \sum_{k=0}^{N_{\text{pred}}-1} \log q_k(t_{k+1})$$
2. **`bpb` (bits per byte):**
   $$\text{bpb} = \frac{\sum_{k=0}^{N_{\text{pred}}-1} -\log_2 q_k(t_{k+1})}{B_{\text{pred}}} = \frac{N_{\text{pred}} \cdot \text{nll}}{B_{\text{pred}} \cdot \ln 2}$$
3. **`top1Agree`:**
   $$\text{top1Agree} = \frac{1}{N_{\text{pred}}} \sum_{k=0}^{N_{\text{pred}}-1} \mathbb{I}\!\left(\arg\max_v z_{k, v}^{\text{ternary}} = \arg\max_v z_{k, v}^{\text{float}}\right)$$
4. **`klToFloat`:**
   $$\text{klToFloat} = \frac{1}{N_{\text{pred}}} \sum_{k=0}^{N_{\text{pred}}-1} \sum_{v=0}^{V-1} p_{k, v}^{\text{float}} \log \frac{p_{k, v}^{\text{float}}}{q_{k, v}^{\text{ternary}}}$$
5. **`bitsPerWeight`:**
   Across the set $\mathcal{C}$ of all converted tensors (where tensor $M \in \mathcal{C}$ has $N_M = R_M C_M$ weights, $S_M$ float32 scales, and $E_M$ extra float32 channel-scale elements):
   $$\text{bitsPerWeight} = \frac{\sum_{M \in \mathcal{C}} \left( N_M \log_2(3) + 32 S_M + 32 E_M \right)}{\sum_{M \in \mathcal{C}} N_M}$$
