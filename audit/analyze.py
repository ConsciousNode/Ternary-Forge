#!/usr/bin/env python3
"""Preregistered analysis: per-method test metrics and paired bootstrap (1,000-token blocks) vs Q4_K_M and Q8_0."""
import numpy as np, glob, os, json, sys
rng = np.random.default_rng(20261111)
out = {}
for split in ('testA-sherlock', 'testB-studio-docs'):
    d = f'res/{split}'
    if not os.path.isdir(d): continue
    R = {}
    for f in sorted(glob.glob(f'{d}/*.npz')):
        z = np.load(f); n = os.path.basename(f)[:-4]
        R[n] = {k: z[k].reshape(-1) for k in z.files}
    nb = len(R['float']['nll']) // 1000
    blk = lambda a: a[:nb * 1000].reshape(nb, 1000).mean(1)
    B = {n: {k: blk(v) for k, v in r.items()} for n, r in R.items()}
    idx = rng.integers(0, nb, size=(1000, nb))
    res = {'tokens': nb * 1000, 'floatNll': float(R['float']['nll'][:nb*1000].mean()), 'methods': {}}
    for n, r in R.items():
        if n == 'float': continue
        m = {'nll': float(r['nll'][:nb*1000].mean()), 'top1Agree': float(r['ag'][:nb*1000].mean()), 'klToFloat': float(r['kl'][:nb*1000].mean())}
        for ref in ('q4_k_m', 'q8_0'):
            if ref not in B or ref == n: continue
            for k, name in (('ag', 'top1Agree'), ('kl', 'klToFloat')):
                diff = B[n][k] - B[ref][k]; bs = diff[idx].mean(1)
                m[f'{name}_minus_{ref}'] = [float(diff.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        res['methods'][n] = m
    out[split] = res
json.dump(out, open('summary.json', 'w'), indent=1)
for split, res in out.items():
    print(f"\n## {split}  ({res['tokens']:,} tokens; float NLL {res['floatNll']:.4f})")
    print('| model | NLL | top1Agree | klToFloat | Δagree vs Q4 [95% CI] | ΔKL vs Q4 [95% CI] |')
    print('|---|---|---|---|---|---|')
    for n, m in sorted(res['methods'].items(), key=lambda kv: kv[1]['klToFloat']):
        a = m.get('top1Agree_minus_q4_k_m'); k = m.get('klToFloat_minus_q4_k_m')
        f = lambda x, s=1: f"{x[0]*s:+.3f} [{x[1]*s:+.3f}, {x[2]*s:+.3f}]" if x else '—'
        print(f"| {n} | {m['nll']:.4f} | {m['top1Agree']*100:.2f}% | {m['klToFloat']:.4f} | {f(a,100)} pp | {f(k)} |")
