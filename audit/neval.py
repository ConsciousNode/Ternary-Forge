#!/usr/bin/env python3
"""neval.py: an independent numpy evaluator for RWKV-7 float checkpoints and Ternary Forge .tforge files.

Written by Kehai Interim for the Ternary Forge audit. It reimplements the RWKV-7 forward pass
(following pips-room.html's RWKV7 class) and decodes .tforge files using only FORMAT.md, sharing no code with forge.html.
Ternary matrices are decoded to dense codes x scales (x channelScale), which is mathematically the same
operator TMAC computes. Many independent sequences run as a batch, for speed.
"""
import json, struct, sys, numpy as np

LORA = ('w1', 'w2', 'a1', 'a2', 'v1', 'v2', 'g1', 'g2')
W_SCALE = np.float32(0.606531)

def is_xm(name):
    return name.split('.')[-1] in LORA

def load_float(binp, jsonp):
    idx = json.load(open(jsonp)); raw = np.memmap(binp, dtype=np.float32, mode='r')
    T = {}
    for k, d in idx['t'].items():
        T[k] = np.array(raw[d['off'] // 4: d['off'] // 4 + d['n']]).reshape(d['shape'])
    return idx['cfg'], to_ops(T, {})

def load_tforge(path):
    b = open(path, 'rb').read()
    magic, ver, flags, hlen, ntens = struct.unpack_from('<IHHII', b, 0)
    assert magic == 0x47524654 and ver == 1, 'not a tforge v1 file'
    hdr = json.loads(b[16:16 + hlen].decode('utf-8')); base = 16 + hlen
    lut = np.array([[((B // 3 ** k) % 3) - 1 for k in range(4)] for B in range(81)], dtype=np.int8)
    T, OPS = {}, {}
    for d in hdr['tensors']:
        if d['kind'] == 'float':
            a = np.frombuffer(b, dtype='<f4', count=d['numFloats'], offset=base + d['dataOffset']).copy()
            T[d['name']] = a.reshape(d['origShape'])
        else:
            R, Cc = d['rows'], d['cols']; bpr = (Cc + 3) // 4
            packed = np.frombuffer(b, dtype=np.uint8, count=d['packedBytes'], offset=base + d['packedOffset'])
            assert packed.max() <= 80, 'invalid packed byte'
            codes = lut[packed.reshape(R, bpr)].reshape(R, bpr * 4)[:, :Cc].astype(np.float32)
            sc = np.frombuffer(b, dtype='<f4', count=d['numScales'], offset=base + d['scalesOffset'])
            g = d['granularity']
            if g == 'tensor': W = codes * sc[0]
            elif g == 'row': W = codes * sc[:, None]
            else:
                G = d['groupSize']; ng = (Cc + G - 1) // G
                W = codes * np.repeat(sc.reshape(R, ng), G, axis=1)[:, :Cc]
            if d.get('numChannelScales', 0) > 0:
                cs = np.frombuffer(b, dtype='<f4', count=d['numChannelScales'], offset=base + d['channelScaleOffset'])
                W = W * cs[None, :]
            OPS[d['name']] = np.ascontiguousarray(W.astype(np.float32))   # operator form: y = W @ x
    return hdr['cfg'], to_ops(T, OPS), hdr

def to_ops(T, OPS):
    """Return dict: matrices in operator form (y = M @ x) plus vectors flattened."""
    P = dict(OPS)
    for k, a in T.items():
        if k in P: continue
        if a.ndim == 2 and k not in ('emb.weight', 'head.weight') and not k.endswith('r_k'):
            P[k] = np.ascontiguousarray(a.T if is_xm(k) else a)
        elif k == 'emb.weight' or k == 'head.weight':
            P[k] = np.ascontiguousarray(a)
        else:
            P[k] = a.reshape(-1).astype(np.float32)
    return P

def ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b

def sig(z): return 1.0 / (1.0 + np.exp(-z))

class Model:
    def __init__(self, cfg, P):
        self.P = P; self.C = cfg['dim']; self.N = cfg.get('headSize', 64); self.H = self.C // self.N; self.L = cfg['numLayers']
    def empty(self, B):
        z = lambda *s: np.zeros(s, np.float32)
        return [dict(xa=z(B, self.C), S=z(B, self.H, self.N, self.N), xf=z(B, self.C)) for _ in range(self.L)]
    def lin(self, n, x): return x @ self.P[n].T
    def step(self, toks, st):
        P, C, H, N = self.P, self.C, self.H, self.N; B = len(toks)
        x = ln(P['emb.weight'][toks], P['blocks.0.ln0.weight'], P['blocks.0.ln0.bias'])
        vfirst = None; new = []
        for l in range(self.L):
            b = f'blocks.{l}.'; a = b + 'att.'; f = b + 'ffn.'; s = st[l]
            x1 = ln(x, P[b + 'ln1.weight'], P[b + 'ln1.bias']); xp = s['xa']
            mix = lambda nm: x1 + (xp - x1) * P[a + nm]
            xr, xw, xk, xv, xa, xg = (mix(n) for n in ('x_r', 'x_w', 'x_k', 'x_v', 'x_a', 'x_g'))
            r = self.lin(a + 'receptance.weight', xr); k0 = self.lin(a + 'key.weight', xk); vb = self.lin(a + 'value.weight', xv)
            wl = self.lin(a + 'w2', np.tanh(self.lin(a + 'w1', xw)))
            av = sig(P[a + 'a0'] + self.lin(a + 'a2', self.lin(a + 'a1', xa)))
            g = self.lin(a + 'g2', sig(self.lin(a + 'g1', xg)))
            kkr = (k0 * P[a + 'k_k']).reshape(B, H, N)
            kk = kkr / np.maximum(np.sqrt((kkr ** 2).sum(-1, keepdims=True)), 1e-12)
            k = k0 * (1 + (av - 1) * P[a + 'k_a'])
            if l == 0: v = vb; vfirst = vb
            else:
                gate = sig(P[a + 'v0'] + self.lin(a + 'v2', self.lin(a + 'v1', xv))); v = vb + (vfirst - vb) * gate
            w = np.exp(-W_SCALE * sig(P[a + 'w0'] + wl))
            S0 = s['S']; kkh = kk; avh = av.reshape(B, H, N); vh = v.reshape(B, H, N); kh = k.reshape(B, H, N); rh = r.reshape(B, H, N)
            q = np.einsum('bhij,bhj->bhi', S0, kkh)
            S = S0 * w.reshape(B, H, 1, N) - q[..., :, None] * (kkh * avh)[..., None, :] + vh[..., :, None] * kh[..., None, :]
            o = np.einsum('bhij,bhj->bhi', S, rh)
            m = o.mean(-1, keepdims=True); vr = ((o - m) ** 2).mean(-1, keepdims=True)
            y = ((o - m) / np.sqrt(vr + 64e-5)).reshape(B, C) * P[a + 'ln_x.weight'] + P[a + 'ln_x.bias']
            bon = (rh * kh * P[a + 'r_k'].reshape(H, N)).sum(-1, keepdims=True)
            y = y + (bon * vh).reshape(B, C)
            out = self.lin(a + 'output.weight', y * g)
            x1r = x + out
            x2 = ln(x1r, P[b + 'ln2.weight'], P[b + 'ln2.bias'])
            kin = x2 + (s['xf'] - x2) * P[f + 'x_k']
            hk = self.lin(f + 'key.weight', kin); rel = np.where(hk > 0, hk * hk, 0)
            x = x1r + self.lin(f + 'value.weight', rel)
            new.append(dict(xa=x1, S=S.astype(np.float32), xf=x2))
        lo = ln(x, P['ln_out.weight'], P['ln_out.bias'])
        return (lo @ P['head.weight'].T).astype(np.float32), new

def logsoftmax(z):
    m = z.max(-1, keepdims=True); return z - m - np.log(np.exp(z - m).sum(-1, keepdims=True))

def evaluate(student, teacher, ids, nseq, seqlen, report=None):
    """Per-token arrays (nll, kl, agree, isTarget) over nseq independent contiguous chunks."""
    need = nseq * (seqlen + 1); ids = ids[:need].reshape(nseq, seqlen + 1)
    ss, ts = student.empty(nseq), (teacher.empty(nseq) if teacher is not student else None)
    NLL = np.zeros((nseq, seqlen), np.float64); KL = np.zeros_like(NLL); AG = np.zeros_like(NLL)
    for t in range(seqlen):
        zs, ss = student.step(ids[:, t], ss)
        ls = logsoftmax(zs.astype(np.float64))
        NLL[:, t] = -ls[np.arange(nseq), ids[:, t + 1]]
        if teacher is student: KL[:, t] = 0; AG[:, t] = 1
        else:
            zt, ts = teacher.step(ids[:, t], ts); lt = logsoftmax(zt.astype(np.float64))
            pt = np.exp(lt); KL[:, t] = (pt * (lt - ls)).sum(-1); AG[:, t] = (zs.argmax(-1) == zt.argmax(-1))
        if report and (t % 50 == 0): report(t)
    return NLL, KL, AG

if __name__ == '__main__':
    pass
