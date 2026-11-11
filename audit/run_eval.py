import sys, os, json, time, pickle, hashlib, numpy as np
sys.path.insert(0, '/home/claude/tf-exp'); from neval import *
split, nseq, seqlen, outdir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; names = sys.argv[5:]
os.makedirs(outdir, exist_ok=True)
ids = np.fromfile(f'data/{split}.tok.u32', dtype=np.uint32).astype(np.int64)
def load(n):
    if n.endswith('.tforge'): cfg, P, _ = load_tforge('models/' + n)
    elif n == 'float': cfg, P = load_float('float.bin', 'float.json')
    else: cfg, P = load_float(f'models/{n}.bin', f'models/{n}.json')
    return Model(cfg, P)
teacher = load('float'); studs = {}
for n in names:
    m = load(n)
    for k in ('emb.weight', 'head.weight'):   # share identical big float tables with the teacher (memory)
        if m.P[k].shape == teacher.P[k].shape and np.array_equal(m.P[k], teacher.P[k]): m.P[k] = teacher.P[k]
        else: print(f'  note: {n} {k} differs from float; kept its own copy', flush=True)
    studs[n] = m
need = nseq * (seqlen + 1); X = ids[:need].reshape(nseq, seqlen + 1)
ts = teacher.empty(nseq); ss = {n: m.empty(nseq) for n, m in studs.items()}
R = {n: dict(nll=np.zeros((nseq, seqlen)), kl=np.zeros((nseq, seqlen)), ag=np.zeros((nseq, seqlen))) for n in names}
Rf = np.zeros((nseq, seqlen)); t0 = time.time(); t_start = 0
ck = f"{outdir}/ckpt-{hashlib.md5(' '.join(names).encode()).hexdigest()[:8]}.pkl"
if os.path.exists(ck):
    with open(ck, 'rb') as f: t_start, ts, ss, R, Rf = pickle.load(f)
    print(f'  resumed at t={t_start}', flush=True)
for t in range(t_start, seqlen):
    if t % 64 == 0 and t > t_start:
        with open(ck + '.tmp', 'wb') as f: pickle.dump((t, ts, ss, R, Rf), f)
        os.replace(ck + '.tmp', ck)
    zt, ts = teacher.step(X[:, t], ts); lt = logsoftmax(zt.astype(np.float64)); pt = np.exp(lt); at = zt.argmax(-1)
    Rf[:, t] = -lt[np.arange(nseq), X[:, t + 1]]
    for n, m in studs.items():
        zs, ss[n] = m.step(X[:, t], ss[n]); ls = logsoftmax(zs.astype(np.float64))
        R[n]['nll'][:, t] = -ls[np.arange(nseq), X[:, t + 1]]; R[n]['kl'][:, t] = (pt * (lt - ls)).sum(-1); R[n]['ag'][:, t] = (zs.argmax(-1) == at)
    if t % 64 == 0: print(f'  t={t} {time.time()-t0:.0f}s', flush=True)
np.savez(f'{outdir}/float.npz', nll=Rf)
for n in names:
    np.savez(f"{outdir}/{n.replace('.tforge','')}.npz", **R[n])
    print(f"{n:20s} nll {R[n]['nll'].mean():.4f} (float {Rf.mean():.4f})  top1Agree {R[n]['ag'].mean()*100:6.2f}%  KL {R[n]['kl'].mean():.4f}", flush=True)

if os.path.exists(ck): os.remove(ck)
