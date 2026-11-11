import numpy as np, time, sys
sys.path.insert(0,'/home/claude/tf-exp'); from neval import *
ids=np.fromfile('data/val-sherlock.tok.u32',dtype=np.uint32).astype(np.int64)
for label,(cfg,P),ref in [('float',load_float('float.bin','float.json'),'ref-float-logits.f32'),('m1-row tforge',load_tforge('m1-row.tforge')[:2],'ref-m1row-logits.f32')]:
    m=Model(cfg,P); R=np.fromfile(ref,dtype=np.float32).reshape(16,-1); st=m.empty(1); md=0; agree=0; t0=time.time()
    for i in range(16):
        z,st=m.step(ids[i:i+1],st); d=np.abs(z[0]-R[i]).max()/ (np.abs(R[i]).max()); md=max(md,d); agree+=int(z[0].argmax()==R[i].argmax())
    print(f'{label}: max rel |dlogit| {md:.2e}, top1 agree {agree}/16, {16/(time.time()-t0):.1f} tok/s (B=1)')
cfg,P=load_float('float.bin','float.json'); m=Model(cfg,P)
for B in (16,32):
    st=m.empty(B); t0=time.time()
    for i in range(5): z,st=m.step(ids[i*B:(i+1)*B] if False else ids[:B]+i,st)
    print(f'B={B}: {5*B/(time.time()-t0):.1f} tok/s')
