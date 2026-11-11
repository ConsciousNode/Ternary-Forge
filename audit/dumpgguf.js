const {loadForgeEngine}=require('/home/claude/ternary-forge/convert.js');const fs=require('fs');const TF=loadForgeEngine();
const [inp,outp]=process.argv.slice(2);const b=fs.readFileSync(inp);
const L=TF.loadRWKV7FromBuffer(b.buffer.slice(b.byteOffset,b.byteOffset+b.length),inp.split('/').pop(),{inference:true});
const d=TF.cloneOrDequantizeFloatModel(L.model);const fm=d.floatModel;console.log('quant',L.report.quant,'deq',d.wasDequantized,d.dequantizedFrom);
const idx={cfg:fm.cfg,t:{}};let off=0;const fd=fs.openSync(outp+'.bin','w');
for(const k of Object.keys(fm.p).sort()){const a=Float32Array.from(fm.p[k]);idx.t[k]={shape:fm.shape[k],off,n:a.length};fs.writeSync(fd,Buffer.from(a.buffer));off+=a.length*4;}
fs.closeSync(fd);fs.writeFileSync(outp+'.json',JSON.stringify(idx));
