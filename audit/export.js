const {loadForgeEngine}=require('/home/claude/ternary-forge/convert.js');const fs=require('fs');
const TF=loadForgeEngine();const O='/home/claude/tf-exp/models/';fs.mkdirSync(O,{recursive:true});
const b=fs.readFileSync('/home/claude/models/rwkv7-g1d-0.1b.pth');const L=TF.loadAnyModelFromBuffer(b.buffer.slice(b.byteOffset,b.byteOffset+b.length),'m.pth',{inference:true});
const fm=TF.cloneOrDequantizeFloatModel(L.model).floatModel;
const calib=process.env.CALIB?fs.readFileSync(process.env.CALIB,'utf8'):null;
const specs=JSON.parse(process.argv[2]);
for(const s of specs){const t0=Date.now();const cm=TF.convertRWKV7Model(fm,{...s,calibText:calib});const buf=TF.exportTernaryModelBuffer(cm);
 fs.writeFileSync(O+s.name+'.tforge',Buffer.from(buf));console.log(s.name,'bpw',cm.conversionMeta.bitsPerWeight.toFixed(4),'bytes',buf.byteLength,'ms',Date.now()-t0);}
