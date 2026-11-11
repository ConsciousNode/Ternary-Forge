const {loadForgeEngine}=require('/home/claude/ternary-forge/convert.js');const fs=require('fs');
const TF=loadForgeEngine();const O='/home/claude/tf-exp/models/';
const b=fs.readFileSync('/home/claude/models/rwkv7-g1d-0.1b.pth');const L=TF.loadAnyModelFromBuffer(b.buffer.slice(b.byteOffset,b.byteOffset+b.length),'m.pth',{inference:true});
const fm=TF.cloneOrDequantizeFloatModel(L.model).floatModel;
const calib=fs.readFileSync('/home/claude/tf-exp/data/calib-4k.txt','utf8');const healC=fs.readFileSync('/home/claude/tf-exp/data/calib-600.txt','utf8');
for(const s of JSON.parse(process.argv[2])){const t0=Date.now();
 const cm=TF.convertRWKV7Model(fm,{...s,calibText:calib});console.log(s.name,'converted',((Date.now()-t0)/1000).toFixed(0)+'s');
 fs.writeFileSync(O+s.name+'.tforge',Buffer.from(TF.exportTernaryModelBuffer(cm)));
 if(s.alsoHeal){const t1=Date.now();const ht=TF.tokenizeTextForModel(healC,65536);
  const r=TF.runHealStage(fm,cm,ht,{healSteps:s.alsoHeal,healLr:1e-3,healSte:false});console.log(' heal',ht.length,'tokens',((Date.now()-t1)/1000).toFixed(0)+'s',JSON.stringify(r&&r.losses?r.losses:r).slice(0,200));
  fs.writeFileSync(O+s.name+'-heal.tforge',Buffer.from(TF.exportTernaryModelBuffer(cm)));}
 console.log(' bpw',cm.conversionMeta.bitsPerWeight.toFixed(4));}
