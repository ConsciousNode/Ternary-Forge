const {loadForgeEngine}=require('/home/claude/ternary-forge/convert.js');const fs=require('fs');const TF=loadForgeEngine();
const b=fs.readFileSync('/home/claude/models/rwkv7-g1d-0.1b.pth');const L=TF.loadAnyModelFromBuffer(b.buffer.slice(b.byteOffset,b.byteOffset+b.length),'m.pth',{inference:true});
const fm=TF.cloneOrDequantizeFloatModel(L.model).floatModel;const ht=TF.tokenizeTextForModel(fs.readFileSync('data/calib-600.txt','utf8'),65536);
for(const [lr,steps] of JSON.parse(process.argv[2])){const tb=fs.readFileSync('models/m4-row.tforge');
 const {model:sm}=TF.importTernaryModelBuffer(tb.buffer.slice(tb.byteOffset,tb.byteOffset+tb.length));
 const t0=Date.now();const r=TF.runHealStage(fm,sm,ht,{healSteps:steps,healLr:lr,healSte:false});
 console.log('lr',lr,'steps',steps,((Date.now()-t0)/1000).toFixed(0)+'s',JSON.stringify(r).slice(0,160));
 fs.writeFileSync(`models/m4-row-heal-${lr}.tforge`,Buffer.from(TF.exportTernaryModelBuffer(sm)));}
