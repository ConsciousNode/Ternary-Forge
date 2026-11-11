#!/usr/bin/env node
'use strict';

/**
 * eval.js — Ternary Forge CLI: Model File (.tforge / .pth / .safetensors / .gguf) + Text -> Metrics JSON
 * Zero external dependencies. Loads the canonical engine from forge.html.
 */

const fs = require('fs');
const path = require('path');
const { loadForgeEngine, parseArgs } = require('./convert.js');

function toArrayBuffer(buf) {
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

function runEval(opts) {
  const TF = loadForgeEngine();
  const modelPath = opts.model || opts.input;
  const textPath = opts.text || opts.test || opts.val;
  if (!modelPath || (!textPath && !opts.evalText)) {
    throw new Error('Usage: node eval.js --model <model.tforge|checkpoint> [--ref <float-checkpoint>] --text <eval.txt> [--calib <calib.txt>] [--val <val.txt>] [--test <test.txt>] [--out <metrics.json>]');
  }

  // Optional split overlap guard if multiple split files are supplied
  const calibBuf = opts.calib && fs.existsSync(opts.calib) ? fs.readFileSync(opts.calib) : null;
  const valBuf = opts.val && fs.existsSync(opts.val) ? fs.readFileSync(opts.val) : null;
  const testBuf = opts.test && fs.existsSync(opts.test) ? fs.readFileSync(opts.test) : null;
  if ([calibBuf, valBuf, testBuf].filter(Boolean).length >= 2) {
    TF.checkSplitOverlap({ calib: calibBuf, val: valBuf, test: testBuf });
  }

  const evalBuf = opts.evalText ? Buffer.from(opts.evalText, 'utf8') : fs.readFileSync(textPath);
  const evalText = evalBuf.toString('utf8');

  const modelBuf = fs.readFileSync(modelPath);
  const modelSha256 = TF.sha256Hex(modelBuf);
  const loaded = TF.loadAnyModelFromBuffer(toArrayBuffer(modelBuf), path.basename(modelPath), { inference: true });
  const model = loaded.model;

  let floatRefModel = model;
  if (opts.ref) {
    const refBuf = fs.readFileSync(opts.ref);
    const refLoaded = TF.loadAnyModelFromBuffer(toArrayBuffer(refBuf), path.basename(opts.ref), { inference: true });
    floatRefModel = TF.cloneOrDequantizeFloatModel(refLoaded.model).floatModel;
  }

  const record = TF.evaluateModel(model, evalText, {
    floatModel: floatRefModel,
    fileBytes: modelBuf.byteLength,
    modelSha256,
    calibSha256: calibBuf ? TF.sha256Hex(calibBuf) : null,
    valSha256: valBuf ? TF.sha256Hex(valBuf) : null,
    testSha256: testBuf ? TF.sha256Hex(testBuf) : TF.sha256Hex(evalBuf)
  });

  if (opts.out || opts.output) {
    fs.writeFileSync(opts.out || opts.output, JSON.stringify(record, null, 2) + '\n', 'utf8');
  }
  return record;
}

if (require.main === module) {
  try {
    const args = parseArgs(process.argv);
    const rec = runEval(args);
    if (!args.quiet) {
      process.stdout.write(JSON.stringify(rec, null, 2) + '\n');
    }
  } catch (err) {
    process.stderr.write(`Error: ${err.message}\n`);
    process.exit(1);
  }
}

module.exports = { runEval };