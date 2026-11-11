#!/usr/bin/env node
'use strict';

/**
 * run-batch.js — Ternary Forge CLI: Grid of Methods × Settings -> results.jsonl
 * Enforces the 64-byte shared substring split leak guard across --calib, --val, and --test.
 * Includes references in every batch:
 *   1. the float model itself (method: "none")
 *   2. a random-code control (method: "random", same scales as M1, fixed seed)
 *   3. any supplied .gguf reference file(s)
 * Never selects, ranks, or hides methods in output; every configured run lands in results.jsonl.
 */

const fs = require('fs');
const path = require('path');
const { loadForgeEngine, parseArgs } = require('./convert.js');

function toArrayBuffer(buf) {
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

const DEFAULT_BATCH_GRID = [
  { method: 'absmean', granularity: 'tensor', scope: 'default' },
  { method: 'absmean', granularity: 'row', scope: 'default' },
  { method: 'absmean', granularity: 'group:32', scope: 'default' },
  { method: 'twn', granularity: 'row', scope: 'default', twnMode: 'closed' },
  { method: 'twn', granularity: 'row', scope: 'default', twnMode: 'search' },
  { method: 'act-aware', granularity: 'row', scope: 'default', actAlpha: 0.5 },
  { method: 'error-comp', granularity: 'row', scope: 'default', blockSize: 64, damping: 0.01 },
  { method: 'absmean', granularity: 'row', scope: 'default', heal: true, healSteps: 3, healLr: 1e-3, healSte: false }
];

function runBatch(opts) {
  const TF = loadForgeEngine();
  const modelPath = opts.model || opts.input || opts.checkpoint;
  const calibPath = opts.calib;
  const valPath = opts.val;
  const testPath = opts.test;
  const outPath = opts.out || opts.output || 'results.jsonl';

  if (!modelPath || !calibPath || !valPath || !testPath) {
    throw new Error('Usage: node run-batch.js --model <checkpoint> --calib <calib.txt> --val <val.txt> --test <test.txt> [--gguf <quant.gguf>] [--grid <grid.json>] [--split test|val] [--out results.jsonl]');
  }

  const calibBuf = fs.readFileSync(calibPath);
  const valBuf = fs.readFileSync(valPath);
  const testBuf = fs.readFileSync(testPath);

  // Refuse to run if any two of (--calib, --val, --test) share any 64-byte substring
  TF.checkSplitOverlap({
    calib: calibBuf,
    val: valBuf,
    test: testBuf
  });

  const calibText = calibBuf.toString('utf8');
  const valText = valBuf.toString('utf8');
  const testText = testBuf.toString('utf8');
  const calibSha256 = TF.sha256Hex(calibBuf);
  const valSha256 = TF.sha256Hex(valBuf);
  const testSha256 = TF.sha256Hex(testBuf);

  const evalSplitName = opts.split === 'val' ? 'val' : 'test';
  const evalText = evalSplitName === 'val' ? valText : testText;

  const modelBuf = fs.readFileSync(modelPath);
  const modelSha256 = TF.sha256Hex(modelBuf);
  const loaded = TF.loadAnyModelFromBuffer(toArrayBuffer(modelBuf), path.basename(modelPath), { inference: true });
  const deq = TF.cloneOrDequantizeFloatModel(loaded.model);
  const floatModel = deq.floatModel;

  // Determine grid of runs
  let userGrid = DEFAULT_BATCH_GRID;
  if (opts.grid) {
    if (Array.isArray(opts.grid)) {
      userGrid = opts.grid;
    } else if (typeof opts.grid === 'string') {
      const gridStr = fs.existsSync(opts.grid) ? fs.readFileSync(opts.grid, 'utf8') : opts.grid;
      userGrid = JSON.parse(gridStr);
    }
  }

  const seed = opts.seed !== undefined ? Number(opts.seed) : 42;
  const runs = [];

  // Reference 1: Float model itself (method: 'none') unless already first in userGrid
  if (!userGrid.some(g => g.method === 'none')) {
    runs.push({ method: 'none', granularity: 'tensor', scope: 'default', seed });
  }

  // Reference 2: Random-code control (same scales as M1, uniform random codes with fixed seed)
  if (!userGrid.some(g => g.method === 'random')) {
    const firstGran = (userGrid.find(g => g.method && g.method !== 'none') || {}).granularity || 'row';
    runs.push({ method: 'random', granularity: firstGran, scope: 'default', seed });
  }

  // Append all user-configured runs in exact order
  for (const item of userGrid) {
    runs.push({ seed, ...item });
  }

  const records = [];

  for (const runCfg of runs) {
    const converted = TF.convertRWKV7Model(floatModel, {
      ...runCfg,
      calibText,
      wasDequantized: deq.wasDequantized,
      dequantizedFrom: deq.dequantizedFrom
    });

    const fileBytes = runCfg.method === 'none'
      ? modelBuf.byteLength
      : TF.exportTernaryModelBuffer(converted).byteLength;

    const rec = TF.evaluateModel(converted, evalText, {
      floatModel,
      fileBytes,
      modelSha256,
      calibSha256,
      valSha256,
      testSha256
    });
    rec.evalSplit = evalSplitName;
    records.push(rec);
  }

  // Reference 3: When a .gguf of the same model is supplied, evaluate it in the same harness
  const ggufList = opts.gguf
    ? (Array.isArray(opts.gguf) ? opts.gguf : String(opts.gguf).split(',').map(s => s.trim()).filter(Boolean))
    : [];

  for (const ggufPath of ggufList) {
    const gBuf = fs.readFileSync(ggufPath);
    const gSha = TF.sha256Hex(gBuf);
    const gLoaded = TF.loadRWKV7FromBuffer(toArrayBuffer(gBuf), path.basename(ggufPath), { inference: true });
    const gQuant = gLoaded.report.quant || 'F32';
    gLoaded.model.conversionMeta = {
      method: `gguf:${gQuant}`,
      granularity: 'gguf-block',
      scope: 'gguf',
      keep: [],
      settings: { method: `gguf:${gQuant}`, file: path.basename(ggufPath), quant: gQuant }
    };
    const rec = TF.evaluateModel(gLoaded.model, evalText, {
      floatModel,
      fileBytes: gBuf.byteLength,
      modelSha256: gSha,
      calibSha256,
      valSha256,
      testSha256
    });
    rec.evalSplit = evalSplitName;
    records.push(rec);
  }

  const jsonl = records.map(r => JSON.stringify(r)).join('\n') + '\n';
  fs.writeFileSync(outPath, jsonl, 'utf8');
  return { outPath, count: records.length, records };
}

if (require.main === module) {
  try {
    const args = parseArgs(process.argv);
    const res = runBatch(args);
    if (!args.quiet) {
      process.stdout.write(`Wrote ${res.count} records to ${res.outPath}\n`);
    }
  } catch (err) {
    process.stderr.write(`Error: ${err.message}\n`);
    process.exit(1);
  }
}

module.exports = { runBatch, DEFAULT_BATCH_GRID };