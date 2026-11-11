#!/usr/bin/env node
'use strict';

/**
 * convert.js — Ternary Forge CLI: Checkpoint (.pth / .safetensors / .gguf) -> Ternary Model (.tforge)
 * Zero external dependencies. Loads the canonical engine from forge.html.
 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');

function loadForgeEngine() {
  if (globalThis.TernaryForge) return globalThis.TernaryForge;
  const htmlPath = path.join(__dirname, 'forge.html');
  const html = fs.readFileSync(htmlPath, 'utf8');
  const vocabMatch = html.match(/<script[^>]*id="rwkv-world-vocab"[^>]*>([\s\S]*?)<\/script>/);
  const coreMatch = html.match(/<script[^>]*id="ternary-forge-core"[^>]*>([\s\S]*?)<\/script>/);
  if (!coreMatch) throw new Error('Could not find #ternary-forge-core in forge.html');
  globalThis.__RWKV_WORLD_VOCAB_JSON__ = vocabMatch ? vocabMatch[1] : '{"tokens":[]}';
  globalThis.__FORGE_CORE_SOURCE__ = coreMatch[1];
  vm.runInThisContext(coreMatch[1], { filename: 'forge.html#ternary-forge-core' });
  return globalThis.TernaryForge;
}

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const eq = a.indexOf('=');
      if (eq > 0) {
        args[a.slice(2, eq)] = a.slice(eq + 1);
      } else {
        const key = a.slice(2);
        const next = argv[i + 1];
        if (next !== undefined && !next.startsWith('--')) {
          args[key] = next;
          i++;
        } else {
          args[key] = true;
        }
      }
    } else if (a === '-i' && argv[i + 1]) {
      args.input = argv[++i];
    } else if (a === '-o' && argv[i + 1]) {
      args.output = argv[++i];
    } else if (!args.input) {
      args.input = a;
    } else if (!args.output) {
      args.output = a;
    }
  }
  return args;
}

function toArrayBuffer(buf) {
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

function runConvert(opts) {
  const TF = loadForgeEngine();
  const inputPath = opts.input || opts.model || opts.checkpoint;
  const outputPath = opts.output || opts.out;
  if (!inputPath || !outputPath) {
    throw new Error('Usage: node convert.js --input <checkpoint> --output <out.tforge> [--method absmean|twn|act-aware|error-comp|random|none] [--granularity tensor|row|group:32|group:64|group:128] [--scope default] [--keep t1,t2] [--calib calib.txt]');
  }

  const rawBuf = fs.readFileSync(inputPath);
  const ab = toArrayBuffer(rawBuf);
  const modelSha256 = TF.sha256Hex(rawBuf);
  const loaded = TF.loadAnyModelFromBuffer(ab, path.basename(inputPath), { inference: true });
  const deq = TF.cloneOrDequantizeFloatModel(loaded.model);

  let calibText = null;
  let calibSha256 = null;
  if (opts.calib) {
    const calibBuf = fs.readFileSync(opts.calib);
    calibText = calibBuf.toString('utf8');
    calibSha256 = TF.sha256Hex(calibBuf);
  } else if (opts.calibText) {
    calibText = opts.calibText;
    calibSha256 = TF.sha256Hex(calibText);
  }

  const convOpts = {
    method: opts.method || 'absmean',
    granularity: opts.granularity || 'row',
    scope: opts.scope || 'default',
    keep: opts.keep || '',
    seed: opts.seed !== undefined ? Number(opts.seed) : 42,
    twnMode: opts['twn-mode'] || opts.twnMode || 'closed',
    gridSteps: opts['grid-steps'] ? Number(opts['grid-steps']) : 100,
    actAlpha: opts.alpha !== undefined ? Number(opts.alpha) : (opts['act-alpha'] !== undefined ? Number(opts['act-alpha']) : 0.5),
    baseMethod: opts['base-method'] || opts.baseMethod || 'absmean',
    blockSize: opts['block-size'] !== undefined ? Number(opts['block-size']) : (opts.blockSize || 64),
    damping: opts.damping !== undefined ? Number(opts.damping) : 0.01,
    heal: !!(opts.heal && opts.heal !== 'false' && opts.heal !== '0'),
    healSteps: opts['heal-steps'] !== undefined ? Number(opts['heal-steps']) : (opts.healSteps || 5),
    healLr: opts['heal-lr'] !== undefined ? Number(opts['heal-lr']) : (opts.healLr || 1e-3),
    healSte: !!(opts['heal-ste'] || opts.healSte),
    healVectors: !!(opts['heal-vectors'] || opts.healVectors),
    calibText,
    wasDequantized: deq.wasDequantized,
    dequantizedFrom: deq.dequantizedFrom
  };

  const t0 = performance.now();
  const converted = TF.convertRWKV7Model(deq.floatModel, convOpts);
  const outAb = TF.exportTernaryModelBuffer(converted, {
    sourceFormat: loaded.report.format,
    modelSha256,
    calibSha256,
    codeSha256: TF.getCoreCodeSha256()
  });
  const elapsedMs = performance.now() - t0;

  fs.writeFileSync(outputPath, Buffer.from(outAb));
  return {
    outputPath,
    fileBytes: outAb.byteLength,
    elapsedMs,
    method: converted.conversionMeta.method,
    granularity: converted.conversionMeta.granularity,
    scope: converted.conversionMeta.scope,
    bitsPerWeight: converted.conversionMeta.bitsPerWeight,
    wasDequantized: converted.conversionMeta.wasDequantized,
    dequantizedFrom: converted.conversionMeta.dequantizedFrom,
    modelSha256,
    calibSha256,
    codeSha256: TF.getCoreCodeSha256()
  };
}

if (require.main === module) {
  try {
    const args = parseArgs(process.argv);
    const res = runConvert(args);
    if (!args.quiet) {
      process.stdout.write(JSON.stringify(res, null, 2) + '\n');
    }
  } catch (err) {
    process.stderr.write(`Error: ${err.message}\n`);
    process.exit(1);
  }
}

module.exports = { loadForgeEngine, runConvert, parseArgs };