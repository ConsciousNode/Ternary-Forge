#!/usr/bin/env node
'use strict';

/**
 * tables.js — Ternary Forge CLI: results.jsonl -> Markdown tables (every number computed, none typed).
 * Preserves exact record order without ranking, filtering, or selecting winners.
 */

const fs = require('fs');
const { loadForgeEngine, parseArgs } = require('./convert.js');

function generateTablesFromJsonl(jsonlPath, outPath = null) {
  const TF = loadForgeEngine();
  const content = fs.readFileSync(jsonlPath, 'utf8');
  const lines = content.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  const records = lines.map((line, i) => {
    try {
      return JSON.parse(line);
    } catch (e) {
      throw new Error(`Invalid JSON on line ${i + 1} of ${jsonlPath}: ${e.message}`);
    }
  });

  const md = TF.formatResultsMarkdown(records);
  if (outPath) {
    fs.writeFileSync(outPath, md, 'utf8');
  }
  return md;
}

if (require.main === module) {
  try {
    const args = parseArgs(process.argv);
    const inFile = args.input || args.in || process.argv[2];
    if (!inFile || inFile.startsWith('--')) {
      throw new Error('Usage: node tables.js <results.jsonl> [--out <tables.md>]');
    }
    const md = generateTablesFromJsonl(inFile, args.out || args.output);
    if (!args.quiet) {
      process.stdout.write(md);
    }
  } catch (err) {
    process.stderr.write(`Error: ${err.message}\n`);
    process.exit(1);
  }
}

module.exports = { generateTablesFromJsonl };