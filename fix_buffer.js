const fs = require('fs');
let code = fs.readFileSync('tests/playwright/runner.ts', 'utf-8');
code = code.replace(/result\.set\(new Float32Array\(copyArrayBuffer\)\.slice\(0, length\)\);/, 'result.set(new Float32Array(copyArrayBuffer as ArrayBuffer).slice(0, length));');
fs.writeFileSync('tests/playwright/runner.ts', code);
