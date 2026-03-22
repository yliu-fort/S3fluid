const fs = require('fs');
let code = fs.readFileSync('tests/playwright/runner.ts', 'utf-8');
console.log(code.match(/import initRandomSrc from '\.\.\/\.\.\/src\/shaders\/initRandom\.wgsl\?raw';/));
