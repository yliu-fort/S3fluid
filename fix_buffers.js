const fs = require('fs');
let code = fs.readFileSync('src/solver/buffers.ts', 'utf-8');
code = code.replace(/lapEigs!: GPUBuffer;\n    specFilter!: GPUBuffer;/, "lapEigs!: GPUBuffer;\n    specFilter!: GPUBuffer;\n    initSlope!: GPUBuffer;\n    sinTheta!: GPUBuffer;");
fs.writeFileSync('src/solver/buffers.ts', code);
