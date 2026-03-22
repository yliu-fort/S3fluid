const fs = require('fs');
let code = fs.readFileSync('src/solver/pipeline.ts', 'utf-8');

// I need to add rk4CoeffBuffer
code = code.replace(
    /paramsBuffer!: GPUBuffer;/,
    "paramsBuffer!: GPUBuffer;\n    rk4CoeffBuffer!: GPUBuffer;"
);

code = code.replace(
    /this\.device\.createBuffer\(\{[\s\S]*?size: 4 \* 4,[\s\S]*?usage: GPUBufferUsage\.UNIFORM \| GPUBufferUsage\.COPY_DST[\s\S]*?\}\);/,
    "this.device.createBuffer({ size: 4 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });\n        this.rk4CoeffBuffer = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });"
);

code = code.replace(
    /this\.rk4StageBindGroupLayout = this\.device\.createBindGroupLayout\(\{[\s\S]*?entries: \[[\s\S]*?\{ binding: 3, visibility: GPUShaderStage\.COMPUTE, buffer: \{ type: "uniform" \} \}[\s\S]*?\][\s\S]*?\}\);/,
    "this.rk4StageBindGroupLayout = this.device.createBindGroupLayout({\n" +
    "            entries: [\n" +
    "                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: \"read-only-storage\" } }, // z\n" +
    "                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: \"read-only-storage\" } }, // k\n" +
    "                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: \"storage\" } }, // zNext\n" +
    "                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: \"uniform\" } },\n" +
    "                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: \"uniform\" } } // rk4Coeff\n" +
    "            ]\n" +
    "        });"
);

code = code.replace(
    /passRk4Stage\(pass: GPUComputePassEncoder, z: GPUBuffer, k: GPUBuffer, zNext: GPUBuffer, coeff: number\) \{[\s\S]*?const bindGroup = this\.device\.createBindGroup\(\{[\s\S]*?layout: this\.rk4StageBindGroupLayout,[\s\S]*?entries: \[[\s\S]*?\{ binding: 0, resource: \{ buffer: z \} \},[\s\S]*?\{ binding: 1, resource: \{ buffer: k \} \},[\s\S]*?\{ binding: 2, resource: \{ buffer: zNext \} \},[\s\S]*?\{ binding: 3, resource: \{ buffer: this\.paramsBuffer \} \}[\s\S]*?\][\s\S]*?\}\);/,
    "passRk4Stage(pass: GPUComputePassEncoder, z: GPUBuffer, k: GPUBuffer, zNext: GPUBuffer, coeff: number) {\n" +
    "        this.device.queue.writeBuffer(this.rk4CoeffBuffer, 0, new Float32Array([coeff]));\n" +
    "        const bindGroup = this.device.createBindGroup({\n" +
    "            layout: this.rk4StageBindGroupLayout,\n" +
    "            entries: [\n" +
    "                { binding: 0, resource: { buffer: z } },\n" +
    "                { binding: 1, resource: { buffer: k } },\n" +
    "                { binding: 2, resource: { buffer: zNext } },\n" +
    "                { binding: 3, resource: { buffer: this.paramsBuffer } },\n" +
    "                { binding: 4, resource: { buffer: this.rk4CoeffBuffer } }\n" +
    "            ]\n" +
    "        });"
);

fs.writeFileSync('src/solver/pipeline.ts', code);
