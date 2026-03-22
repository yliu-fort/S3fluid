const fs = require('fs');
let code = fs.readFileSync('src/solver/pipeline.ts', 'utf-8');

const binds = `
    // newly added
    initRandomBindGroupLayout!: GPUBindGroupLayout;
    velocityFromPsiBindGroupLayout!: GPUBindGroupLayout;
    advectGridBindGroupLayout!: GPUBindGroupLayout;
    rhsComposeBindGroupLayout!: GPUBindGroupLayout;
    rk4StageBindGroupLayout!: GPUBindGroupLayout;
    rk4CombineBindGroupLayout!: GPUBindGroupLayout;
    energyIntegrandBindGroupLayout!: GPUBindGroupLayout;
    reduceSumBindGroupLayout!: GPUBindGroupLayout;

    initRandomPipeline!: GPUComputePipeline;
    velocityFromPsiPipeline!: GPUComputePipeline;
    advectGridPipeline!: GPUComputePipeline;
    rhsComposePipeline!: GPUComputePipeline;
    rk4StagePipeline!: GPUComputePipeline;
    rk4CombinePipeline!: GPUComputePipeline;
    energyIntegrandPipeline!: GPUComputePipeline;
    reduceSumPipeline!: GPUComputePipeline;
`;

code = code.replace('filterSpectrumPipeline!: GPUComputePipeline;', 'filterSpectrumPipeline!: GPUComputePipeline;' + binds);

const params = `
        this.filterSpectrumBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
            ]
        });

        this.initRandomBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // initSlope
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // specFilter
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // zetaLM
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
            ]
        });

        this.velocityFromPsiBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // dpsiDphi
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // dpsiDtheta
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // sinTheta
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // uTheta
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // uPhi
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
            ]
        });

        this.advectGridBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // uTheta
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // uPhi
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // dzetaDtheta
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // dzetaDphi
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // advGrid
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
            ]
        });

        this.rhsComposeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // advLM
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // zetaLM
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // lapEigs
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // rhsLM
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
            ]
        });

        this.rk4StageBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // z
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // k
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // zNext
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
            ]
        });

        this.rk4CombineBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // z
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // k1
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // k2
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // k3
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // k4
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // specFilter
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // zNext
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
            ]
        });

        this.energyIntegrandBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // psi
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // zeta
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // w
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // energyTerms
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
            ]
        });

        this.reduceSumBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // input
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // output
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
            ]
        });
`;

code = code.replace(
    /this\.filterSpectrumBindGroupLayout = this\.device\.createBindGroupLayout\(\{\n\s*entries: \[\n\s*\{ binding: 0, visibility: GPUShaderStage\.COMPUTE, buffer: \{ type: "read-only-storage" \} \},\n\s*\{ binding: 1, visibility: GPUShaderStage\.COMPUTE, buffer: \{ type: "read-only-storage" \} \},\n\s*\{ binding: 2, visibility: GPUShaderStage\.COMPUTE, buffer: \{ type: "storage" \} \},\n\s*\{ binding: 3, visibility: GPUShaderStage\.COMPUTE, buffer: \{ type: "uniform" \} \}\n\s*\]\n\s*\}\);/,
    params
);

const args = `
    async init(
        fftForwardCode: string,
        fftInverseCode: string,
        legendreAnalysisCode: string,
        legendreSynthesisCode: string,
        legendreSynthesisDThetaCode: string,
        mulIMCode: string,
        applyLaplacianCode: string,
        invertLaplacianCode: string,
        filterSpectrumCode: string,
        initRandomCode: string,
        velocityFromPsiCode: string,
        advectGridCode: string,
        rhsComposeCode: string,
        rk4StageCode: string,
        rk4CombineCode: string,
        energyIntegrandCode: string,
        reduceSumCode: string
    ) {
`;

code = code.replace(
    /async init\([\s\S]*?filterSpectrumCode: string\n\s*\) \{/,
    args
);


const modules = `
        this.filterSpectrumPipeline = await this.device.createComputePipelineAsync({
            label: "filterSpectrumPipeline",
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.filterSpectrumBindGroupLayout]
            }),
            compute: { module: filterSpectrumModule, entryPoint: "main" }
        });

        this.initRandomPipeline = await this.device.createComputePipelineAsync({
            label: "initRandomPipeline",
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.initRandomBindGroupLayout] }),
            compute: { module: this.device.createShaderModule({ label: "initRandom", code: initRandomCode }), entryPoint: "main" }
        });

        this.velocityFromPsiPipeline = await this.device.createComputePipelineAsync({
            label: "velocityFromPsiPipeline",
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.velocityFromPsiBindGroupLayout] }),
            compute: { module: this.device.createShaderModule({ label: "velocityFromPsi", code: velocityFromPsiCode }), entryPoint: "main" }
        });

        this.advectGridPipeline = await this.device.createComputePipelineAsync({
            label: "advectGridPipeline",
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.advectGridBindGroupLayout] }),
            compute: { module: this.device.createShaderModule({ label: "advectGrid", code: advectGridCode }), entryPoint: "main" }
        });

        this.rhsComposePipeline = await this.device.createComputePipelineAsync({
            label: "rhsComposePipeline",
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.rhsComposeBindGroupLayout] }),
            compute: { module: this.device.createShaderModule({ label: "rhsCompose", code: rhsComposeCode }), entryPoint: "main" }
        });

        this.rk4StagePipeline = await this.device.createComputePipelineAsync({
            label: "rk4StagePipeline",
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.rk4StageBindGroupLayout] }),
            compute: { module: this.device.createShaderModule({ label: "rk4Stage", code: rk4StageCode }), entryPoint: "main" }
        });

        this.rk4CombinePipeline = await this.device.createComputePipelineAsync({
            label: "rk4CombinePipeline",
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.rk4CombineBindGroupLayout] }),
            compute: { module: this.device.createShaderModule({ label: "rk4Combine", code: rk4CombineCode }), entryPoint: "main" }
        });

        this.energyIntegrandPipeline = await this.device.createComputePipelineAsync({
            label: "energyIntegrandPipeline",
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.energyIntegrandBindGroupLayout] }),
            compute: { module: this.device.createShaderModule({ label: "energyIntegrand", code: energyIntegrandCode }), entryPoint: "main" }
        });

        this.reduceSumPipeline = await this.device.createComputePipelineAsync({
            label: "reduceSumPipeline",
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.reduceSumBindGroupLayout] }),
            compute: { module: this.device.createShaderModule({ label: "reduceSum", code: reduceSumCode }), entryPoint: "main" }
        });
`;

code = code.replace(
    /this\.filterSpectrumPipeline = await this\.device\.createComputePipelineAsync\(\{\n\s*label: "filterSpectrumPipeline",\n\s*layout: this\.device\.createPipelineLayout\(\{\n\s*bindGroupLayouts: \[this\.filterSpectrumBindGroupLayout\]\n\s*\}\),\n\s*compute: \{\n\s*module: filterSpectrumModule,\n\s*entryPoint: "main"\n\s*\}\n\s*\}\);/,
    modules
);


const newPasses = `
    passInitRandom(pass: GPUComputePassEncoder, initSlopeBuffer: GPUBuffer, specFilterBuffer: GPUBuffer, zetaLMOut: GPUBuffer) {
        const bindGroup = this.device.createBindGroup({
            layout: this.initRandomBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: initSlopeBuffer } },
                { binding: 1, resource: { buffer: specFilterBuffer } },
                { binding: 2, resource: { buffer: zetaLMOut } },
                { binding: 3, resource: { buffer: this.paramsBuffer } }
            ]
        });
        pass.setPipeline(this.initRandomPipeline);
        pass.setBindGroup(0, bindGroup);
        const workgroupCountX = Math.ceil((this.config.lmax + 1) / 16);
        const workgroupCountY = Math.ceil((this.config.lmax + 1) / 16);
        pass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    }

    passVelocityFromPsi(pass: GPUComputePassEncoder, dpsiDphi: GPUBuffer, dpsiDtheta: GPUBuffer, sinTheta: GPUBuffer, uTheta: GPUBuffer, uPhi: GPUBuffer) {
        const bindGroup = this.device.createBindGroup({
            layout: this.velocityFromPsiBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: dpsiDphi } },
                { binding: 1, resource: { buffer: dpsiDtheta } },
                { binding: 2, resource: { buffer: sinTheta } },
                { binding: 3, resource: { buffer: uTheta } },
                { binding: 4, resource: { buffer: uPhi } },
                { binding: 5, resource: { buffer: this.paramsBuffer } }
            ]
        });
        pass.setPipeline(this.velocityFromPsiPipeline);
        pass.setBindGroup(0, bindGroup);
        const workgroupCount = Math.ceil(this.config.nlat / 64);
        pass.dispatchWorkgroups(workgroupCount);
    }

    passAdvectGrid(pass: GPUComputePassEncoder, uTheta: GPUBuffer, uPhi: GPUBuffer, dzetaDtheta: GPUBuffer, dzetaDphi: GPUBuffer, advGrid: GPUBuffer) {
        const bindGroup = this.device.createBindGroup({
            layout: this.advectGridBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: uTheta } },
                { binding: 1, resource: { buffer: uPhi } },
                { binding: 2, resource: { buffer: dzetaDtheta } },
                { binding: 3, resource: { buffer: dzetaDphi } },
                { binding: 4, resource: { buffer: advGrid } },
                { binding: 5, resource: { buffer: this.paramsBuffer } }
            ]
        });
        pass.setPipeline(this.advectGridPipeline);
        pass.setBindGroup(0, bindGroup);
        const workgroupCount = Math.ceil(this.config.nlat / 64);
        pass.dispatchWorkgroups(workgroupCount);
    }

    passRhsCompose(pass: GPUComputePassEncoder, advLM: GPUBuffer, zetaLM: GPUBuffer, lapEigs: GPUBuffer, rhsLM: GPUBuffer) {
        const bindGroup = this.device.createBindGroup({
            layout: this.rhsComposeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: advLM } },
                { binding: 1, resource: { buffer: zetaLM } },
                { binding: 2, resource: { buffer: lapEigs } },
                { binding: 3, resource: { buffer: rhsLM } },
                { binding: 4, resource: { buffer: this.paramsBuffer } }
            ]
        });
        pass.setPipeline(this.rhsComposePipeline);
        pass.setBindGroup(0, bindGroup);
        const workgroupCountX = Math.ceil((this.config.lmax + 1) / 16);
        const workgroupCountY = Math.ceil((this.config.lmax + 1) / 16);
        pass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    }

    passRk4Stage(pass: GPUComputePassEncoder, z: GPUBuffer, k: GPUBuffer, zNext: GPUBuffer, coeff: number) {
        const bindGroup = this.device.createBindGroup({
            layout: this.rk4StageBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: z } },
                { binding: 1, resource: { buffer: k } },
                { binding: 2, resource: { buffer: zNext } },
                { binding: 3, resource: { buffer: this.paramsBuffer } } // We need a way to pass coeff...
            ]
        });
        pass.setPipeline(this.rk4StagePipeline);
        pass.setBindGroup(0, bindGroup);
        const workgroupCountX = Math.ceil((this.config.lmax + 1) / 16);
        const workgroupCountY = Math.ceil((this.config.lmax + 1) / 16);
        pass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    }

    passRk4Combine(pass: GPUComputePassEncoder, z: GPUBuffer, k1: GPUBuffer, k2: GPUBuffer, k3: GPUBuffer, k4: GPUBuffer, specFilter: GPUBuffer, zNext: GPUBuffer) {
        const bindGroup = this.device.createBindGroup({
            layout: this.rk4CombineBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: z } },
                { binding: 1, resource: { buffer: k1 } },
                { binding: 2, resource: { buffer: k2 } },
                { binding: 3, resource: { buffer: k3 } },
                { binding: 4, resource: { buffer: k4 } },
                { binding: 5, resource: { buffer: specFilter } },
                { binding: 6, resource: { buffer: zNext } },
                { binding: 7, resource: { buffer: this.paramsBuffer } }
            ]
        });
        pass.setPipeline(this.rk4CombinePipeline);
        pass.setBindGroup(0, bindGroup);
        const workgroupCountX = Math.ceil((this.config.lmax + 1) / 16);
        const workgroupCountY = Math.ceil((this.config.lmax + 1) / 16);
        pass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    }

    passEnergyIntegrand(pass: GPUComputePassEncoder, psi: GPUBuffer, zeta: GPUBuffer, w: GPUBuffer, energyTerms: GPUBuffer) {
        const bindGroup = this.device.createBindGroup({
            layout: this.energyIntegrandBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: psi } },
                { binding: 1, resource: { buffer: zeta } },
                { binding: 2, resource: { buffer: w } },
                { binding: 3, resource: { buffer: energyTerms } },
                { binding: 4, resource: { buffer: this.paramsBuffer } }
            ]
        });
        pass.setPipeline(this.energyIntegrandPipeline);
        pass.setBindGroup(0, bindGroup);
        const workgroupCount = Math.ceil(this.config.nlat / 64);
        pass.dispatchWorkgroups(workgroupCount);
    }

    passReduceSum(pass: GPUComputePassEncoder, input: GPUBuffer, output: GPUBuffer, size: number) {
        const bindGroup = this.device.createBindGroup({
            layout: this.reduceSumBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: input } },
                { binding: 1, resource: { buffer: output } },
                { binding: 2, resource: { buffer: this.paramsBuffer } }
            ]
        });
        pass.setPipeline(this.reduceSumPipeline);
        pass.setBindGroup(0, bindGroup);
        const workgroupCount = Math.ceil(size / 256);
        pass.dispatchWorkgroups(workgroupCount);
    }
}
`;

code = code.replace(/}\s*$/, newPasses);

fs.writeFileSync('src/solver/pipeline.ts', code);
