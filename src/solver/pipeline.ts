import type { SimulationConfig } from "./config";
import { SimulationBuffers } from "./buffers";

export class SimulationPipeline {
    device: GPUDevice;
    config: SimulationConfig;
    buffers: SimulationBuffers;

    fftForwardPipeline: GPUComputePipeline;
    fftInversePipeline: GPUComputePipeline;
    legendreAnalysisPipeline: GPUComputePipeline;
    legendreSynthesisPipeline: GPUComputePipeline;
    legendreSynthesisDThetaPipeline: GPUComputePipeline;
    mulIMPipeline: GPUComputePipeline;
    applyLaplacianPipeline: GPUComputePipeline;
    invertLaplacianPipeline: GPUComputePipeline;
    filterSpectrumPipeline: GPUComputePipeline;

    initRandomBindGroupLayout: GPUBindGroupLayout;
    velocityFromPsiBindGroupLayout: GPUBindGroupLayout;
    advectGridBindGroupLayout: GPUBindGroupLayout;
    rhsComposeBindGroupLayout: GPUBindGroupLayout;
    rk4StageBindGroupLayout: GPUBindGroupLayout;
    rk4CombineBindGroupLayout: GPUBindGroupLayout;
    energyIntegrandBindGroupLayout: GPUBindGroupLayout;
    reduceSumBindGroupLayout: GPUBindGroupLayout;

    initRandomPipeline: GPUComputePipeline;
    velocityFromPsiPipeline: GPUComputePipeline;
    advectGridPipeline: GPUComputePipeline;
    rhsComposePipeline: GPUComputePipeline;
    rk4StagePipeline: GPUComputePipeline;
    rk4CombinePipeline: GPUComputePipeline;
    energyIntegrandPipeline: GPUComputePipeline;
    reduceSumPipeline: GPUComputePipeline;

    paramsBuffer: GPUBuffer;
    rk4CoeffBuffer: GPUBuffer;

    fftForwardBindGroupLayout: GPUBindGroupLayout;
    fftInverseBindGroupLayout: GPUBindGroupLayout;
    legendreAnalysisBindGroupLayout: GPUBindGroupLayout;
    legendreSynthesisBindGroupLayout: GPUBindGroupLayout;
    legendreSynthesisDThetaBindGroupLayout: GPUBindGroupLayout;
    mulIMBindGroupLayout: GPUBindGroupLayout;
    applyLaplacianBindGroupLayout: GPUBindGroupLayout;
    invertLaplacianBindGroupLayout: GPUBindGroupLayout;
    filterSpectrumBindGroupLayout: GPUBindGroupLayout;

    constructor(device: GPUDevice, config: SimulationConfig, buffers: SimulationBuffers) {
        this.device = device;
        this.config = config;
        this.buffers = buffers;

        // Create params buffer
        this.paramsBuffer = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.rk4CoeffBuffer = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

        const paramsData = new Uint32Array([
            this.config.nlat,
            this.config.nlon,
            this.config.lmax + 1, // M
            this.config.lmax + 1  // L
        ]);
        this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);

        // FFT Forward layout
        this.fftForwardBindGroupLayout = this.device.createBindGroupLayout({
            label: "fftForwardBindGroupLayout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
            ]
        });

        // FFT Inverse layout
        this.fftInverseBindGroupLayout = this.device.createBindGroupLayout({
            label: "fftInverseBindGroupLayout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
            ]
        });

        // Legendre Analysis layout
        this.legendreAnalysisBindGroupLayout = this.device.createBindGroupLayout({
            label: "legendreAnalysisBindGroupLayout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
            ]
        });

        // Legendre Synthesis layout
        this.legendreSynthesisBindGroupLayout = this.device.createBindGroupLayout({
            label: "legendreSynthesisBindGroupLayout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
            ]
        });

        // Legendre Synthesis DTheta layout
        this.legendreSynthesisDThetaBindGroupLayout = this.device.createBindGroupLayout({
            label: "legendreSynthesisDThetaBindGroupLayout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
            ]
        });

        // mulIM layout
        this.mulIMBindGroupLayout = this.device.createBindGroupLayout({
            label: "mulIMBindGroupLayout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
            ]
        });

        // applyLaplacian layout
        this.applyLaplacianBindGroupLayout = this.device.createBindGroupLayout({
            label: "applyLaplacianBindGroupLayout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
            ]
        });

        // invertLaplacian layout
        this.invertLaplacianBindGroupLayout = this.device.createBindGroupLayout({
            label: "invertLaplacianBindGroupLayout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
            ]
        });

        // filterSpectrum layout
        this.filterSpectrumBindGroupLayout = this.device.createBindGroupLayout({
            label: "filterSpectrumBindGroupLayout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
            ]
        });
    }


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

        const fftForwardModule = this.device.createShaderModule({
            label: "fftForwardLon",
            code: fftForwardCode
        });

        this.fftForwardPipeline = await this.device.createComputePipelineAsync({
            label: "fftForwardPipeline",
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.fftForwardBindGroupLayout]
            }),
            compute: {
                module: fftForwardModule,
                entryPoint: "main"
            }
        });

        const fftInverseModule = this.device.createShaderModule({
            label: "fftInverseLon",
            code: fftInverseCode
        });

        this.fftInversePipeline = await this.device.createComputePipelineAsync({
            label: "fftInversePipeline",
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.fftInverseBindGroupLayout]
            }),
            compute: {
                module: fftInverseModule,
                entryPoint: "main"
            }
        });

        const legendreAnalysisModule = this.device.createShaderModule({
            label: "legendreAnalysis",
            code: legendreAnalysisCode
        });

        this.legendreAnalysisPipeline = await this.device.createComputePipelineAsync({
            label: "legendreAnalysisPipeline",
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.legendreAnalysisBindGroupLayout]
            }),
            compute: {
                module: legendreAnalysisModule,
                entryPoint: "main"
            }
        });

        const legendreSynthesisModule = this.device.createShaderModule({
            label: "legendreSynthesis",
            code: legendreSynthesisCode
        });

        this.legendreSynthesisPipeline = await this.device.createComputePipelineAsync({
            label: "legendreSynthesisPipeline",
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.legendreSynthesisBindGroupLayout]
            }),
            compute: {
                module: legendreSynthesisModule,
                entryPoint: "main"
            }
        });

        const legendreSynthesisDThetaModule = this.device.createShaderModule({
            label: "legendreSynthesisDTheta",
            code: legendreSynthesisDThetaCode
        });

        this.legendreSynthesisDThetaPipeline = await this.device.createComputePipelineAsync({
            label: "legendreSynthesisDThetaPipeline",
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.legendreSynthesisDThetaBindGroupLayout]
            }),
            compute: {
                module: legendreSynthesisDThetaModule,
                entryPoint: "main"
            }
        });

        const mulIMModule = this.device.createShaderModule({
            label: "mulIM",
            code: mulIMCode
        });

        this.mulIMPipeline = await this.device.createComputePipelineAsync({
            label: "mulIMPipeline",
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.mulIMBindGroupLayout]
            }),
            compute: {
                module: mulIMModule,
                entryPoint: "main"
            }
        });

        const applyLaplacianModule = this.device.createShaderModule({
            label: "applyLaplacian",
            code: applyLaplacianCode
        });

        this.applyLaplacianPipeline = await this.device.createComputePipelineAsync({
            label: "applyLaplacianPipeline",
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.applyLaplacianBindGroupLayout]
            }),
            compute: {
                module: applyLaplacianModule,
                entryPoint: "main"
            }
        });

        const invertLaplacianModule = this.device.createShaderModule({
            label: "invertLaplacian",
            code: invertLaplacianCode
        });

        this.invertLaplacianPipeline = await this.device.createComputePipelineAsync({
            label: "invertLaplacianPipeline",
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.invertLaplacianBindGroupLayout]
            }),
            compute: {
                module: invertLaplacianModule,
                entryPoint: "main"
            }
        });

        const filterSpectrumModule = this.device.createShaderModule({
            label: "filterSpectrum",
            code: filterSpectrumCode
        });


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

    }

    passFFTForward(pass: GPUComputePassEncoder, gridIn: GPUBuffer, freqOut: GPUBuffer) {
        const bindGroup = this.device.createBindGroup({
            layout: this.fftForwardBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: gridIn } },
                { binding: 1, resource: { buffer: freqOut } },
                { binding: 2, resource: { buffer: this.paramsBuffer } }
            ]
        });

        pass.setPipeline(this.fftForwardPipeline);
        pass.setBindGroup(0, bindGroup);
        // Workgroup size is 64, we need nlat threads
        const workgroupCount = Math.ceil(this.config.nlat / 64);
        pass.dispatchWorkgroups(workgroupCount);
    }

    passFFTInverse(pass: GPUComputePassEncoder, freqIn: GPUBuffer, gridOut: GPUBuffer) {
        const bindGroup = this.device.createBindGroup({
            layout: this.fftInverseBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: freqIn } },
                { binding: 1, resource: { buffer: gridOut } },
                { binding: 2, resource: { buffer: this.paramsBuffer } }
            ]
        });

        pass.setPipeline(this.fftInversePipeline);
        pass.setBindGroup(0, bindGroup);
        const workgroupCount = Math.ceil(this.config.nlat / 64);
        pass.dispatchWorkgroups(workgroupCount);
    }

    passLegendreAnalysis(pass: GPUComputePassEncoder, freqIn: GPUBuffer, wBuffer: GPUBuffer, pLmBuffer: GPUBuffer, aLmOut: GPUBuffer) {
        const bindGroup = this.device.createBindGroup({
            layout: this.legendreAnalysisBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: freqIn } },
                { binding: 1, resource: { buffer: wBuffer } },
                { binding: 2, resource: { buffer: pLmBuffer } },
                { binding: 3, resource: { buffer: aLmOut } },
                { binding: 4, resource: { buffer: this.paramsBuffer } }
            ]
        });

        pass.setPipeline(this.legendreAnalysisPipeline);
        pass.setBindGroup(0, bindGroup);
        // Workgroup size is 16x16, we need M x L threads
        const workgroupCountX = Math.ceil((this.config.lmax + 1) / 16);
        const workgroupCountY = Math.ceil((this.config.lmax + 1) / 16);
        pass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    }

    passLegendreSynthesis(pass: GPUComputePassEncoder, aLmIn: GPUBuffer, pLmBuffer: GPUBuffer, freqOut: GPUBuffer) {
        const bindGroup = this.device.createBindGroup({
            layout: this.legendreSynthesisBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: aLmIn } },
                { binding: 1, resource: { buffer: pLmBuffer } },
                { binding: 2, resource: { buffer: freqOut } },
                { binding: 3, resource: { buffer: this.paramsBuffer } }
            ]
        });

        pass.setPipeline(this.legendreSynthesisPipeline);
        pass.setBindGroup(0, bindGroup);
        // Workgroup size is 16x16, we need nlat x M threads
        const workgroupCountX = Math.ceil(this.config.nlat / 16);
        const workgroupCountY = Math.ceil((this.config.lmax + 1) / 16);
        pass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    }

    passLegendreSynthesisDTheta(pass: GPUComputePassEncoder, aLmIn: GPUBuffer, dpLmDThetaBuffer: GPUBuffer, freqOut: GPUBuffer) {
        const bindGroup = this.device.createBindGroup({
            layout: this.legendreSynthesisDThetaBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: aLmIn } },
                { binding: 1, resource: { buffer: dpLmDThetaBuffer } },
                { binding: 2, resource: { buffer: freqOut } },
                { binding: 3, resource: { buffer: this.paramsBuffer } }
            ]
        });

        pass.setPipeline(this.legendreSynthesisDThetaPipeline);
        pass.setBindGroup(0, bindGroup);
        const workgroupCountX = Math.ceil(this.config.nlat / 16);
        const workgroupCountY = Math.ceil((this.config.lmax + 1) / 16);
        pass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    }

    passMulIM(pass: GPUComputePassEncoder, aLmIn: GPUBuffer, aLmOut: GPUBuffer) {
        const bindGroup = this.device.createBindGroup({
            layout: this.mulIMBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: aLmIn } },
                { binding: 1, resource: { buffer: aLmOut } },
                { binding: 2, resource: { buffer: this.paramsBuffer } }
            ]
        });

        pass.setPipeline(this.mulIMPipeline);
        pass.setBindGroup(0, bindGroup);
        const workgroupCountX = Math.ceil((this.config.lmax + 1) / 16);
        const workgroupCountY = Math.ceil((this.config.lmax + 1) / 16);
        pass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    }

    passApplyLaplacian(pass: GPUComputePassEncoder, aLmIn: GPUBuffer, lapEigsBuffer: GPUBuffer, aLmOut: GPUBuffer) {
        const bindGroup = this.device.createBindGroup({
            layout: this.applyLaplacianBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: aLmIn } },
                { binding: 1, resource: { buffer: lapEigsBuffer } },
                { binding: 2, resource: { buffer: aLmOut } },
                { binding: 3, resource: { buffer: this.paramsBuffer } }
            ]
        });

        pass.setPipeline(this.applyLaplacianPipeline);
        pass.setBindGroup(0, bindGroup);
        const workgroupCountX = Math.ceil((this.config.lmax + 1) / 16);
        const workgroupCountY = Math.ceil((this.config.lmax + 1) / 16);
        pass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    }

    passInvertLaplacian(pass: GPUComputePassEncoder, aLmIn: GPUBuffer, lapEigsBuffer: GPUBuffer, aLmOut: GPUBuffer) {
        const bindGroup = this.device.createBindGroup({
            layout: this.invertLaplacianBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: aLmIn } },
                { binding: 1, resource: { buffer: lapEigsBuffer } },
                { binding: 2, resource: { buffer: aLmOut } },
                { binding: 3, resource: { buffer: this.paramsBuffer } }
            ]
        });

        pass.setPipeline(this.invertLaplacianPipeline);
        pass.setBindGroup(0, bindGroup);
        const workgroupCountX = Math.ceil((this.config.lmax + 1) / 16);
        const workgroupCountY = Math.ceil((this.config.lmax + 1) / 16);
        pass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    }

    passFilterSpectrum(pass: GPUComputePassEncoder, aLmIn: GPUBuffer, specFilterBuffer: GPUBuffer, aLmOut: GPUBuffer) {
        const bindGroup = this.device.createBindGroup({
            layout: this.filterSpectrumBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: aLmIn } },
                { binding: 1, resource: { buffer: specFilterBuffer } },
                { binding: 2, resource: { buffer: aLmOut } },
                { binding: 3, resource: { buffer: this.paramsBuffer } }
            ]
        });

        pass.setPipeline(this.filterSpectrumPipeline);
        pass.setBindGroup(0, bindGroup);
        const workgroupCountX = Math.ceil((this.config.lmax + 1) / 16);
        const workgroupCountY = Math.ceil((this.config.lmax + 1) / 16);
        pass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    }

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
        this.device.queue.writeBuffer(this.rk4CoeffBuffer, 0, new Float32Array([coeff]));
        const bindGroup = this.device.createBindGroup({
            layout: this.rk4StageBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: z } },
                { binding: 1, resource: { buffer: k } },
                { binding: 2, resource: { buffer: zNext } },
                { binding: 3, resource: { buffer: this.paramsBuffer } },
                { binding: 4, resource: { buffer: this.rk4CoeffBuffer } }
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
