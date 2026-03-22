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

    paramsBuffer: GPUBuffer;

    fftForwardBindGroupLayout: GPUBindGroupLayout;
    fftInverseBindGroupLayout: GPUBindGroupLayout;
    legendreAnalysisBindGroupLayout: GPUBindGroupLayout;
    legendreSynthesisBindGroupLayout: GPUBindGroupLayout;
    legendreSynthesisDThetaBindGroupLayout: GPUBindGroupLayout;

    constructor(device: GPUDevice, config: SimulationConfig, buffers: SimulationBuffers) {
        this.device = device;
        this.config = config;
        this.buffers = buffers;

        // Create params buffer
        this.paramsBuffer = this.device.createBuffer({
            label: "pipeline_params",
            size: 16, // 4 u32s: nlat, nlon, M, L
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

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
    }

    async init(
        fftForwardCode: string,
        fftInverseCode: string,
        legendreAnalysisCode: string,
        legendreSynthesisCode: string,
        legendreSynthesisDThetaCode: string
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
}
