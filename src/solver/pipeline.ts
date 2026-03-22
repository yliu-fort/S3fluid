import { SimulationConfig } from "./config";
import { SimulationBuffers } from "./buffers";

export class SimulationPipeline {
    device: GPUDevice;
    config: SimulationConfig;
    buffers: SimulationBuffers;

    fftForwardPipeline: GPUComputePipeline;
    fftInversePipeline: GPUComputePipeline;

    paramsBuffer: GPUBuffer;

    fftForwardBindGroupLayout: GPUBindGroupLayout;
    fftInverseBindGroupLayout: GPUBindGroupLayout;

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
    }

    async init(fftForwardCode: string, fftInverseCode: string) {
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
}
