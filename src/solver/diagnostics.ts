import { SimulationConfig } from "./config";
import { SimulationBuffers } from "./buffers";
import { SimulationPipeline } from "./pipeline";
import { getGridSize } from "./layout";

export class SimulationDiagnostics {
    device: GPUDevice;
    config: SimulationConfig;
    buffers: SimulationBuffers;
    pipeline: SimulationPipeline;

    energyHistory: number[] = [];
    energyReadBuffer!: GPUBuffer;

    // For reduction
    reductionBuffers: GPUBuffer[] = [];

    constructor(device: GPUDevice, config: SimulationConfig, buffers: SimulationBuffers, pipeline: SimulationPipeline) {
        this.device = device;
        this.config = config;
        this.buffers = buffers;
        this.pipeline = pipeline;

        this.energyReadBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        this.initReductionBuffers();
    }

    private initReductionBuffers() {
        const gridSize = getGridSize(this.config);
        let currentSize = gridSize;

        while (currentSize > 1) {
            const nextSize = Math.ceil(currentSize / 256);
            const buffer = this.device.createBuffer({
                size: nextSize * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
            });
            this.reductionBuffers.push(buffer);
            currentSize = nextSize;
        }
    }

    synthesizeZetaAndPsi(passEncoder: GPUComputePassEncoder, zetaLMBuffer: GPUBuffer) {
        // synthesis(zetaLM) -> freq -> iFFT -> zetaGrid
        this.pipeline.passLegendreSynthesis(passEncoder, zetaLMBuffer, this.buffers.P_lm, this.buffers.tmpLM);
        this.pipeline.passFFTInverse(passEncoder, this.buffers.tmpLM, this.buffers.zetaGrid);

        // invertLaplacian(zetaLM) -> psiLM
        this.pipeline.passInvertLaplacian(passEncoder, zetaLMBuffer, this.buffers.lapEigs, this.buffers.psiLM);

        // synthesis(psiLM) -> freq -> iFFT -> psiGrid
        this.pipeline.passLegendreSynthesis(passEncoder, this.buffers.psiLM, this.buffers.P_lm, this.buffers.tmpLM);
        this.pipeline.passFFTInverse(passEncoder, this.buffers.tmpLM, this.buffers.psiGrid);
    }

    computeEnergy(passEncoder: GPUComputePassEncoder) {
        // compute local energy terms
        this.pipeline.passEnergyIntegrand(
            passEncoder,
            this.buffers.psiGrid,
            this.buffers.zetaGrid,
            this.buffers.w,
            this.buffers.energyTerms
        );

        // Reduce sum
        let currentInput = this.buffers.energyTerms;
        let currentSize = getGridSize(this.config);

        for (let i = 0; i < this.reductionBuffers.length; i++) {
            const currentOutput = this.reductionBuffers[i];
            this.pipeline.passReduceSum(passEncoder, currentInput, currentOutput, currentSize);
            currentInput = currentOutput;
            currentSize = Math.ceil(currentSize / 256);
        }
    }

    async readEnergyAsync(): Promise<number> {
        const lastBuffer = this.reductionBuffers[this.reductionBuffers.length - 1];

        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(lastBuffer, 0, this.energyReadBuffer, 0, 4);
        this.device.queue.submit([commandEncoder.finish()]);

        await this.energyReadBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = this.energyReadBuffer.getMappedRange();
        const value = new Float32Array(arrayBuffer)[0];
        this.energyReadBuffer.unmap();

        // Integrate with 2pi / nlon
        const energy = value * (2.0 * Math.PI / this.config.nlon);
        this.energyHistory.push(energy);

        if (this.energyHistory.length > 500) {
            this.energyHistory.shift();
        }

        return energy;
    }

    reset() {
        this.energyHistory = [];
    }
}
