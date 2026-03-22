import { SimulationBuffers } from '../../src/solver/buffers';
import { SimulationPipeline } from '../../src/solver/pipeline';
import { createConfig } from '../../src/solver/config';
import { getGridSize, getSpectralSize } from '../../src/solver/layout';
import { initPrecomputeBuffers } from '../../src/solver/precompute';

// Import raw WGSL strings
import fftForwardLonSrc from '../../src/shaders/fftForwardLon.wgsl?raw';
import fftInverseLonSrc from '../../src/shaders/fftInverseLon.wgsl?raw';
import legendreAnalysisSrc from '../../src/shaders/legendreAnalysis.wgsl?raw';
import legendreSynthesisSrc from '../../src/shaders/legendreSynthesis.wgsl?raw';
import legendreSynthesisDThetaSrc from '../../src/shaders/legendreSynthesisDTheta.wgsl?raw';

class TestRunner {
    device: GPUDevice | null = null;
    config: any;
    buffers: SimulationBuffers | null = null;
    pipeline: SimulationPipeline | null = null;

    async init(lmax = 31) {
        if (!navigator.gpu) throw new Error("WebGPU not supported");

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("No adapter found");

        this.device = await adapter.requestDevice();
        this.config = createConfig({ lmax });

        this.buffers = new SimulationBuffers(this.device, this.config);

        await initPrecomputeBuffers(this.device, this.config, this.buffers);

        this.pipeline = new SimulationPipeline(this.device, this.config, this.buffers);
        await this.pipeline.init(
            fftForwardLonSrc,
            fftInverseLonSrc,
            legendreAnalysisSrc,
            legendreSynthesisSrc,
            legendreSynthesisDThetaSrc
        );
        (window as any).TestRunnerReady = true;
    }

    uploadFloat32Array(buffer: GPUBuffer, data: Float32Array) {
        if (!this.device) return;
        this.device.queue.writeBuffer(buffer, 0, data);
    }

    async readFloat32Array(buffer: GPUBuffer, length: number): Promise<Float32Array> {
        if (!this.device) throw new Error("Device not initialized");

        const size = length * Float32Array.BYTES_PER_ELEMENT;
        const alignedSize = Math.ceil(size / 4) * 4;

        const readBuffer = this.device.createBuffer({
            size: alignedSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, alignedSize);
        this.device.queue.submit([commandEncoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const copyArrayBuffer = readBuffer.getMappedRange();

        const result = new Float32Array(length);
        result.set(new Float32Array(copyArrayBuffer).slice(0, length));

        readBuffer.unmap();
        readBuffer.destroy();
        return result;
    }

    async testFFTForward(inputGrid: Float32Array): Promise<Float32Array> {
        if (!this.device || !this.buffers || !this.pipeline || !this.config) throw new Error("Not initialized");
        this.uploadFloat32Array(this.buffers.zetaGrid, inputGrid);
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        this.pipeline.passFFTForward(passEncoder, this.buffers.zetaGrid, this.buffers.tmpLM);
        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
        const spectralSize = getSpectralSize(this.config) * 2;
        return this.readFloat32Array(this.buffers.tmpLM, spectralSize);
    }

    async testFFTInverse(inputFreq: Float32Array): Promise<Float32Array> {
        if (!this.device || !this.buffers || !this.pipeline || !this.config) throw new Error("Not initialized");
        this.uploadFloat32Array(this.buffers.tmpLM, inputFreq);
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        this.pipeline.passFFTInverse(passEncoder, this.buffers.tmpLM, this.buffers.zetaGrid);
        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
        const gridSize = getGridSize(this.config);
        return this.readFloat32Array(this.buffers.zetaGrid, gridSize);
    }

    async testLegendreAnalysis(inputFreq: Float32Array): Promise<Float32Array> {
        if (!this.device || !this.buffers || !this.pipeline || !this.config) throw new Error("Not initialized");
        this.uploadFloat32Array(this.buffers.tmpLM, inputFreq);
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        this.pipeline.passLegendreAnalysis(passEncoder, this.buffers.tmpLM, this.buffers.zetaLM_A, this.buffers.w, this.buffers.P_lm);
        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
        const spectralSize = getSpectralSize(this.config) * 2;
        return this.readFloat32Array(this.buffers.zetaLM_A, spectralSize);
    }

    async testLegendreSynthesis(inputLM: Float32Array): Promise<Float32Array> {
        if (!this.device || !this.buffers || !this.pipeline || !this.config) throw new Error("Not initialized");
        this.uploadFloat32Array(this.buffers.zetaLM_A, inputLM);
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        this.pipeline.passLegendreSynthesis(passEncoder, this.buffers.zetaLM_A, this.buffers.tmpLM, this.buffers.P_lm);
        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
        const spectralSize = getSpectralSize(this.config) * 2;
        return this.readFloat32Array(this.buffers.tmpLM, spectralSize);
    }

    async testLegendreSynthesisDTheta(inputLM: Float32Array): Promise<Float32Array> {
        if (!this.device || !this.buffers || !this.pipeline || !this.config) throw new Error("Not initialized");
        this.uploadFloat32Array(this.buffers.zetaLM_A, inputLM);
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        this.pipeline.passLegendreSynthesisDTheta(passEncoder, this.buffers.zetaLM_A, this.buffers.tmpLM, this.buffers.dP_lm_dtheta);
        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
        const spectralSize = getSpectralSize(this.config) * 2;
        return this.readFloat32Array(this.buffers.tmpLM, spectralSize);
    }
}

declare global {
    interface Window {
        testRunner: TestRunner;
        TestRunnerReady: boolean;
    }
}

(window as any).testRunner = new TestRunner();
(window as any).TestRunnerReady = false;
(window as any).testRunner.init().catch(console.error);
