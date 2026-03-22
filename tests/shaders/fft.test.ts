import { defaultConfig } from "../../src/solver/config";
import { SimulationBuffers } from "../../src/solver/buffers";
import { SimulationPipeline } from "../../src/solver/pipeline";

// Simple mock for GPUDevice and its related objects
class MockGPUBuffer {
    label: string;
    size: number;
    usage: number;
    constructor(descriptor: any) {
        this.label = descriptor.label;
        this.size = descriptor.size;
        this.usage = descriptor.usage;
    }
}

class MockGPUBindGroupLayout {
    label: string;
    entries: any;
    constructor(descriptor: any) {
        this.label = descriptor.label;
        this.entries = descriptor.entries;
    }
}

class MockGPUPipelineLayout {
    bindGroupLayouts: any;
    constructor(descriptor: any) {
        this.bindGroupLayouts = descriptor.bindGroupLayouts;
    }
}

class MockGPUShaderModule {
    label: string;
    code: string;
    constructor(descriptor: any) {
        this.label = descriptor.label;
        this.code = descriptor.code;
    }
}

class MockGPUComputePipeline {
    label: string;
    layout: any;
    compute: any;
    constructor(descriptor: any) {
        this.label = descriptor.label;
        this.layout = descriptor.layout;
        this.compute = descriptor.compute;
    }
}

class MockGPUDevice {
    queue = {
        writeBuffer: jest.fn()
    };

    createBuffer(descriptor: any) { return new MockGPUBuffer(descriptor) as unknown as GPUBuffer; }
    createBindGroupLayout(descriptor: any) { return new MockGPUBindGroupLayout(descriptor) as unknown as GPUBindGroupLayout; }
    createPipelineLayout(descriptor: any) { return new MockGPUPipelineLayout(descriptor) as unknown as GPUPipelineLayout; }
    createShaderModule(descriptor: any) { return new MockGPUShaderModule(descriptor) as unknown as GPUShaderModule; }
    createBindGroup(descriptor: any) { return {} as unknown as GPUBindGroup; }

    async createComputePipelineAsync(descriptor: any) {
        return new MockGPUComputePipeline(descriptor) as unknown as GPUComputePipeline;
    }
}

describe("SimulationPipeline FFT passes", () => {
    let device: GPUDevice;
    let buffers: SimulationBuffers;
    let pipeline: SimulationPipeline;

    beforeEach(() => {
        // Mock global GPUBufferUsage etc if not present
        if (!global.GPUBufferUsage) {
            (global as any).GPUBufferUsage = {
                MAP_READ: 1,
                MAP_WRITE: 2,
                COPY_SRC: 4,
                COPY_DST: 8,
                INDEX: 16,
                VERTEX: 32,
                UNIFORM: 64,
                STORAGE: 128,
                INDIRECT: 256,
                QUERY_RESOLVE: 512,
            };
        }
        if (!global.GPUShaderStage) {
            (global as any).GPUShaderStage = {
                VERTEX: 1,
                FRAGMENT: 2,
                COMPUTE: 4,
            };
        }

        device = new MockGPUDevice() as unknown as GPUDevice;
        buffers = new SimulationBuffers(device, defaultConfig);
        pipeline = new SimulationPipeline(device, defaultConfig, buffers);
    });

    test("Initializes pipelines", async () => {
        const dummyCode = "@compute @workgroup_size(64) fn main() {}";
        await pipeline.init(dummyCode, dummyCode);

        expect(pipeline.fftForwardPipeline).toBeDefined();
        expect(pipeline.fftInversePipeline).toBeDefined();
    });

    test("Dispatches FFT Forward pass", async () => {
        const dummyCode = "@compute @workgroup_size(64) fn main() {}";
        await pipeline.init(dummyCode, dummyCode);

        const mockPass = {
            setPipeline: jest.fn(),
            setBindGroup: jest.fn(),
            dispatchWorkgroups: jest.fn()
        } as unknown as GPUComputePassEncoder;

        pipeline.passFFTForward(mockPass, buffers.zetaGrid, buffers.tmpLM);

        expect(mockPass.setPipeline).toHaveBeenCalledWith(pipeline.fftForwardPipeline);
        expect(mockPass.setBindGroup).toHaveBeenCalled();
        expect(mockPass.dispatchWorkgroups).toHaveBeenCalledWith(Math.ceil(defaultConfig.nlat / 64));
    });

    test("Dispatches FFT Inverse pass", async () => {
        const dummyCode = "@compute @workgroup_size(64) fn main() {}";
        await pipeline.init(dummyCode, dummyCode);

        const mockPass = {
            setPipeline: jest.fn(),
            setBindGroup: jest.fn(),
            dispatchWorkgroups: jest.fn()
        } as unknown as GPUComputePassEncoder;

        pipeline.passFFTInverse(mockPass, buffers.tmpLM, buffers.zetaGrid);

        expect(mockPass.setPipeline).toHaveBeenCalledWith(pipeline.fftInversePipeline);
        expect(mockPass.setBindGroup).toHaveBeenCalled();
        expect(mockPass.dispatchWorkgroups).toHaveBeenCalledWith(Math.ceil(defaultConfig.nlat / 64));
    });
});
