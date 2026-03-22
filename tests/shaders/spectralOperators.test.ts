import { defaultConfig } from "../../src/solver/config";
import { SimulationBuffers } from "../../src/solver/buffers";
import { SimulationPipeline } from "../../src/solver/pipeline";
import * as fs from "fs";
import * as path from "path";

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

describe("SimulationPipeline Spectral Operator passes", () => {
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
        const dummyCode = "@compute @workgroup_size(256) fn main() {}";
        await pipeline.init(
            dummyCode, dummyCode, dummyCode, dummyCode, dummyCode,
            dummyCode, dummyCode, dummyCode, dummyCode
        );

        expect(pipeline.mulIMPipeline).toBeDefined();
        expect(pipeline.applyLaplacianPipeline).toBeDefined();
        expect(pipeline.invertLaplacianPipeline).toBeDefined();
        expect(pipeline.filterSpectrumPipeline).toBeDefined();
    });

    test("Dispatches mulIM pass", async () => {
        const dummyCode = "@compute @workgroup_size(256) fn main() {}";
        await pipeline.init(
            dummyCode, dummyCode, dummyCode, dummyCode, dummyCode,
            dummyCode, dummyCode, dummyCode, dummyCode
        );

        const mockPass = {
            setPipeline: jest.fn(),
            setBindGroup: jest.fn(),
            dispatchWorkgroups: jest.fn()
        } as unknown as GPUComputePassEncoder;

        const dummyBuffer = device.createBuffer({ size: 4, usage: 0 });

        pipeline.passMulIM(mockPass, dummyBuffer);

        expect(mockPass.setPipeline).toHaveBeenCalledWith(pipeline.mulIMPipeline);
        expect(mockPass.setBindGroup).toHaveBeenCalled();
        const L = defaultConfig.lmax + 1;
        const totalElements = L * L;
        expect(mockPass.dispatchWorkgroups).toHaveBeenCalledWith(Math.ceil(totalElements / 256));
    });

    test("Dispatches applyLaplacian pass", async () => {
        const dummyCode = "@compute @workgroup_size(256) fn main() {}";
        await pipeline.init(
            dummyCode, dummyCode, dummyCode, dummyCode, dummyCode,
            dummyCode, dummyCode, dummyCode, dummyCode
        );

        const mockPass = {
            setPipeline: jest.fn(),
            setBindGroup: jest.fn(),
            dispatchWorkgroups: jest.fn()
        } as unknown as GPUComputePassEncoder;

        const dummyBuffer = device.createBuffer({ size: 4, usage: 0 });

        pipeline.passApplyLaplacian(mockPass, dummyBuffer, dummyBuffer);

        expect(mockPass.setPipeline).toHaveBeenCalledWith(pipeline.applyLaplacianPipeline);
        expect(mockPass.setBindGroup).toHaveBeenCalled();
        const L = defaultConfig.lmax + 1;
        const totalElements = L * L;
        expect(mockPass.dispatchWorkgroups).toHaveBeenCalledWith(Math.ceil(totalElements / 256));
    });

    test("Dispatches invertLaplacian pass", async () => {
        const dummyCode = "@compute @workgroup_size(256) fn main() {}";
        await pipeline.init(
            dummyCode, dummyCode, dummyCode, dummyCode, dummyCode,
            dummyCode, dummyCode, dummyCode, dummyCode
        );

        const mockPass = {
            setPipeline: jest.fn(),
            setBindGroup: jest.fn(),
            dispatchWorkgroups: jest.fn()
        } as unknown as GPUComputePassEncoder;

        const dummyBuffer = device.createBuffer({ size: 4, usage: 0 });

        pipeline.passInvertLaplacian(mockPass, dummyBuffer, dummyBuffer);

        expect(mockPass.setPipeline).toHaveBeenCalledWith(pipeline.invertLaplacianPipeline);
        expect(mockPass.setBindGroup).toHaveBeenCalled();
        const L = defaultConfig.lmax + 1;
        const totalElements = L * L;
        expect(mockPass.dispatchWorkgroups).toHaveBeenCalledWith(Math.ceil(totalElements / 256));
    });

    test("Dispatches filterSpectrum pass", async () => {
        const dummyCode = "@compute @workgroup_size(256) fn main() {}";
        await pipeline.init(
            dummyCode, dummyCode, dummyCode, dummyCode, dummyCode,
            dummyCode, dummyCode, dummyCode, dummyCode
        );

        const mockPass = {
            setPipeline: jest.fn(),
            setBindGroup: jest.fn(),
            dispatchWorkgroups: jest.fn()
        } as unknown as GPUComputePassEncoder;

        const dummyBuffer = device.createBuffer({ size: 4, usage: 0 });

        pipeline.passFilterSpectrum(mockPass, dummyBuffer, dummyBuffer);

        expect(mockPass.setPipeline).toHaveBeenCalledWith(pipeline.filterSpectrumPipeline);
        expect(mockPass.setBindGroup).toHaveBeenCalled();
        const L = defaultConfig.lmax + 1;
        const totalElements = L * L;
        expect(mockPass.dispatchWorkgroups).toHaveBeenCalledWith(Math.ceil(totalElements / 256));
    });

    test("WGSL shaders are valid", () => {
        const shadersDir = path.join(__dirname, "../../src/shaders");

        const mulIMCode = fs.readFileSync(path.join(shadersDir, "mulIM.wgsl"), "utf-8");
        expect(mulIMCode).toContain("@compute");
        expect(mulIMCode).toContain("@workgroup_size(256)");
        expect(mulIMCode).toContain("-fm * val.y");

        const applyLaplacianCode = fs.readFileSync(path.join(shadersDir, "applyLaplacian.wgsl"), "utf-8");
        expect(applyLaplacianCode).toContain("@compute");
        expect(applyLaplacianCode).toContain("@workgroup_size(256)");
        expect(applyLaplacianCode).toContain("val.x * eig");

        const invertLaplacianCode = fs.readFileSync(path.join(shadersDir, "invertLaplacian.wgsl"), "utf-8");
        expect(invertLaplacianCode).toContain("@compute");
        expect(invertLaplacianCode).toContain("@workgroup_size(256)");
        expect(invertLaplacianCode).toContain("val.x / eig");
        expect(invertLaplacianCode).toContain("l == 0");

        const filterSpectrumCode = fs.readFileSync(path.join(shadersDir, "filterSpectrum.wgsl"), "utf-8");
        expect(filterSpectrumCode).toContain("@compute");
        expect(filterSpectrumCode).toContain("@workgroup_size(256)");
        expect(filterSpectrumCode).toContain("val.x * filter_val");
    });
});
