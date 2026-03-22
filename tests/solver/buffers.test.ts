import { SimulationBuffers } from "../../src/solver/buffers";
import { createConfig } from "../../src/solver/config";
import { getGridSize, getSpectralSize } from "../../src/solver/layout";

describe("Buffer Allocations", () => {
    let mockDevice: GPUDevice;

    beforeAll(() => {
        // mock GPUBufferUsage enum missing in JSDom environment
        (global as any).GPUBufferUsage = {
            STORAGE: 0x0080,
            COPY_DST: 0x0008,
            COPY_SRC: 0x0004
        };
    });

    beforeEach(() => {
        // create a minimal mock for GPUDevice so we don't need headless-gl webgpu
        mockDevice = {
            createBuffer: jest.fn().mockImplementation((desc) => {
                return { size: desc.size, label: desc.label, usage: desc.usage };
            }),
        } as unknown as GPUDevice;
    });

    it("should instantiate memory buffers properly", () => {
        const config = createConfig({ lmax: 31 });
        const buffers = new SimulationBuffers(mockDevice, config);

        // Calculate expected sizes in bytes based on vec2<f32> complex numbers
        const expectedSpectralBytes = getSpectralSize(config) * 8; // 8 bytes for vec2<f32>
        const expectedGridBytes = getGridSize(config) * 4;       // 4 bytes for f32

        expect(buffers.zetaLM_A).toEqual({
            label: "zetaLM_A",
            size: expectedSpectralBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });

        expect(buffers.zetaGrid).toEqual({
            label: "zetaGrid",
            size: expectedGridBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        // Test mock device received exact count of allocations
        expect(mockDevice.createBuffer).toHaveBeenCalledTimes(18); // 8 spectral + 10 grid
    });
});
