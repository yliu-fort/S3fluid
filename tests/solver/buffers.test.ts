import { SimulationBuffers } from "../../src/solver/buffers";
import { createConfig } from "../../src/solver/config";
import { getGridSize, getSpectralSize } from "../../src/solver/layout";

describe("Buffer Allocations", () => {
    let mockDevice: GPUDevice;

    beforeAll(() => {
        (globalThis as any).GPUBufferUsage = {
            STORAGE: 0x0080,
            COPY_DST: 0x0008,
            COPY_SRC: 0x0004
        };
        (globalThis as any).GPUBuffer = class GPUBuffer {};
    });

    beforeEach(() => {
        mockDevice = {
            createBuffer: jest.fn().mockImplementation((desc) => {
                return { size: desc.size, label: desc.label, usage: desc.usage };
            }),
        } as unknown as GPUDevice;
    });

    it("should instantiate memory buffers properly", () => {
        const config = createConfig({ lmax: 31 });
        const buffers = new SimulationBuffers(mockDevice, config);

        const expectedSpectralBytes = getSpectralSize(config) * 8;
        const expectedGridBytes = getGridSize(config) * 4;

        expect(buffers.zetaLM_A).toEqual({
            label: "zetaLM_A",
            size: expectedSpectralBytes,
            usage: (globalThis as any).GPUBufferUsage.STORAGE | (globalThis as any).GPUBufferUsage.COPY_DST | (globalThis as any).GPUBufferUsage.COPY_SRC
        });

        expect(buffers.zetaGrid).toEqual({
            label: "zetaGrid",
            size: expectedGridBytes,
            usage: (globalThis as any).GPUBufferUsage.STORAGE | (globalThis as any).GPUBufferUsage.COPY_DST | (globalThis as any).GPUBufferUsage.COPY_SRC
        });

        expect(mockDevice.createBuffer).toHaveBeenCalledTimes(18);
    });
});
