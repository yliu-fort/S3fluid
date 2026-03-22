import { SOLVER_CONFIG } from '../../src/solver/config';
import { SolverBuffers } from '../../src/solver/buffers';
import { generatePrecomputedData } from '../../src/solver/precompute';

// Mock WebGPU globals for Jest
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

describe('SolverBuffers', () => {
    it('creates all storage buffers with correct sizes', () => {
        const mockDevice = {
            createBuffer: jest.fn().mockImplementation((desc) => ({ label: desc.label, size: desc.size })),
            queue: {
                writeBuffer: jest.fn()
            }
        } as unknown as GPUDevice;

        const buffers = new SolverBuffers(mockDevice);
        const precomputed = generatePrecomputedData(SOLVER_CONFIG.lmax, SOLVER_CONFIG.nlat, SOLVER_CONFIG.nlon);
        buffers.init(precomputed);

        // check correct sizes
        const spectralCount = (SOLVER_CONFIG.lmax + 1) * (SOLVER_CONFIG.lmax + 1) * 2;
        const gridCount = SOLVER_CONFIG.nlat * SOLVER_CONFIG.nlon;

        expect((buffers.zetaLM_A as any).size).toBe(spectralCount * 4);
        expect((buffers.zetaGrid as any).size).toBe(gridCount * 4);
        expect((buffers.P_lmBuffer as any).size).toBe(SOLVER_CONFIG.nlat * (SOLVER_CONFIG.lmax + 1) * (SOLVER_CONFIG.lmax + 1) * 4);

        expect(mockDevice.createBuffer).toHaveBeenCalled();
        expect(mockDevice.queue.writeBuffer).toHaveBeenCalled();
    });
});
