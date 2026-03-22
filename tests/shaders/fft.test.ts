import { PipelineManager } from '../../src/solver/pipeline';
import { SOLVER_CONFIG } from '../../src/solver/config';
import { generatePrecomputedData } from '../../src/solver/precompute';

describe('FFT Shaders Pipeline', () => {
    it('compiles without errors and setup properly', async () => {
        // Only run test if WebGPU is supported (requires setup in Jest, testing compilation logic via mock)

        // Let's create a dummy PipelineManager to ensure it can load shaders.
        const mockDevice = {
            createShaderModule: jest.fn().mockReturnValue({}),
            createComputePipeline: jest.fn().mockReturnValue({}),
        } as unknown as GPUDevice;

        const manager = new PipelineManager(mockDevice);

        // Forward FFT
        await manager.createComputePipeline('fftForwardLon.wgsl');
        expect(mockDevice.createShaderModule).toHaveBeenCalled();
        expect(mockDevice.createComputePipeline).toHaveBeenCalled();

        // Inverse FFT
        await manager.createComputePipeline('fftInverseLon.wgsl');
        expect(mockDevice.createShaderModule).toHaveBeenCalledTimes(2);
        expect(mockDevice.createComputePipeline).toHaveBeenCalledTimes(2);
    });
});
