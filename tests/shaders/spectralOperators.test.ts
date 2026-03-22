import { PipelineManager } from '../../src/solver/pipeline';

describe('Spectral Operators Pipeline', () => {
    it('compiles without errors and setup properly', async () => {
        const mockDevice = {
            createShaderModule: jest.fn().mockReturnValue({}),
            createComputePipeline: jest.fn().mockReturnValue({}),
        } as unknown as GPUDevice;

        const manager = new PipelineManager(mockDevice);

        await manager.createComputePipeline('mulIM.wgsl');
        await manager.createComputePipeline('applyLaplacian.wgsl');
        await manager.createComputePipeline('invertLaplacian.wgsl');
        await manager.createComputePipeline('filterSpectrum.wgsl');

        expect(mockDevice.createShaderModule).toHaveBeenCalledTimes(4);
        expect(mockDevice.createComputePipeline).toHaveBeenCalledTimes(4);
    });
});
