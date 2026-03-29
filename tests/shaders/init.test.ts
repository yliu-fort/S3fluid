import { PipelineManager } from '../../src/solver/pipeline';

describe('Initialization Pipeline', () => {
    it('compiles without errors and setup properly', async () => {
        const mockDevice = {
            createShaderModule: jest.fn().mockReturnValue({}),
            createComputePipeline: jest.fn().mockReturnValue({}),
        } as unknown as GPUDevice;

        const manager = new PipelineManager(mockDevice);

        await manager.createComputePipeline('initRandom.wgsl');

        expect(mockDevice.createShaderModule).toHaveBeenCalledTimes(1);
        expect(mockDevice.createComputePipeline).toHaveBeenCalledTimes(1);
    });
});
