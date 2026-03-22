import { PipelineManager } from '../../src/solver/pipeline';

describe('Legendre Synthesis Pipeline', () => {
    it('compiles without errors and setup properly', async () => {
        const mockDevice = {
            createShaderModule: jest.fn().mockReturnValue({}),
            createComputePipeline: jest.fn().mockReturnValue({}),
        } as unknown as GPUDevice;

        const manager = new PipelineManager(mockDevice);

        await manager.createComputePipeline('legendreSynthesis.wgsl');
        await manager.createComputePipeline('legendreSynthesisDTheta.wgsl');

        expect(mockDevice.createShaderModule).toHaveBeenCalledTimes(2);
        expect(mockDevice.createComputePipeline).toHaveBeenCalledTimes(2);
    });
});
