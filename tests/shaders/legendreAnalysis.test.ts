import { PipelineManager } from '../../src/solver/pipeline';

describe('Legendre Analysis Pipeline', () => {
    it('compiles without errors and setup properly', async () => {
        const mockDevice = {
            createShaderModule: jest.fn().mockReturnValue({}),
            createComputePipeline: jest.fn().mockReturnValue({}),
        } as unknown as GPUDevice;

        const manager = new PipelineManager(mockDevice);

        await manager.createComputePipeline('legendreAnalysis.wgsl');
        expect(mockDevice.createShaderModule).toHaveBeenCalled();
        expect(mockDevice.createComputePipeline).toHaveBeenCalled();
    });
});
