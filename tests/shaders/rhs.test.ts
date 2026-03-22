import { PipelineManager } from '../../src/solver/pipeline';

describe('RHS Pipeline', () => {
    it('compiles without errors and setup properly', async () => {
        const mockDevice = {
            createShaderModule: jest.fn().mockReturnValue({}),
            createComputePipeline: jest.fn().mockReturnValue({}),
        } as unknown as GPUDevice;

        const manager = new PipelineManager(mockDevice);

        await manager.createComputePipeline('velocityFromPsi.wgsl');
        await manager.createComputePipeline('advectGrid.wgsl');
        await manager.createComputePipeline('rhsCompose.wgsl');

        expect(mockDevice.createShaderModule).toHaveBeenCalledTimes(3);
        expect(mockDevice.createComputePipeline).toHaveBeenCalledTimes(3);
    });
});
