import { PipelineManager } from '../../src/solver/pipeline';
import { Diagnostics } from '../../src/solver/diagnostics';

describe('Diagnostics Pipeline', () => {
    it('compiles shaders without errors', async () => {
        const mockDevice = {
            createShaderModule: jest.fn().mockReturnValue({}),
            createComputePipeline: jest.fn().mockReturnValue({}),
        } as unknown as GPUDevice;

        const manager = new PipelineManager(mockDevice);

        await manager.createComputePipeline('energyIntegrand.wgsl');
        await manager.createComputePipeline('reduceSum.wgsl');

        expect(mockDevice.createShaderModule).toHaveBeenCalledTimes(2);
        expect(mockDevice.createComputePipeline).toHaveBeenCalledTimes(2);
    });
});

describe('Diagnostics Class', () => {
    it('records energy correctly', () => {
        const diag = new Diagnostics();
        diag.historyLength = 3;

        expect(diag.getLatestEnergy()).toBeNull();

        diag.recordEnergy(10);
        expect(diag.getLatestEnergy()).toBe(10);

        diag.recordEnergy(20);
        diag.recordEnergy(30);
        expect(diag.energyHistory).toEqual([10, 20, 30]);

        // check history trimming
        diag.recordEnergy(40);
        expect(diag.energyHistory).toEqual([20, 30, 40]);
        expect(diag.getLatestEnergy()).toBe(40);

        diag.reset();
        expect(diag.energyHistory).toEqual([]);
    });
});
