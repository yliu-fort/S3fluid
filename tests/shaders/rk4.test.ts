import { PipelineManager } from '../../src/solver/pipeline';
import { AppLoop } from '../../src/app/loop';

describe('RK4 Pipeline', () => {
    it('compiles shaders without errors', async () => {
        const mockDevice = {
            createShaderModule: jest.fn().mockReturnValue({}),
            createComputePipeline: jest.fn().mockReturnValue({}),
        } as unknown as GPUDevice;

        const manager = new PipelineManager(mockDevice);

        await manager.createComputePipeline('rk4Stage.wgsl');
        await manager.createComputePipeline('rk4Combine.wgsl');

        expect(mockDevice.createShaderModule).toHaveBeenCalledTimes(2);
        expect(mockDevice.createComputePipeline).toHaveBeenCalledTimes(2);
    });
});

describe('AppLoop', () => {
    beforeEach(() => {
        jest.useFakeTimers();
        (global as any).requestAnimationFrame = (cb: FrameRequestCallback) => setTimeout(() => cb(performance.now()), 16) as any;
        (global as any).cancelAnimationFrame = (id: any) => clearTimeout(id);
    });

    afterEach(() => {
        jest.useRealTimers();
        delete (global as any).requestAnimationFrame;
        delete (global as any).cancelAnimationFrame;
    });

    it('executes correct number of steps per frame', () => {
        const stepFn = jest.fn();
        const renderFn = jest.fn();
        const getStepsPerFrame = () => 3;

        const loop = new AppLoop(stepFn, renderFn, getStepsPerFrame);
        loop.start();

        jest.advanceTimersByTime(16); // simulate 1 frame

        // Should have called stepFn 3 times
        expect(stepFn).toHaveBeenCalledTimes(3);
        // Should have called renderFn 1 time
        expect(renderFn).toHaveBeenCalledTimes(1);

        loop.stop();
    });

    it('does not execute steps if paused', () => {
        const stepFn = jest.fn();
        const renderFn = jest.fn();
        const getStepsPerFrame = () => 3;

        const loop = new AppLoop(stepFn, renderFn, getStepsPerFrame);
        loop.isPaused = true;
        loop.start();

        jest.advanceTimersByTime(16);

        // Should NOT have called stepFn
        expect(stepFn).not.toHaveBeenCalled();
        // Should STILL have called renderFn to display paused state
        expect(renderFn).toHaveBeenCalledTimes(1);

        loop.stop();
    });
});
