import { createConfig } from '../solver/config';
import { SimulationBuffers } from '../solver/buffers';
import { initPrecomputeBuffers } from '../solver/precompute';
import { SimulationPipeline } from '../solver/pipeline';
import { SimulationDiagnostics } from '../solver/diagnostics';
import { SimulationLoop } from './loop';
import { SphereView } from '../render/sphereView';
import { AppGUI } from './gui';

import fftForwardLonSrc from '../shaders/fftForwardLon.wgsl?raw';
import fftInverseLonSrc from '../shaders/fftInverseLon.wgsl?raw';
import legendreAnalysisSrc from '../shaders/legendreAnalysis.wgsl?raw';
import legendreSynthesisSrc from '../shaders/legendreSynthesis.wgsl?raw';
import legendreSynthesisDThetaSrc from '../shaders/legendreSynthesisDTheta.wgsl?raw';
import mulIMSrc from '../shaders/mulIM.wgsl?raw';
import applyLaplacianSrc from '../shaders/applyLaplacian.wgsl?raw';
import invertLaplacianSrc from '../shaders/invertLaplacian.wgsl?raw';
import filterSpectrumSrc from '../shaders/filterSpectrum.wgsl?raw';
import initRandomSrc from '../shaders/initRandom.wgsl?raw';
import velocityFromPsiSrc from '../shaders/velocityFromPsi.wgsl?raw';
import advectGridSrc from '../shaders/advectGrid.wgsl?raw';
import rhsComposeSrc from '../shaders/rhsCompose.wgsl?raw';
import rk4StageSrc from '../shaders/rk4Stage.wgsl?raw';
import rk4CombineSrc from '../shaders/rk4Combine.wgsl?raw';
import energyIntegrandSrc from '../shaders/energyIntegrand.wgsl?raw';
import reduceSumSrc from '../shaders/reduceSum.wgsl?raw';

class App {
    device!: GPUDevice;
    config: any;
    buffers!: SimulationBuffers;
    pipeline!: SimulationPipeline;
    diagnostics!: SimulationDiagnostics;
    loop!: SimulationLoop;
    view!: SphereView;
    gui!: AppGUI;

    canvas!: HTMLCanvasElement;

    // For updating
    frameId: number = 0;

    constructor() {
        this.canvas = document.getElementById('webgpu-canvas') as HTMLCanvasElement;
        if (!this.canvas) {
            this.canvas = document.createElement('canvas');
            this.canvas.id = 'webgpu-canvas';
            document.body.appendChild(this.canvas);
        }
    }

    async init(initialConfigParams: any = {}) {
        if (!navigator.gpu) {
            console.error("WebGPU is not supported");
            return;
        }
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            console.error("No WebGPU adapter found");
            return;
        }

        this.device = await adapter.requestDevice();

        const defaultConfig = { lmax: 63 };
        this.config = createConfig({ ...defaultConfig, ...initialConfigParams });

        this.buffers = new SimulationBuffers(this.device, this.config);

        await initPrecomputeBuffers(this.device, this.config, this.buffers);

        this.pipeline = new SimulationPipeline(this.device, this.config, this.buffers);
        await this.pipeline.init(
            fftForwardLonSrc,
            fftInverseLonSrc,
            legendreAnalysisSrc,
            legendreSynthesisSrc,
            legendreSynthesisDThetaSrc,
            mulIMSrc,
            applyLaplacianSrc,
            invertLaplacianSrc,
            filterSpectrumSrc,
            initRandomSrc,
            velocityFromPsiSrc,
            advectGridSrc,
            rhsComposeSrc,
            rk4StageSrc,
            rk4CombineSrc,
            energyIntegrandSrc,
            reduceSumSrc
        );

        this.diagnostics = new SimulationDiagnostics(this.device, this.config, this.buffers, this.pipeline);
        this.loop = new SimulationLoop(this.device, this.config, this.buffers, this.pipeline, this.diagnostics);

        this.loop.reset();

        if (this.gui) {
            this.gui.gui.destroy();
        }

        this.gui = new AppGUI(this.loop, this.config, (newConfig) => {
            this.restart(newConfig);
        });

        // Initialize WebGL/Three.js view
        if (this.view) {
             // cleanup previous view
        }
        this.view = new SphereView(this.canvas, this.config);

        this.startLoop();
    }

    async restart(newConfigParams: any) {
        cancelAnimationFrame(this.frameId);
        await this.init(newConfigParams);
    }

    startLoop() {
        const renderFrame = async () => {
            if (!this.loop.isPaused) {
                this.loop.step();
                this.loop.updateDiagnostics();

                await this.view.updateData(this.device, this.buffers.zetaGrid, this.config);

                if (this.gui.params.showEnergy) {
                    const energy = await this.diagnostics.readEnergyAsync();
                    // Optional: Plot the energy history
                    // console.log("Energy: ", energy);
                }
            }

            this.view.render();
            this.frameId = requestAnimationFrame(renderFrame);
        };
        this.frameId = requestAnimationFrame(renderFrame);
    }
}

// Start application
const app = new App();
app.init().catch(console.error);

export default app;
