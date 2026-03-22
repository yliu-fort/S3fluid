import GUI from 'lil-gui';
import { SimulationLoop } from './loop';
import { SimulationConfig } from '../solver/config';

export class AppGUI {
    gui: GUI;
    loop: SimulationLoop;
    config: SimulationConfig;

    params = {
        lmax: 31,
        dt: 1e-2,
        nu: 1e-7,
        filterAlpha: 36.0,
        filterOrder: 8.0,
        seed: 42,
        amplitude: 1.0,
        stepsPerFrame: 4,
        displayScale: 10.0,
        pause: false,
        reset: () => {
            this.loop.reset();
        },
        showEnergy: true
    };

    onConfigChange: (newConfig: any) => void;

    constructor(loop: SimulationLoop, config: SimulationConfig, onChange: (c: any) => void) {
        this.gui = new GUI();
        this.loop = loop;
        this.config = config;
        this.onConfigChange = onChange;

        // initialize
        this.params.lmax = config.lmax;
        this.params.dt = config.dt;
        this.params.nu = config.nu;
        this.params.filterAlpha = config.filterAlpha;
        this.params.filterOrder = config.filterOrder;
        this.params.seed = config.seed;
        this.params.amplitude = config.amplitude;
        this.params.stepsPerFrame = config.stepsPerFrame;

        this.build();
    }

    build() {
        const simFolder = this.gui.addFolder('Simulation');

        simFolder.add(this.params, 'lmax', [31, 63]).onChange((val: number) => {
            this.onConfigChange({ ...this.params, lmax: val });
        });

        simFolder.add(this.params, 'dt', 1e-4, 5e-2, 1e-4).onChange((val: number) => {
            this.config.dt = val;
            this.updateConfigUniforms();
        });

        simFolder.add(this.params, 'nu', 0, 1e-4, 1e-7).onChange((val: number) => {
            this.config.nu = val;
            this.updateConfigUniforms();
        });

        simFolder.add(this.params, 'stepsPerFrame', 1, 20, 1).onChange((val: number) => {
            this.config.stepsPerFrame = val;
        });

        simFolder.add(this.params, 'pause').onChange((val: boolean) => {
            this.loop.isPaused = val;
        });

        simFolder.add(this.params, 'reset');

        const initFolder = this.gui.addFolder('Initialization');
        initFolder.add(this.params, 'seed', 0, 100, 1).onChange((val: number) => {
            this.onConfigChange({ ...this.params, seed: val });
        });
        initFolder.add(this.params, 'amplitude', 0.1, 10.0, 0.1).onChange((val: number) => {
            this.onConfigChange({ ...this.params, amplitude: val });
        });

        const filterFolder = this.gui.addFolder('Filter');
        filterFolder.add(this.params, 'filterAlpha', 0, 100, 1).onChange((val: number) => {
            this.onConfigChange({ ...this.params, filterAlpha: val });
        });
        filterFolder.add(this.params, 'filterOrder', 2, 16, 2).onChange((val: number) => {
            this.onConfigChange({ ...this.params, filterOrder: val });
        });

        const viewFolder = this.gui.addFolder('View');
        viewFolder.add(this.params, 'displayScale', 1.0, 50.0, 1.0);
        viewFolder.add(this.params, 'showEnergy');
    }

    private updateConfigUniforms() {
        // Here we'd need to re-upload the uniform buffer config.
        // Assuming there is a way to trigger that.
        // We'll leave it to the loop or pipeline to pick it up or we can provide a callback
    }
}
