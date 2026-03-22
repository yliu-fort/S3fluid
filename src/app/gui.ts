import GUI from 'lil-gui';
import { SOLVER_CONFIG } from '../../src/solver/config';

export interface AppState {
    lmax: number;
    dt: number;
    nu: number;
    filterAlpha: number;
    filterOrder: number;
    seed: number;
    amplitude: number;
    stepsPerFrame: number;
    displayScale: number;
    showEnergy: boolean;
    pause: boolean;
    reset: () => void;
}

export class AppGUI {
    public gui: GUI;
    public state: AppState;

    constructor(
        onReset: () => void,
        onRebuild: () => void
    ) {
        this.gui = new GUI();

        this.state = {
            lmax: SOLVER_CONFIG.lmax,
            dt: SOLVER_CONFIG.dt,
            nu: SOLVER_CONFIG.nu,
            filterAlpha: SOLVER_CONFIG.filterAlpha,
            filterOrder: SOLVER_CONFIG.filterOrder,
            seed: SOLVER_CONFIG.seed,
            amplitude: SOLVER_CONFIG.amplitude,
            stepsPerFrame: SOLVER_CONFIG.stepsPerFrame,
            displayScale: 0.1,
            showEnergy: true,
            pause: false,
            reset: onReset
        };

        const simFolder = this.gui.addFolder('Simulation');
        // If lmax is changed, we need a complete rebuild of buffers and pipelines
        simFolder.add(this.state, 'lmax', [15, 31, 63, 127, 255]).onChange(() => {
            SOLVER_CONFIG.lmax = this.state.lmax;
            onRebuild();
        });
        simFolder.add(this.state, 'dt', 0.001, 0.1).name('dt').onChange((v: number) => SOLVER_CONFIG.dt = v);
        simFolder.add(this.state, 'nu', 0, 1e-4).name('nu (diffusion)').onChange((v: number) => SOLVER_CONFIG.nu = v);
        simFolder.add(this.state, 'filterAlpha', 0, 100).onChange((v: number) => SOLVER_CONFIG.filterAlpha = v);
        simFolder.add(this.state, 'filterOrder', 2, 16, 2).onChange((v: number) => SOLVER_CONFIG.filterOrder = v);
        simFolder.add(this.state, 'seed').onChange((v: number) => SOLVER_CONFIG.seed = v);
        simFolder.add(this.state, 'amplitude').onChange((v: number) => SOLVER_CONFIG.amplitude = v);
        simFolder.add(this.state, 'stepsPerFrame', 1, 50, 1).onChange((v: number) => SOLVER_CONFIG.stepsPerFrame = v);

        const viewFolder = this.gui.addFolder('Visualization');
        viewFolder.add(this.state, 'displayScale', 0.01, 2.0);
        viewFolder.add(this.state, 'showEnergy');

        const ctrlFolder = this.gui.addFolder('Controls');
        ctrlFolder.add(this.state, 'pause');
        ctrlFolder.add(this.state, 'reset');
    }
}
