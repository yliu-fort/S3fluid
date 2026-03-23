import { SimulationConfig } from "../solver/config";
import { SimulationBuffers } from "../solver/buffers";
import { SimulationPipeline } from "../solver/pipeline";
import { SimulationDiagnostics } from "../solver/diagnostics";

export class SimulationLoop {
    device: GPUDevice;
    config: SimulationConfig;
    buffers: SimulationBuffers;
    pipeline: SimulationPipeline;
    diagnostics: SimulationDiagnostics;

    isPaused: boolean = false;
    currentZetaBuffer: GPUBuffer;
    nextZetaBuffer: GPUBuffer;

    constructor(
        device: GPUDevice,
        config: SimulationConfig,
        buffers: SimulationBuffers,
        pipeline: SimulationPipeline,
        diagnostics: SimulationDiagnostics
    ) {
        this.device = device;
        this.config = config;
        this.buffers = buffers;
        this.pipeline = pipeline;
        this.diagnostics = diagnostics;

        this.currentZetaBuffer = buffers.zetaLM_A;
        this.nextZetaBuffer = buffers.zetaLM_B;
    }

    private rhsPass(passEncoder: GPUComputePassEncoder, inLM: GPUBuffer, kOut: GPUBuffer) {
        // psi = invLap(zeta)
        this.pipeline.passInvertLaplacian(passEncoder, inLM, this.buffers.lapEigs, this.buffers.psiLM);

        // dpsiDphi = synth(i * m * psiLM)
        this.pipeline.passMulIM(passEncoder, this.buffers.psiLM, this.buffers.tmpLM);
        this.pipeline.passLegendreSynthesis(passEncoder, this.buffers.tmpLM, this.buffers.P_lm, this.buffers.tmpLM);
        this.pipeline.passFFTInverse(passEncoder, this.buffers.tmpLM, this.buffers.dpsiDphiGrid);

        // dpsiDtheta = synth_dtheta(psiLM)
        this.pipeline.passLegendreSynthesisDTheta(passEncoder, this.buffers.psiLM, this.buffers.dP_lm_dtheta, this.buffers.tmpLM);
        this.pipeline.passFFTInverse(passEncoder, this.buffers.tmpLM, this.buffers.dpsiDthetaGrid);

        // uTheta, uPhi = velocityFromPsi
        this.pipeline.passVelocityFromPsi(passEncoder, this.buffers.dpsiDphiGrid, this.buffers.dpsiDthetaGrid, this.buffers.sinTheta, this.buffers.uThetaGrid, this.buffers.uPhiGrid);

        // dzetaDtheta = synth_dtheta(zetaLM)
        this.pipeline.passLegendreSynthesisDTheta(passEncoder, inLM, this.buffers.dP_lm_dtheta, this.buffers.tmpLM);
        this.pipeline.passFFTInverse(passEncoder, this.buffers.tmpLM, this.buffers.dzetaDthetaGrid);

        // dzetaDphi = synth(i * m * zetaLM) / sinTheta (wait, adv needs dzetaDphi, we can just compute it)
        this.pipeline.passMulIM(passEncoder, inLM, this.buffers.tmpLM);
        this.pipeline.passLegendreSynthesis(passEncoder, this.buffers.tmpLM, this.buffers.P_lm, this.buffers.tmpLM);
        this.pipeline.passFFTInverse(passEncoder, this.buffers.tmpLM, this.buffers.dzetaDphiGrid);

        // advGrid
        this.pipeline.passAdvectGrid(passEncoder, this.buffers.uThetaGrid, this.buffers.uPhiGrid, this.buffers.dzetaDthetaGrid, this.buffers.dzetaDphiGrid, this.buffers.advGrid);

        // advLM = filter(analysis(advGrid))
        this.pipeline.passFFTForward(passEncoder, this.buffers.advGrid, this.buffers.tmpLM);
        this.pipeline.passLegendreAnalysis(passEncoder, this.buffers.tmpLM, this.buffers.w, this.buffers.P_lm, this.buffers.tmpLM);
        this.pipeline.passFilterSpectrum(passEncoder, this.buffers.tmpLM, this.buffers.specFilter, this.buffers.tmpLM);

        // rhsCompose
        this.pipeline.passRhsCompose(passEncoder, this.buffers.tmpLM, inLM, this.buffers.lapEigs, kOut);
    }

    step() {
        if (this.isPaused) return;

        for (let s = 0; s < this.config.stepsPerFrame; s++) {
            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();

            // k1 = RHS(z)
            this.rhsPass(passEncoder, this.currentZetaBuffer, this.buffers.k1);

            // tmpLM = z + 0.5 * dt * k1
            this.pipeline.passRk4Stage(passEncoder, this.currentZetaBuffer, this.buffers.k1, this.buffers.tmpLM, 0.5);

            // k2 = RHS(tmpLM)
            this.rhsPass(passEncoder, this.buffers.tmpLM, this.buffers.k2);

            // tmpLM = z + 0.5 * dt * k2
            this.pipeline.passRk4Stage(passEncoder, this.currentZetaBuffer, this.buffers.k2, this.buffers.tmpLM, 0.5);

            // k3 = RHS(tmpLM)
            this.rhsPass(passEncoder, this.buffers.tmpLM, this.buffers.k3);

            // tmpLM = z + dt * k3
            this.pipeline.passRk4Stage(passEncoder, this.currentZetaBuffer, this.buffers.k3, this.buffers.tmpLM, 1.0);

            // k4 = RHS(tmpLM)
            this.rhsPass(passEncoder, this.buffers.tmpLM, this.buffers.k4);

            // zNext = filter(z + (dt/6) * (k1 + 2k2 + 2k3 + k4))
            this.pipeline.passRk4Combine(
                passEncoder,
                this.currentZetaBuffer,
                this.buffers.k1,
                this.buffers.k2,
                this.buffers.k3,
                this.buffers.k4,
                this.buffers.specFilter,
                this.nextZetaBuffer
            );

            passEncoder.end();
            this.device.queue.submit([commandEncoder.finish()]);

            // Swap buffers
            const temp = this.currentZetaBuffer;
            this.currentZetaBuffer = this.nextZetaBuffer;
            this.nextZetaBuffer = temp;
        }
    }

    updateDiagnostics() {
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();

        // synthesize field for visualization and diagnostics
        this.diagnostics.synthesizeZetaAndPsi(passEncoder, this.currentZetaBuffer);

        // compute energy
        this.diagnostics.computeEnergy(passEncoder);

        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }

    reset() {
        // Generate random field
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();

        this.pipeline.passInitRandom(
            passEncoder,
            this.buffers.initSlope,
            this.buffers.specFilter,
            this.currentZetaBuffer
        );

        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }
}
