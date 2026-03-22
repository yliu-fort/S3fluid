import { AppGUI } from './gui';
import { AppLoop } from './loop';
import { SphereView } from '../render/sphereView';
import { SOLVER_CONFIG } from '../solver/config';
import { SolverBuffers } from '../solver/buffers';
import { generatePrecomputedData } from '../solver/precompute';
import { PipelineManager } from '../solver/pipeline';
import { Diagnostics } from '../solver/diagnostics';
import * as THREE from 'three';

// Main execution entrypoint for demo
class MainApp {
    public gui!: AppGUI;
    public loop!: AppLoop;
    public view!: SphereView;
    public device!: GPUDevice;
    public buffers!: SolverBuffers;
    public pipelines!: PipelineManager;
    public diagnostics!: Diagnostics;

    public scalarTexture!: THREE.DataTexture;
    public scalarData!: Float32Array;
    private readbackBuffer!: GPUBuffer;

    private forwardFFT!: GPUComputePipeline;
    private inverseFFT!: GPUComputePipeline;
    private legendreAnalysis!: GPUComputePipeline;
    private legendreSynthesis!: GPUComputePipeline;
    private legendreSynthesisDTheta!: GPUComputePipeline;
    private mulIM!: GPUComputePipeline;
    private applyLaplacian!: GPUComputePipeline;
    private invertLaplacian!: GPUComputePipeline;
    private filterSpectrum!: GPUComputePipeline;
    private velocityFromPsi!: GPUComputePipeline;
    private advectGrid!: GPUComputePipeline;
    private rhsCompose!: GPUComputePipeline;
    private rk4Stage!: GPUComputePipeline;
    private rk4Combine!: GPUComputePipeline;
    private energyIntegrand!: GPUComputePipeline;
    private reduceSum!: GPUComputePipeline;
    private initSpectrum!: GPUComputePipeline;

    private configBuffer!: GPUBuffer;
    private rhsConfigBuffer!: GPUBuffer;
    private rk4ConfigBuffer1!: GPUBuffer;
    private rk4ConfigBuffer2!: GPUBuffer;
    private reduceConfigBuffer!: GPUBuffer;

    constructor() {
        this.init().catch(console.error);
    }

    async init() {
        if (!navigator.gpu) {
            console.error("WebGPU not supported on this browser.");
            return;
        }
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            console.error("No appropriate WebGPU adapter found.");
            return;
        }
        this.device = await adapter.requestDevice();

        const canvas = document.getElementById('renderCanvas') as HTMLCanvasElement;
        this.view = new SphereView(canvas);

        this.gui = new AppGUI(
            () => this.resetSimulation(),
            () => this.rebuildSimulation()
        );

        this.diagnostics = new Diagnostics();

        await this.setupWebGPU();

        this.loop = new AppLoop(
            () => this.step(),
            () => this.render(),
            () => this.gui.state.stepsPerFrame
        );

        this.loop.start();
    }

    async setupWebGPU() {
        const precomputed = generatePrecomputedData(SOLVER_CONFIG.lmax, SOLVER_CONFIG.nlat, SOLVER_CONFIG.nlon);

        this.buffers = new SolverBuffers(this.device);
        this.buffers.init(precomputed);

        const configData = new Uint32Array([
            SOLVER_CONFIG.nlat,
            SOLVER_CONFIG.nlon,
            SOLVER_CONFIG.lmax,
            0
        ]);
        this.configBuffer = this.device.createBuffer({
            size: configData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.configBuffer, 0, configData);

        this.readbackBuffer = this.device.createBuffer({
            size: SOLVER_CONFIG.nlat * SOLVER_CONFIG.nlon * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        // Diagnostics readback
        this.buffers.energyOutput = this.device.createBuffer({
            size: 4, // 1 float
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        // Preallocate uniform buffers for RK4 and RHS loops
        this.rhsConfigBuffer = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.rk4ConfigBuffer1 = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.rk4ConfigBuffer2 = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.reduceConfigBuffer = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

        this.pipelines = new PipelineManager(this.device);

        [
            this.forwardFFT,
            this.inverseFFT,
            this.legendreAnalysis,
            this.legendreSynthesis,
            this.legendreSynthesisDTheta,
            this.mulIM,
            this.applyLaplacian,
            this.invertLaplacian,
            this.filterSpectrum,
            this.velocityFromPsi,
            this.advectGrid,
            this.rhsCompose,
            this.rk4Stage,
            this.rk4Combine,
            this.energyIntegrand,
            this.reduceSum,
            this.initSpectrum
        ] = await Promise.all([
            this.pipelines.createComputePipeline('fftForwardLon.wgsl'),
            this.pipelines.createComputePipeline('fftInverseLon.wgsl'),
            this.pipelines.createComputePipeline('legendreAnalysis.wgsl'),
            this.pipelines.createComputePipeline('legendreSynthesis.wgsl'),
            this.pipelines.createComputePipeline('legendreSynthesisDTheta.wgsl'),
            this.pipelines.createComputePipeline('mulIM.wgsl'),
            this.pipelines.createComputePipeline('applyLaplacian.wgsl'),
            this.pipelines.createComputePipeline('invertLaplacian.wgsl'),
            this.pipelines.createComputePipeline('filterSpectrum.wgsl'),
            this.pipelines.createComputePipeline('velocityFromPsi.wgsl'),
            this.pipelines.createComputePipeline('advectGrid.wgsl'),
            this.pipelines.createComputePipeline('rhsCompose.wgsl'),
            this.pipelines.createComputePipeline('rk4Stage.wgsl'),
            this.pipelines.createComputePipeline('rk4Combine.wgsl'),
            this.pipelines.createComputePipeline('energyIntegrand.wgsl'),
            this.pipelines.createComputePipeline('reduceSum.wgsl'),
            this.pipelines.createComputePipeline('initSpectrum.wgsl')
        ]);

        const initRandom = await this.pipelines.createComputePipeline('initRandom.wgsl');
        const initConfigData = new Float32Array([SOLVER_CONFIG.seed, SOLVER_CONFIG.amplitude]);
        const initConfigBuffer = this.device.createBuffer({
            size: initConfigData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(initConfigBuffer, 0, initConfigData);

        const initBindGroup0 = this.device.createBindGroup({
            layout: initRandom.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: this.buffers.zetaGrid } }]
        });
        const initBindGroup1 = this.device.createBindGroup({
            layout: initRandom.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.configBuffer } },
                { binding: 1, resource: { buffer: initConfigBuffer } }
            ]
        });

        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(initRandom);
        passEncoder.setBindGroup(0, initBindGroup0);
        passEncoder.setBindGroup(1, initBindGroup1);
        passEncoder.dispatchWorkgroups(Math.ceil(SOLVER_CONFIG.nlon / 16), Math.ceil(SOLVER_CONFIG.nlat / 16));
        passEncoder.end();

        this.dispatchAnalysis(commandEncoder, this.buffers.zetaGrid, this.buffers.zetaLM_A);

        const initSpecBg0 = this.device.createBindGroup({
            layout: this.initSpectrum.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.zetaLM_A } },
                { binding: 1, resource: { buffer: this.buffers.initSlopeBuffer } },
                { binding: 2, resource: { buffer: this.buffers.specFilterBuffer } }
            ]
        });
        const configBg1 = this.device.createBindGroup({
            layout: this.initSpectrum.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this.configBuffer } }]
        });
        const specPass = commandEncoder.beginComputePass();
        specPass.setPipeline(this.initSpectrum);
        specPass.setBindGroup(0, initSpecBg0);
        specPass.setBindGroup(1, configBg1);
        specPass.dispatchWorkgroups(Math.ceil((SOLVER_CONFIG.lmax + 1) / 16), Math.ceil((SOLVER_CONFIG.lmax + 1) / 16));
        specPass.end();

        this.device.queue.submit([commandEncoder.finish()]);

        this.scalarData = new Float32Array(SOLVER_CONFIG.nlat * SOLVER_CONFIG.nlon);
        this.scalarTexture = new THREE.DataTexture(this.scalarData, SOLVER_CONFIG.nlon, SOLVER_CONFIG.nlat, THREE.RedFormat, THREE.FloatType);
        this.scalarTexture.needsUpdate = true;
        this.view.updateTexture(this.scalarTexture);
    }

    private dispatchAnalysis(encoder: GPUCommandEncoder, gridIn: GPUBuffer, coeffOut: GPUBuffer) {
        const fftPass = encoder.beginComputePass();
        fftPass.setPipeline(this.forwardFFT);
        fftPass.setBindGroup(0, this.device.createBindGroup({
            layout: this.forwardFFT.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: gridIn } },
                { binding: 1, resource: { buffer: this.buffers.freqBuffer } },
                { binding: 2, resource: { buffer: this.buffers.phiBuffer } }
            ]
        }));
        fftPass.setBindGroup(1, this.device.createBindGroup({
            layout: this.forwardFFT.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this.configBuffer } }]
        }));
        fftPass.dispatchWorkgroups(Math.ceil((SOLVER_CONFIG.lmax + 1) / 16), Math.ceil(SOLVER_CONFIG.nlat / 16));
        fftPass.end();

        const legPass = encoder.beginComputePass();
        legPass.setPipeline(this.legendreAnalysis);
        legPass.setBindGroup(0, this.device.createBindGroup({
            layout: this.legendreAnalysis.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.freqBuffer } },
                { binding: 1, resource: { buffer: coeffOut } },
                { binding: 2, resource: { buffer: this.buffers.wBuffer } },
                { binding: 3, resource: { buffer: this.buffers.P_lmBuffer } }
            ]
        }));
        legPass.setBindGroup(1, this.device.createBindGroup({
            layout: this.legendreAnalysis.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this.configBuffer } }]
        }));
        legPass.dispatchWorkgroups(Math.ceil((SOLVER_CONFIG.lmax + 1) / 16), Math.ceil((SOLVER_CONFIG.lmax + 1) / 16));
        legPass.end();
    }

    private dispatchSynthesis(encoder: GPUCommandEncoder, coeffIn: GPUBuffer, gridOut: GPUBuffer) {
        const legPass = encoder.beginComputePass();
        legPass.setPipeline(this.legendreSynthesis);
        legPass.setBindGroup(0, this.device.createBindGroup({
            layout: this.legendreSynthesis.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: coeffIn } },
                { binding: 1, resource: { buffer: this.buffers.freqBuffer } },
                { binding: 2, resource: { buffer: this.buffers.P_lmBuffer } }
            ]
        }));
        legPass.setBindGroup(1, this.device.createBindGroup({
            layout: this.legendreSynthesis.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this.configBuffer } }]
        }));
        legPass.dispatchWorkgroups(Math.ceil(SOLVER_CONFIG.nlat / 16), Math.ceil((SOLVER_CONFIG.lmax + 1) / 16));
        legPass.end();

        const fftPass = encoder.beginComputePass();
        fftPass.setPipeline(this.inverseFFT);
        fftPass.setBindGroup(0, this.device.createBindGroup({
            layout: this.inverseFFT.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.freqBuffer } },
                { binding: 1, resource: { buffer: gridOut } },
                { binding: 2, resource: { buffer: this.buffers.phiBuffer } }
            ]
        }));
        fftPass.setBindGroup(1, this.device.createBindGroup({
            layout: this.inverseFFT.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this.configBuffer } }]
        }));
        fftPass.dispatchWorkgroups(Math.ceil(SOLVER_CONFIG.nlon / 16), Math.ceil(SOLVER_CONFIG.nlat / 16));
        fftPass.end();
    }

    private dispatchRHS(encoder: GPUCommandEncoder, zetaIn: GPUBuffer, rhsOut: GPUBuffer) {
        const Lmax = SOLVER_CONFIG.lmax;
        const wkL = Math.ceil((Lmax + 1) / 16);

        // 1. psi_lm = invert_laplacian(zeta_lm)
        const pInv = encoder.beginComputePass();
        pInv.setPipeline(this.invertLaplacian);
        pInv.setBindGroup(0, this.device.createBindGroup({
            layout: this.invertLaplacian.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: zetaIn } },
                { binding: 1, resource: { buffer: this.buffers.lapEigsBuffer } },
                { binding: 2, resource: { buffer: this.buffers.psiLM } }
            ]
        }));
        pInv.setBindGroup(1, this.device.createBindGroup({
            layout: this.invertLaplacian.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this.configBuffer } }]
        }));
        pInv.dispatchWorkgroups(wkL, wkL);
        pInv.end();

        // 2. dpsi/dphi and dpsi/dtheta
        // dpsiDphi = synthesis(i*m * psiLM)
        const pMul = encoder.beginComputePass();
        pMul.setPipeline(this.mulIM);
        pMul.setBindGroup(0, this.device.createBindGroup({
            layout: this.mulIM.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.psiLM } },
                { binding: 1, resource: { buffer: this.buffers.tmpLM } }
            ]
        }));
        pMul.setBindGroup(1, this.device.createBindGroup({
            layout: this.mulIM.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this.configBuffer } }]
        }));
        pMul.dispatchWorkgroups(wkL, wkL);
        pMul.end();

        this.dispatchSynthesis(encoder, this.buffers.tmpLM, this.buffers.dpsiDphiGrid);

        // dpsiDtheta = synthesisDTheta(psiLM)
        const legPassD = encoder.beginComputePass();
        legPassD.setPipeline(this.legendreSynthesisDTheta);
        legPassD.setBindGroup(0, this.device.createBindGroup({
            layout: this.legendreSynthesisDTheta.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.psiLM } },
                { binding: 1, resource: { buffer: this.buffers.freqBuffer } },
                { binding: 2, resource: { buffer: this.buffers.dP_lm_dthetaBuffer } }
            ]
        }));
        legPassD.setBindGroup(1, this.device.createBindGroup({
            layout: this.legendreSynthesisDTheta.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this.configBuffer } }]
        }));
        legPassD.dispatchWorkgroups(Math.ceil(SOLVER_CONFIG.nlat / 16), wkL);
        legPassD.end();

        const fftPass = encoder.beginComputePass();
        fftPass.setPipeline(this.inverseFFT);
        fftPass.setBindGroup(0, this.device.createBindGroup({
            layout: this.inverseFFT.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.freqBuffer } },
                { binding: 1, resource: { buffer: this.buffers.dpsiDthetaGrid } },
                { binding: 2, resource: { buffer: this.buffers.phiBuffer } }
            ]
        }));
        fftPass.setBindGroup(1, this.device.createBindGroup({
            layout: this.inverseFFT.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this.configBuffer } }]
        }));
        fftPass.dispatchWorkgroups(Math.ceil(SOLVER_CONFIG.nlon / 16), Math.ceil(SOLVER_CONFIG.nlat / 16));
        fftPass.end();

        // 3. velocities
        const velPass = encoder.beginComputePass();
        velPass.setPipeline(this.velocityFromPsi);
        velPass.setBindGroup(0, this.device.createBindGroup({
            layout: this.velocityFromPsi.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.dpsiDphiGrid } },
                { binding: 1, resource: { buffer: this.buffers.dpsiDthetaGrid } },
                { binding: 2, resource: { buffer: this.buffers.sinThetaBuffer } },
                { binding: 3, resource: { buffer: this.buffers.uThetaGrid } },
                { binding: 4, resource: { buffer: this.buffers.uPhiGrid } }
            ]
        }));
        velPass.setBindGroup(1, this.device.createBindGroup({
            layout: this.velocityFromPsi.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this.configBuffer } }]
        }));
        velPass.dispatchWorkgroups(Math.ceil(SOLVER_CONFIG.nlon / 16), Math.ceil(SOLVER_CONFIG.nlat / 16));
        velPass.end();

        // 4. dzeta/dphi, dzeta/dtheta
        const pMul2 = encoder.beginComputePass();
        pMul2.setPipeline(this.mulIM);
        pMul2.setBindGroup(0, this.device.createBindGroup({
            layout: this.mulIM.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: zetaIn } },
                { binding: 1, resource: { buffer: this.buffers.tmpLM2 } }
            ]
        }));
        pMul2.setBindGroup(1, this.device.createBindGroup({
            layout: this.mulIM.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this.configBuffer } }]
        }));
        pMul2.dispatchWorkgroups(wkL, wkL);
        pMul2.end();

        this.dispatchSynthesis(encoder, this.buffers.tmpLM2, this.buffers.dzetaDphiGrid);

        const legPassD2 = encoder.beginComputePass();
        legPassD2.setPipeline(this.legendreSynthesisDTheta);
        legPassD2.setBindGroup(0, this.device.createBindGroup({
            layout: this.legendreSynthesisDTheta.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: zetaIn } },
                { binding: 1, resource: { buffer: this.buffers.freqBuffer } },
                { binding: 2, resource: { buffer: this.buffers.dP_lm_dthetaBuffer } }
            ]
        }));
        legPassD2.setBindGroup(1, this.device.createBindGroup({
            layout: this.legendreSynthesisDTheta.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this.configBuffer } }]
        }));
        legPassD2.dispatchWorkgroups(Math.ceil(SOLVER_CONFIG.nlat / 16), wkL);
        legPassD2.end();

        const fftPass2 = encoder.beginComputePass();
        fftPass2.setPipeline(this.inverseFFT);
        fftPass2.setBindGroup(0, this.device.createBindGroup({
            layout: this.inverseFFT.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.freqBuffer } },
                { binding: 1, resource: { buffer: this.buffers.dzetaDthetaGrid } },
                { binding: 2, resource: { buffer: this.buffers.phiBuffer } }
            ]
        }));
        fftPass2.setBindGroup(1, this.device.createBindGroup({
            layout: this.inverseFFT.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this.configBuffer } }]
        }));
        fftPass2.dispatchWorkgroups(Math.ceil(SOLVER_CONFIG.nlon / 16), Math.ceil(SOLVER_CONFIG.nlat / 16));
        fftPass2.end();

        // 5. Advect Grid
        const advPass = encoder.beginComputePass();
        advPass.setPipeline(this.advectGrid);
        advPass.setBindGroup(0, this.device.createBindGroup({
            layout: this.advectGrid.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.uThetaGrid } },
                { binding: 1, resource: { buffer: this.buffers.uPhiGrid } },
                { binding: 2, resource: { buffer: this.buffers.dzetaDthetaGrid } },
                { binding: 3, resource: { buffer: this.buffers.dzetaDphiGrid } },
                { binding: 4, resource: { buffer: this.buffers.sinThetaBuffer } },
                { binding: 5, resource: { buffer: this.buffers.advGrid } },
            ]
        }));
        advPass.setBindGroup(1, this.device.createBindGroup({
            layout: this.advectGrid.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this.configBuffer } }]
        }));
        advPass.dispatchWorkgroups(Math.ceil(SOLVER_CONFIG.nlon / 16), Math.ceil(SOLVER_CONFIG.nlat / 16));
        advPass.end();

        // 6. adv_lm = filter(analysis(advGrid))
        this.dispatchAnalysis(encoder, this.buffers.advGrid, this.buffers.tmpLM);

        const filterPass = encoder.beginComputePass();
        filterPass.setPipeline(this.filterSpectrum);
        filterPass.setBindGroup(0, this.device.createBindGroup({
            layout: this.filterSpectrum.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.tmpLM } },
                { binding: 1, resource: { buffer: this.buffers.specFilterBuffer } },
                { binding: 2, resource: { buffer: this.buffers.advGrid } } // use advGrid as tmp out to avoid rw conflict
            ]
        }));
        filterPass.setBindGroup(1, this.device.createBindGroup({
            layout: this.filterSpectrum.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this.configBuffer } }]
        }));
        filterPass.dispatchWorkgroups(wkL, wkL);
        filterPass.end();

        // copy back
        encoder.copyBufferToBuffer(this.buffers.advGrid, 0, this.buffers.tmpLM, 0, this.buffers.tmpLM.size);

        // 7. rhsCompose
        const rhsCData = new Float32Array([SOLVER_CONFIG.nu, 0, 0, 0]);
        const rhsCBuffer = this.device.createBuffer({
            size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(rhsCBuffer, 0, rhsCData);

        const compPass = encoder.beginComputePass();
        compPass.setPipeline(this.rhsCompose);
        compPass.setBindGroup(0, this.device.createBindGroup({
            layout: this.rhsCompose.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.tmpLM } },
                { binding: 1, resource: { buffer: zetaIn } },
                { binding: 2, resource: { buffer: this.buffers.lapEigsBuffer } },
                { binding: 3, resource: { buffer: rhsOut } }
            ]
        }));
        compPass.setBindGroup(1, this.device.createBindGroup({
            layout: this.rhsCompose.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.configBuffer } },
                { binding: 1, resource: { buffer: rhsCBuffer } }
            ]
        }));
        compPass.dispatchWorkgroups(wkL, wkL);
        compPass.end();
    }

    private dispatchRK4(encoder: GPUCommandEncoder) {
        const wkL = Math.ceil((SOLVER_CONFIG.lmax + 1) / 16);

        const rk4Data1 = new Float32Array([SOLVER_CONFIG.dt, 0.5, 0, 0]);
        this.device.queue.writeBuffer(this.rk4ConfigBuffer1, 0, rk4Data1);

        const configBg1 = this.device.createBindGroup({
            layout: this.rk4Stage.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.configBuffer } },
                { binding: 1, resource: { buffer: this.rk4ConfigBuffer1 } } // coeff = 0.5
            ]
        });

        // Current state is in zetaLM_A
        // k1
        this.dispatchRHS(encoder, this.buffers.zetaLM_A, this.buffers.k1);

        // z_temp = z + 0.5 * dt * k1
        let pStage = encoder.beginComputePass();
        pStage.setPipeline(this.rk4Stage);
        pStage.setBindGroup(0, this.device.createBindGroup({
            layout: this.rk4Stage.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.zetaLM_A } },
                { binding: 1, resource: { buffer: this.buffers.k1 } },
                { binding: 2, resource: { buffer: this.buffers.zTemp } } // Using dedicated zTemp
            ]
        }));
        pStage.setBindGroup(1, configBg1);
        pStage.dispatchWorkgroups(wkL, wkL);
        pStage.end();

        // k2
        this.dispatchRHS(encoder, this.buffers.zTemp, this.buffers.k2);

        // z_temp = z + 0.5 * dt * k2
        pStage = encoder.beginComputePass();
        pStage.setPipeline(this.rk4Stage);
        pStage.setBindGroup(0, this.device.createBindGroup({
            layout: this.rk4Stage.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.zetaLM_A } },
                { binding: 1, resource: { buffer: this.buffers.k2 } },
                { binding: 2, resource: { buffer: this.buffers.zTemp } }
            ]
        }));
        pStage.setBindGroup(1, configBg1);
        pStage.dispatchWorkgroups(wkL, wkL);
        pStage.end();

        // k3
        this.dispatchRHS(encoder, this.buffers.zTemp, this.buffers.k3);

        const rk4Data2 = new Float32Array([SOLVER_CONFIG.dt, 1.0, 0, 0]);
        this.device.queue.writeBuffer(this.rk4ConfigBuffer2, 0, rk4Data2);

        const configBg1Full = this.device.createBindGroup({
            layout: this.rk4Stage.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.configBuffer } },
                { binding: 1, resource: { buffer: this.rk4ConfigBuffer2 } } // coeff = 1.0
            ]
        });

        // z_temp = z + dt * k3
        pStage = encoder.beginComputePass();
        pStage.setPipeline(this.rk4Stage);
        pStage.setBindGroup(0, this.device.createBindGroup({
            layout: this.rk4Stage.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.zetaLM_A } },
                { binding: 1, resource: { buffer: this.buffers.k3 } },
                { binding: 2, resource: { buffer: this.buffers.zTemp } }
            ]
        }));
        pStage.setBindGroup(1, configBg1Full);
        pStage.dispatchWorkgroups(wkL, wkL);
        pStage.end();

        // k4
        this.dispatchRHS(encoder, this.buffers.zTemp, this.buffers.k4);

        // Combine
        const combPass = encoder.beginComputePass();
        combPass.setPipeline(this.rk4Combine);
        combPass.setBindGroup(0, this.device.createBindGroup({
            layout: this.rk4Combine.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.zetaLM_A } },
                { binding: 1, resource: { buffer: this.buffers.k1 } },
                { binding: 2, resource: { buffer: this.buffers.k2 } },
                { binding: 3, resource: { buffer: this.buffers.k3 } },
                { binding: 4, resource: { buffer: this.buffers.k4 } },
                { binding: 5, resource: { buffer: this.buffers.zetaLM_B } }, // Next state
            ]
        }));
        combPass.setBindGroup(1, this.device.createBindGroup({
            layout: this.rk4Combine.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.configBuffer } },
                { binding: 1, resource: { buffer: this.rk4ConfigBuffer2 } }
            ]
        }));
        combPass.dispatchWorkgroups(wkL, wkL);
        combPass.end();

        // Final filter: B -> A
        const filterPass = encoder.beginComputePass();
        filterPass.setPipeline(this.filterSpectrum);
        filterPass.setBindGroup(0, this.device.createBindGroup({
            layout: this.filterSpectrum.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.zetaLM_B } },
                { binding: 1, resource: { buffer: this.buffers.specFilterBuffer } },
                { binding: 2, resource: { buffer: this.buffers.zetaLM_A } } // Ping-pong back
            ]
        }));
        const cfgBgBase = this.device.createBindGroup({
            layout: this.filterSpectrum.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this.configBuffer } }]
        });
        filterPass.setBindGroup(1, cfgBgBase);
        filterPass.dispatchWorkgroups(wkL, wkL);
        filterPass.end();
    }

    private dispatchDiagnostics(encoder: GPUCommandEncoder) {
        // 1. Energy Integrand
        const intPass = encoder.beginComputePass();
        intPass.setPipeline(this.energyIntegrand);
        intPass.setBindGroup(0, this.device.createBindGroup({
            layout: this.energyIntegrand.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.psiGrid } },
                { binding: 1, resource: { buffer: this.buffers.zetaGrid } },
                { binding: 2, resource: { buffer: this.buffers.wBuffer } },
                { binding: 3, resource: { buffer: this.buffers.energyTerms } }
            ]
        }));
        intPass.setBindGroup(1, this.device.createBindGroup({
            layout: this.energyIntegrand.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this.configBuffer } }]
        }));
        intPass.dispatchWorkgroups(Math.ceil(SOLVER_CONFIG.nlon / 16), Math.ceil(SOLVER_CONFIG.nlat / 16));
        intPass.end();

        // 2. Reduce Sum (demo version, assumes nlat*nlon <= 256 for a single workgroup for brevity, or we can just run one huge workgroup)
        // Note: A robust reduction takes multiple passes. For this task, we assume the simple version we wrote.
        const totalElems = SOLVER_CONFIG.nlat * SOLVER_CONFIG.nlon;
        this.device.queue.writeBuffer(this.reduceConfigBuffer, 0, new Uint32Array([totalElems, 0, 0, 0]));

        const redPass = encoder.beginComputePass();
        redPass.setPipeline(this.reduceSum);
        redPass.setBindGroup(0, this.device.createBindGroup({
            layout: this.reduceSum.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.energyTerms } },
                { binding: 1, resource: { buffer: this.buffers.energyOutput } }
            ]
        }));
        redPass.setBindGroup(1, this.device.createBindGroup({
            layout: this.reduceSum.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this.reduceConfigBuffer } }]
        }));
        redPass.dispatchWorkgroups(1);
        redPass.end();
    }

    step() {
        if (!this.device || this.gui.state.pause) return;

        const encoder = this.device.createCommandEncoder();
        this.dispatchRK4(encoder);
        this.device.queue.submit([encoder.finish()]);
    }

    render() {
        if (!this.device || !this.readbackBuffer || this.readbackBuffer.mapState !== 'unmapped') {
            this.view.render();
            return;
        }

        this.view.setDisplayScale(this.gui.state.displayScale);

        // 1. Convert current zetaLM to Grid
        const encoder = this.device.createCommandEncoder();
        this.dispatchSynthesis(encoder, this.buffers.zetaLM_A, this.buffers.zetaGrid);

        // Run diagnostics
        if (this.gui.state.showEnergy) {
            this.dispatchDiagnostics(encoder);
        }

        // 2. Copy grid to readback buffer
        encoder.copyBufferToBuffer(
            this.buffers.zetaGrid, 0,
            this.readbackBuffer, 0,
            this.readbackBuffer.size
        );

        // Optional: Readback energy synchronously or use mapAsync just for energyOutput.
        // For simplicity we aren't displaying a graph overlay right now, but we compute it and store it.
        // Real implementation would copy `energyOutput` to a CPU mapped buffer and plot.

        this.device.queue.submit([encoder.finish()]);

        // Readback async
        this.readbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const arr = new Float32Array(this.readbackBuffer.getMappedRange());
            this.scalarData.set(arr);
            this.readbackBuffer.unmap();

            this.scalarTexture.needsUpdate = true;
            this.view.render();
        }).catch(err => {
            console.error("Readback failed", err);
        });
    }

    async resetSimulation() {
        this.diagnostics.reset();
        await this.setupWebGPU();
    }

    async rebuildSimulation() {
        await this.setupWebGPU();
    }
}

// Bootstrap
window.addEventListener('DOMContentLoaded', () => {
    const app = new MainApp();
    (window as any).app = app;
    (window as any).SOLVER_CONFIG = SOLVER_CONFIG;
});
