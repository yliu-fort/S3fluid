import { SOLVER_CONFIG } from './config';
import { SPECTRAL_COEFFS_F32_COUNT, GRID_POINTS_F32_COUNT } from './layout';
import type { PrecomputedData } from './precompute';

export class SolverBuffers {
    zetaLM_A!: GPUBuffer;
    zetaLM_B!: GPUBuffer;
    psiLM!: GPUBuffer;

    k1!: GPUBuffer;
    k2!: GPUBuffer;
    k3!: GPUBuffer;
    k4!: GPUBuffer;
    tmpLM!: GPUBuffer;
    tmpLM2!: GPUBuffer;
    freqBuffer!: GPUBuffer;

    zetaGrid!: GPUBuffer;
    psiGrid!: GPUBuffer;
    dpsiDphiGrid!: GPUBuffer;
    dpsiDthetaGrid!: GPUBuffer;
    uThetaGrid!: GPUBuffer;
    uPhiGrid!: GPUBuffer;
    dzetaDphiGrid!: GPUBuffer;
    dzetaDthetaGrid!: GPUBuffer;
    advGrid!: GPUBuffer;

    energyTerms!: GPUBuffer;
    energyOutput!: GPUBuffer;

    // Constants
    muBuffer!: GPUBuffer;
    wBuffer!: GPUBuffer;
    thetaBuffer!: GPUBuffer;
    sinThetaBuffer!: GPUBuffer;
    phiBuffer!: GPUBuffer;
    P_lmBuffer!: GPUBuffer;
    dP_lm_dthetaBuffer!: GPUBuffer;
    lapEigsBuffer!: GPUBuffer;
    specFilterBuffer!: GPUBuffer;
    initSlopeBuffer!: GPUBuffer;

    // We can also have ping pong boolean here
    pingPong: boolean = false;

    constructor(private device: GPUDevice) {
    }

    createDataBuffer(sizeInF32: number, usage: GPUBufferUsageFlags, label: string): GPUBuffer {
        return this.device.createBuffer({
            size: sizeInF32 * 4,
            usage,
            label,
        });
    }

    createStorageBuffer(sizeInF32: number, label: string): GPUBuffer {
        return this.createDataBuffer(sizeInF32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, label);
    }

    init(precomputedData: PrecomputedData) {
        const spectralCount = SPECTRAL_COEFFS_F32_COUNT;
        const gridCount = GRID_POINTS_F32_COUNT;

        this.zetaLM_A = this.createStorageBuffer(spectralCount, "zetaLM_A");
        this.zetaLM_B = this.createStorageBuffer(spectralCount, "zetaLM_B");
        this.psiLM = this.createStorageBuffer(spectralCount, "psiLM");

        this.k1 = this.createStorageBuffer(spectralCount, "k1");
        this.k2 = this.createStorageBuffer(spectralCount, "k2");
        this.k3 = this.createStorageBuffer(spectralCount, "k3");
        this.k4 = this.createStorageBuffer(spectralCount, "k4");
        this.tmpLM = this.createStorageBuffer(spectralCount, "tmpLM");
        this.tmpLM2 = this.createStorageBuffer(spectralCount, "tmpLM2");
        this.freqBuffer = this.createStorageBuffer(SOLVER_CONFIG.nlat * (SOLVER_CONFIG.lmax + 1) * 2, "freqBuffer");

        this.zetaGrid = this.createStorageBuffer(gridCount, "zetaGrid");
        this.psiGrid = this.createStorageBuffer(gridCount, "psiGrid");
        this.dpsiDphiGrid = this.createStorageBuffer(gridCount, "dpsiDphiGrid");
        this.dpsiDthetaGrid = this.createStorageBuffer(gridCount, "dpsiDthetaGrid");
        this.uThetaGrid = this.createStorageBuffer(gridCount, "uThetaGrid");
        this.uPhiGrid = this.createStorageBuffer(gridCount, "uPhiGrid");
        this.dzetaDphiGrid = this.createStorageBuffer(gridCount, "dzetaDphiGrid");
        this.dzetaDthetaGrid = this.createStorageBuffer(gridCount, "dzetaDthetaGrid");
        this.advGrid = this.createStorageBuffer(gridCount, "advGrid");

        this.energyTerms = this.createStorageBuffer(gridCount, "energyTerms");
        this.energyOutput = this.createStorageBuffer(1, "energyOutput"); // output float

        const L = SOLVER_CONFIG.lmax + 1;
        const M = SOLVER_CONFIG.lmax + 1;
        const J = SOLVER_CONFIG.nlat;

        this.muBuffer = this.createDataBuffer(J, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "mu");
        this.device.queue.writeBuffer(this.muBuffer, 0, precomputedData.mu.buffer as ArrayBuffer);

        this.wBuffer = this.createDataBuffer(J, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "w");
        this.device.queue.writeBuffer(this.wBuffer, 0, precomputedData.w.buffer as ArrayBuffer);

        this.thetaBuffer = this.createDataBuffer(J, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "theta");
        this.device.queue.writeBuffer(this.thetaBuffer, 0, precomputedData.theta.buffer as ArrayBuffer);

        this.sinThetaBuffer = this.createDataBuffer(J, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "sinTheta");
        this.device.queue.writeBuffer(this.sinThetaBuffer, 0, precomputedData.sinTheta.buffer as ArrayBuffer);

        this.phiBuffer = this.createDataBuffer(SOLVER_CONFIG.nlon, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "phi");
        this.device.queue.writeBuffer(this.phiBuffer, 0, precomputedData.phi.buffer as ArrayBuffer);

        this.P_lmBuffer = this.createDataBuffer(J * M * L, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "P_lm");
        this.device.queue.writeBuffer(this.P_lmBuffer, 0, precomputedData.P_lm.buffer as ArrayBuffer);

        this.dP_lm_dthetaBuffer = this.createDataBuffer(J * M * L, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "dP_lm_dtheta");
        this.device.queue.writeBuffer(this.dP_lm_dthetaBuffer, 0, precomputedData.dP_lm_dtheta.buffer as ArrayBuffer);

        this.lapEigsBuffer = this.createDataBuffer(M * L, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "lapEigs");
        this.device.queue.writeBuffer(this.lapEigsBuffer, 0, precomputedData.lapEigs.buffer as ArrayBuffer);

        this.specFilterBuffer = this.createDataBuffer(M * L, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "specFilter");
        this.device.queue.writeBuffer(this.specFilterBuffer, 0, precomputedData.specFilter.buffer as ArrayBuffer);

        this.initSlopeBuffer = this.createDataBuffer(M * L, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "initSlope");
        this.device.queue.writeBuffer(this.initSlopeBuffer, 0, precomputedData.initSlope.buffer as ArrayBuffer);
    }
}
