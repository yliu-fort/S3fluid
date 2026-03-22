if (typeof globalThis.GPUBuffer === "undefined") { (globalThis as any).GPUBuffer = class GPUBuffer {}; }
import type { SimulationConfig } from "./config";
import { getSpectralSize, getGridSize } from "./layout";

export class SimulationBuffers {
    device: GPUDevice;
    config: SimulationConfig;

    zetaLM_A: GPUBuffer;
    zetaLM_B: GPUBuffer;
    psiLM: GPUBuffer;
    k1: GPUBuffer;
    k2: GPUBuffer;
    k3: GPUBuffer;
    k4: GPUBuffer;
    tmpLM: GPUBuffer;

    zetaGrid: GPUBuffer;
    psiGrid: GPUBuffer;
    dpsiDphiGrid: GPUBuffer;
    dpsiDthetaGrid: GPUBuffer;
    uThetaGrid: GPUBuffer;
    uPhiGrid: GPUBuffer;
    dzetaDphiGrid: GPUBuffer;
    dzetaDthetaGrid: GPUBuffer;
    advGrid: GPUBuffer;

    energyTerms: GPUBuffer;

    w!: GPUBuffer;
    P_lm!: GPUBuffer;
    dP_lm_dtheta!: GPUBuffer;

    constructor(device: GPUDevice, config: SimulationConfig) {
        this.device = device;
        this.config = config;

        // complex numbers represented as vec2<f32>, so each takes 8 bytes
        const spectralBytes = getSpectralSize(config) * 8;

        // real grid numbers represented as f32, so each takes 4 bytes
        const gridBytes = getGridSize(config) * 4;

        this.zetaLM_A = this.createBuffer(spectralBytes, "zetaLM_A", GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
        this.zetaLM_B = this.createBuffer(spectralBytes, "zetaLM_B", GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
        this.psiLM = this.createBuffer(spectralBytes, "psiLM");
        this.k1 = this.createBuffer(spectralBytes, "k1");
        this.k2 = this.createBuffer(spectralBytes, "k2");
        this.k3 = this.createBuffer(spectralBytes, "k3");
        this.k4 = this.createBuffer(spectralBytes, "k4");
        this.tmpLM = this.createBuffer(spectralBytes, "tmpLM", GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);

        this.zetaGrid = this.createBuffer(gridBytes, "zetaGrid", GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
        this.psiGrid = this.createBuffer(gridBytes, "psiGrid");
        this.dpsiDphiGrid = this.createBuffer(gridBytes, "dpsiDphiGrid");
        this.dpsiDthetaGrid = this.createBuffer(gridBytes, "dpsiDthetaGrid");
        this.uThetaGrid = this.createBuffer(gridBytes, "uThetaGrid");
        this.uPhiGrid = this.createBuffer(gridBytes, "uPhiGrid");
        this.dzetaDphiGrid = this.createBuffer(gridBytes, "dzetaDphiGrid");
        this.dzetaDthetaGrid = this.createBuffer(gridBytes, "dzetaDthetaGrid");
        this.advGrid = this.createBuffer(gridBytes, "advGrid");

        this.energyTerms = this.createBuffer(gridBytes, "energyTerms");
    }

    private createBuffer(size: number, label: string, usage: number = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST): GPUBuffer {
        return this.device.createBuffer({
            label,
            size,
            usage
        });
    }
}
