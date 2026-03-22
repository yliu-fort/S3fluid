import { SOLVER_CONFIG } from './config';
import { PrecomputedData } from './precompute';
import { SolverBuffers } from './buffers';

// ... (previous PipelineManager class implementation)
export class PipelineManager {
    device: GPUDevice;
    moduleCache: Map<string, GPUShaderModule>;

    constructor(device: GPUDevice) {
        this.device = device;
        this.moduleCache = new Map();
    }

    // A helper method for creating shader modules if loading from files directly isn't feasible in a browser environment
    // Instead of fs, we will use a hardcoded mapping or assume a build system bundles them if needed.
    // For the sake of this demo without a bundler, we might need a way to pass the shader code as strings.
    // But since we wrote the files, a bundler or Vite would resolve them.
    // For pure Typescript setup, we will define a fallback or assume fetch works.

    async loadShaderCode(shaderName: string): Promise<string> {
        // If we are in browser, we might use fetch
        if (typeof process !== 'undefined' && process.versions && process.versions.node && typeof fetch === 'undefined') {
            // Node.js environment (Jest tests)
            const fs = require('fs');
            const path = require('path');
            const p = path.join(__dirname, '..', 'shaders', shaderName);
            return fs.readFileSync(p, 'utf8');
        } else {
            if (typeof process !== 'undefined' && process.versions && process.versions.node) {
                // Jest might have global fetch mocked, fallback to readFileSync
                const fs = require('fs');
                const path = require('path');
                const p = path.join(__dirname, '..', 'shaders', shaderName);
                return fs.readFileSync(p, 'utf8');
            }
            const res = await fetch(`/src/shaders/${shaderName}`);
            return await res.text();
        }
    }

    async loadShader(shaderName: string): Promise<GPUShaderModule> {
        if (this.moduleCache.has(shaderName)) {
            return this.moduleCache.get(shaderName)!;
        }

        const code = await this.loadShaderCode(shaderName);

        const module = this.device.createShaderModule({
            label: shaderName,
            code: code,
        });

        this.moduleCache.set(shaderName, module);
        return module;
    }

    async createComputePipeline(shaderName: string): Promise<GPUComputePipeline> {
        const module = await this.loadShader(shaderName);
        return this.device.createComputePipeline({
            label: `Pipeline_${shaderName}`,
            layout: 'auto',
            compute: {
                module,
                entryPoint: 'main',
            },
        });
    }
}
