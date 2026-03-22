import { createConfig, SimulationConfig } from "../../src/solver/config";
import { precompute, PrecomputedData } from "../../src/solver/precompute";
import * as fs from 'fs';
import * as path from 'path';

const initRandomCode = fs.readFileSync(path.join(__dirname, '../../src/shaders/initRandom.wgsl'), 'utf8');

describe('Initialization', () => {
    let config: SimulationConfig;
    let precomp: PrecomputedData;

    beforeAll(() => {
        config = createConfig({ lmax: 31 });
        precomp = precompute(config);
    });

    it('Code compiles without gross WGSL syntax errors', () => {
        expect(initRandomCode).toContain('@compute');
    });

});
