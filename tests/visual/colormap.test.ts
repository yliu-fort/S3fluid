import { createScalarDataTexture } from '../../src/render/colormap';
import * as THREE from 'three';

describe('Colormap Utils', () => {
    it('creates a DataTexture with Float32Array', () => {
        const data = new Float32Array([1.0, 2.0, 3.0, 4.0]);
        const texture = createScalarDataTexture(data, 2, 2);

        expect(texture).toBeDefined();
        expect(texture.image.data).toBe(data);
        expect(texture.image.width).toBe(2);
        expect(texture.image.height).toBe(2);
        expect(texture.type).toBe(THREE.FloatType);
        expect(texture.format).toBe(THREE.RedFormat);
        // It uses setter in ThreeJS, needsUpdate internally sets version
        expect(texture.version).toBeGreaterThan(0);
    });
});
