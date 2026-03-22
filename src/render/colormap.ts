// Since three.js uses WebGL out of the box currently unless WebGPU Node renderer is used,
// for CPU-side texture creation we might need to rely on standard Float32Array texture mapping.
// Alternatively, if we pass float arrays directly we can configure a Three DataTexture.

import * as THREE from 'three';

export function createScalarDataTexture(data: Float32Array, width: number, height: number): THREE.DataTexture {
    const texture = new THREE.DataTexture(data, width, height, THREE.RedFormat, THREE.FloatType);
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.generateMipmaps = false;
    texture.needsUpdate = true;
    return texture;
}
