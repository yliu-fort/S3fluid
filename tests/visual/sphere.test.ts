/**
 * @jest-environment jsdom
 */

import { SphereView } from '../../src/render/sphereView';
import * as THREE from 'three';

describe('SphereView', () => {
    let mockCanvas: HTMLCanvasElement;

    beforeEach(() => {
        mockCanvas = document.createElement('canvas');

        // Mocking canvas API that Three uses
        const getParameter = jest.fn((param) => {
            if (param === 7938) return 'WebGL 1.0';
            if (param === 35661) return 80;
            if (param === 34930) return 16;
            if (param === 34076) return 16384;
            if (param === 34024) return 16384;
            if (param === 3379) return 16384;
            if (param === 34921) return 16;
            if (param === 35660) return 16;
            if (param === 34852) return 'WebGL GLSL ES 1.0';
            return 0;
        });

        mockCanvas.getContext = jest.fn().mockReturnValue({
            getParameter,
            getShaderPrecisionFormat: jest.fn().mockReturnValue({ precision: 1, rangeMin: 1, rangeMax: 1 }),
            getExtension: jest.fn().mockReturnValue(null),
            createTexture: jest.fn(),
            bindTexture: jest.fn(),
            texImage2D: jest.fn(),
            texParameteri: jest.fn(),
            clearColor: jest.fn(),
            clearDepth: jest.fn(),
            clearStencil: jest.fn(),
            enable: jest.fn(),
            disable: jest.fn(),
            depthFunc: jest.fn(),
            frontFace: jest.fn(),
            cullFace: jest.fn(),
            createBuffer: jest.fn(),
            bindBuffer: jest.fn(),
            bufferData: jest.fn(),
            createProgram: jest.fn(),
            createShader: jest.fn(),
            shaderSource: jest.fn(),
            compileShader: jest.fn(),
            getShaderParameter: jest.fn().mockReturnValue(true),
            attachShader: jest.fn(),
            linkProgram: jest.fn(),
            getProgramParameter: jest.fn().mockReturnValue(true),
            useProgram: jest.fn(),
            viewport: jest.fn(),
            clear: jest.fn(),
            drawElements: jest.fn(),
            getShaderPrecisionFormat: jest.fn().mockReturnValue({ precision: 1, rangeMin: 1, rangeMax: 1 }),
        });

        // Object.defineProperty is required to mock readonly client properties
        Object.defineProperty(window, 'innerWidth', { value: 800 });
        Object.defineProperty(window, 'innerHeight', { value: 600 });
    });

    it('initializes the three.js view', () => {
        const view = new SphereView(mockCanvas);

        expect(view.renderer).toBeDefined();
        expect(view.scene).toBeDefined();
        expect(view.camera).toBeDefined();
        expect(view.sphereMesh).toBeDefined();

        expect(view.camera.aspect).toBeCloseTo(800 / 600, 5);
        expect(view.sphereMesh.geometry instanceof THREE.SphereGeometry).toBe(true);
    });

    it('can set a texture', () => {
        const view = new SphereView(mockCanvas);
        const tex = new THREE.Texture();

        view.updateTexture(tex);

        const mat = view.sphereMesh.material as THREE.ShaderMaterial;
        expect(mat.uniforms.scalarTexture.value).toBe(tex);
        expect(mat.uniformsNeedUpdate).toBe(true);

        view.setDisplayScale(2.5);
        expect(mat.uniforms.displayScale.value).toBe(2.5);
    });
});
