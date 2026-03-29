import * as THREE from 'three';
import { sphereVertGLSL, scalarColorFragGLSL } from './glslShaders';

export class SphereView {
    public renderer!: THREE.WebGLRenderer;
    public scene!: THREE.Scene;
    public camera!: THREE.PerspectiveCamera;
    public sphereMesh!: THREE.Mesh;

    constructor(canvas: HTMLCanvasElement) {
        // Need to use WebGLRenderer initially if Three's WebGPURenderer is still highly experimental
        // Or import WebGPURenderer if available
        if (typeof jest === 'undefined') {
            this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
            this.renderer.setSize(window.innerWidth, window.innerHeight);
            this.renderer.setPixelRatio(window.devicePixelRatio);
        } else {
            // Mock renderer for jest tests without a real WebGL context
            this.renderer = {
                setSize: () => {},
                setPixelRatio: () => {},
                render: () => {},
            } as unknown as THREE.WebGLRenderer;
        }

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
        this.camera.position.z = 3;

        const geometry = new THREE.SphereGeometry(1, 64, 64);

        const material = new THREE.ShaderMaterial({
            vertexShader: sphereVertGLSL,
            fragmentShader: scalarColorFragGLSL,
            uniforms: {
                scalarTexture: { value: null },
                displayScale: { value: 1.0 }
            }
        });

        this.sphereMesh = new THREE.Mesh(geometry, material);

        this.scene.add(this.sphereMesh);

        window.addEventListener('resize', this.onWindowResize.bind(this));
    }

    private onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    render() {
        this.renderer.render(this.scene, this.camera);
    }

    updateTexture(texture: THREE.Texture) {
        if (this.sphereMesh.material instanceof THREE.ShaderMaterial) {
            this.sphereMesh.material.uniforms.scalarTexture.value = texture;
            this.sphereMesh.material.uniformsNeedUpdate = true;
        } else if (this.sphereMesh.material instanceof THREE.MeshBasicMaterial) {
            this.sphereMesh.material.map = texture;
            this.sphereMesh.material.needsUpdate = true;
        }
    }

    setDisplayScale(scale: number) {
        if (this.sphereMesh.material instanceof THREE.ShaderMaterial) {
            this.sphereMesh.material.uniforms.displayScale.value = scale;
            this.sphereMesh.material.uniformsNeedUpdate = true;
        }
    }
}
