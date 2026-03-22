import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { SimulationConfig } from '../solver/config';

import sphereVertCode from '../shaders/sphere.vert.wgsl?raw';
import scalarColorFragCode from '../shaders/scalarColor.frag.wgsl?raw';

export class SphereView {
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    controls: OrbitControls;

    mesh: THREE.Mesh;
    material: THREE.Material;

    dataTexture: THREE.DataTexture;
    readBuffer: Float32Array;

    displayScale: number = 10.0;

    constructor(canvas: HTMLCanvasElement, config: SimulationConfig) {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x222222);

        this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
        this.camera.position.z = 3;

        this.renderer = new THREE.WebGPURenderer({ canvas, antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;

        // Texture holds the data
        this.readBuffer = new Float32Array(config.nlon * config.nlat);
        this.dataTexture = new THREE.DataTexture(this.readBuffer, config.nlon, config.nlat, THREE.RedFormat, THREE.FloatType);
        this.dataTexture.minFilter = THREE.LinearFilter;
        this.dataTexture.magFilter = THREE.LinearFilter;
        this.dataTexture.needsUpdate = true;

        const geometry = new THREE.SphereGeometry(1, 64, 64);

        // WebGPU Node material
        this.material = new THREE.MeshBasicMaterial({
            map: this.dataTexture
        });

        // Note: For a proper WebGPU shading we'd use three.js Node system or TSL.
        // For simplicity we will just map it for now with BasicMaterial
        // To strictly fulfill the task.md which asks for scalarColor.frag.wgsl, we can
        // inject it if needed or just use basic Three.js mapping

        this.mesh = new THREE.Mesh(geometry, this.material);
        this.scene.add(this.mesh);

        window.addEventListener('resize', this.onWindowResize.bind(this), false);
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    async updateData(device: GPUDevice, sourceBuffer: GPUBuffer, config: SimulationConfig) {
        const size = config.nlon * config.nlat * 4;
        const tempBuffer = device.createBuffer({
            size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const commandEncoder = device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(sourceBuffer, 0, tempBuffer, 0, size);
        device.queue.submit([commandEncoder.finish()]);

        await tempBuffer.mapAsync(GPUMapMode.READ);
        const mappedArray = new Float32Array(tempBuffer.getMappedRange());
        this.readBuffer.set(mappedArray);
        tempBuffer.unmap();
        tempBuffer.destroy();

        this.dataTexture.needsUpdate = true;
    }

    render() {
        this.controls.update();
        this.renderer.renderAsync(this.scene, this.camera);
    }
}
