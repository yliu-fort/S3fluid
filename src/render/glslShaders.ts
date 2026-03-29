// GLSL versions of the WGSL shaders for Three.js WebGLRenderer fallback.
// Three.js currently strictly uses GLSL for ShaderMaterial.

export const sphereVertGLSL = `
varying vec2 vUv;
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const scalarColorFragGLSL = `
uniform sampler2D scalarTexture;
uniform float displayScale;
varying vec2 vUv;

vec3 colormap(float val) {
    float t = clamp((val + 1.0) * 0.5, 0.0, 1.0);
    vec3 color_blue = vec3(0.0, 0.0, 1.0);
    vec3 color_white = vec3(1.0, 1.0, 1.0);
    vec3 color_red = vec3(1.0, 0.0, 0.0);

    if (t < 0.5) {
        return mix(color_blue, color_white, t * 2.0);
    } else {
        return mix(color_white, color_red, (t - 0.5) * 2.0);
    }
}

void main() {
    float scalar = texture2D(scalarTexture, vUv).r;
    float normalized = scalar / displayScale;
    vec3 color = colormap(normalized);
    gl_FragColor = vec4(color, 1.0);
}
`;
