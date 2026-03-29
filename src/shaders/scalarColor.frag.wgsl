// Colormap mapping
// We want to map [ -scale, scale ] -> cool to warm
// Let's implement a simple RdBu colormap in WGSL.

@group(0) @binding(0) var<uniform> displayScale: f32;
@group(0) @binding(1) var scalarTexture: texture_2d<f32>;
@group(0) @binding(2) var textureSampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Simple approximation of a diverging colormap
fn colormap(val: f32) -> vec3<f32> {
    let t = clamp((val + 1.0) * 0.5, 0.0, 1.0);
    // Dark blue to white to dark red
    let color_blue = vec3<f32>(0.0, 0.0, 1.0);
    let color_white = vec3<f32>(1.0, 1.0, 1.0);
    let color_red = vec3<f32>(1.0, 0.0, 0.0);

    if (t < 0.5) {
        return mix(color_blue, color_white, t * 2.0);
    } else {
        return mix(color_white, color_red, (t - 0.5) * 2.0);
    }
}

@fragment
fn main(in: VertexOutput) -> @location(0) vec4<f32> {
    let scalar = textureSample(scalarTexture, textureSampler, in.uv).r;
    let normalized = scalar / displayScale;
    let color = colormap(normalized);
    return vec4<f32>(color, 1.0);
}
