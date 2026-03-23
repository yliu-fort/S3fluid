@group(1) @binding(0) var gridTexture: texture_2d<f32>;
@group(1) @binding(1) var gridSampler: sampler;

struct Params {
    displayScale: f32
}
@group(1) @binding(2) var<uniform> params: Params;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
};

fn colormap(val: f32) -> vec4<f32> {
    // A simple diverging colormap: Blue (-) to White (0) to Red (+)
    let clamped_val = clamp(val, -1.0, 1.0);

    if (clamped_val < 0.0) {
        // -1 (Blue) to 0 (White)
        let t = clamped_val + 1.0;
        return vec4<f32>(t, t, 1.0, 1.0);
    } else {
        // 0 (White) to 1 (Red)
        let t = 1.0 - clamped_val;
        return vec4<f32>(1.0, t, t, 1.0);
    }
}

@fragment
fn main(in: VertexOutput) -> @location(0) vec4<f32> {
    // sample the texture
    let val = textureSample(gridTexture, gridSampler, in.uv).r;

    // scale
    let scaled = val / params.displayScale;

    return colormap(scaled);
}
