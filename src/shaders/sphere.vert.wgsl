@vertex
fn main(
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>
) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(position, 1.0);
    out.uv = uv;
    return out;
}
