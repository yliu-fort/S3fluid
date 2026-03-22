// Inverse FFT recovering purely real output grid.
// Maps F(j, m) -> grid(j, k)
//
// Workgroup setup:
// x: Longitude (k index)
// y: Latitude (j index)
// z: 1

@group(0) @binding(0) var<storage, read> freq_in: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> grid_out: array<f32>;
@group(0) @binding(2) var<storage, read> phi: array<f32>;

struct Config {
    nlat: u32,
    nlon: u32,
    lmax: u32,
    pad: u32
}

@group(1) @binding(0) var<uniform> config: Config;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k = global_id.x;
    let j = global_id.y;

    if (k >= config.nlon || j >= config.nlat) {
        return;
    }

    let m_limit = config.lmax + 1u;

    var sum_val = 0.0;

    for (var m = 0u; m < m_limit; m = m + 1u) {
        let f = freq_in[j * m_limit + m];
        let angle = f32(m) * phi[k];
        let term = f.x * cos(angle) - f.y * sin(angle);

        if (m == 0u) {
            sum_val = sum_val + term;
        } else {
            sum_val = sum_val + 2.0 * term;
        }
    }

    let out_idx = j * config.nlon + k;
    grid_out[out_idx] = sum_val / f32(config.nlon);
}
