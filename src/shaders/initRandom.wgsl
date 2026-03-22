struct Params {
    nlat: u32,
    nlon: u32,
    M: u32,
    L: u32,
    seed: f32,
    amplitude: f32,
}

@group(0) @binding(0) var<storage, read> initSlope : array<f32>;
@group(0) @binding(1) var<storage, read> specFilter : array<f32>;
@group(0) @binding(2) var<storage, read_write> zetaLM : array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params : Params;

// Simple LCG PRNG for WGSL
fn rand(state: ptr<function, u32>) -> f32 {
    let s = *state;
    // LCG parameters from Numerical Recipes
    let next = (s * 1664525u + 1013904223u);
    *state = next;
    // Map to [0, 1]
    return f32(next) / 4294967296.0;
}

// Box-Muller transform for standard normal variables
fn rand_normal(state: ptr<function, u32>) -> f32 {
    let u = max(1.0 - rand(state), 1e-7); // Avoid log(0)
    let v = rand(state);
    return sqrt(-2.0 * log(u)) * cos(6.28318530718 * v);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let m = global_id.x;
    let l = global_id.y;

    if (m >= params.M || l >= params.L) {
        return;
    }

    let idx = m * params.L + l;

    // Default to zero
    zetaLM[idx] = vec2<f32>(0.0, 0.0);

    if (l >= m) {
        // Compute seed specific to this (m,l)
        // Combine base seed with mode indices to get deterministic noise per mode
        let base_seed = bitcast<u32>(params.seed);
        let mode_seed = base_seed ^ (m * 73856093u) ^ (l * 19349663u);

        var state = mode_seed;

        // Generate random coefficients in spectral space directly
        // Note: Real model initializes in grid space and transforms.
        // We will approximate it by just generating random modes directly, scaled by slope.
        // This is equivalent since white noise in grid is white noise in spectral space.

        var aRe = rand_normal(&state);
        var aIm = rand_normal(&state);

        if (m == 0) {
            aIm = 0.0; // m=0 modes are purely real
        }

        let slope = initSlope[idx];
        let filter = specFilter[idx];

        aRe = aRe * slope * filter * params.amplitude;
        aIm = aIm * slope * filter * params.amplitude;

        // Force l=0 to be 0
        if (l == 0 && m == 0) {
            aRe = 0.0;
            aIm = 0.0;
        }

        zetaLM[idx] = vec2<f32>(aRe, aIm);
    }
}
