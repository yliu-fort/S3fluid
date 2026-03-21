---VERTEX SHADER---
#version 330 core

in vec2 vPosition;
in vec2 vTexCoords0;

out vec2 tex_coord;

void main(void) {
    tex_coord = vTexCoords0;
    gl_Position = vec4(vPosition, 0.0, 1.0);
}

---FRAGMENT SHADER---
#version 330 core

// Input coordinates
in vec2 tex_coord;

// Output float texture
out vec4 frag_color;

// System inputs
uniform int l_max;
uniform int n_lat;
uniform int n_lon;
uniform int num_m;

// Texture Inputs
uniform sampler2D physical_data_tex; // The mesh input field (n_lon x n_lat)
uniform sampler2D dft_fwd_tex;       // The DFT Matrix (n_lon x num_m). R: cos, G: -sin
uniform sampler2D alp_tex;           // The Associated Legendre Polynomials (n_lat x (num_m*(l_max+1)))
uniform sampler2D weights_tex;       // Gauss-Legendre Weights (n_lat x 1)

void main(void) {
    // Current texel coordinate specifies which spectral coefficient (l, m) we are calculating
    // In our spectral texture, x is the m index, y is the l index.
    int m_idx = int(floor(gl_FragCoord.x));
    int l_idx = int(floor(gl_FragCoord.y));

    // Valid coefficient check. For spherical harmonics, l >= m.
    // In our texture, the width is num_m, height is (l_max + 1).
    // m_idx corresponds to an actual m value from rfft.
    // Assuming m_idx goes 0, 1, 2... up to num_m - 1.
    // Which means m = m_idx.
    int m = m_idx;

    if (l_idx < m || l_idx > l_max || m_idx >= num_m) {
        frag_color = vec4(0.0);
        return;
    }

    // We want to compute:
    // C_{l, m} = \sum_{j=0}^{n_lat - 1} W_j P_l^m(\cos\theta_j) \left[ \frac{2\pi}{N_{lon}} \sum_{k=0}^{n_lon - 1} f(\theta_j, \phi_k) e^{-i m \phi_k} \right]

    vec2 coeff = vec2(0.0); // Real and imaginary parts

    // For ALP texture, we need the exact row for (m, l)
    int y_alp_int = m_idx * (l_max + 1) + l_idx;

    for (int j = 0; j < n_lat; j++) {
        // Physical latitude coordinate
        // In Kivy textures loaded via FBO write_pixels_float, data mapping corresponds to exactly (x, y)
        // with origin at bottom-left. We use texelFetch for absolute precision mapping.

        // Weight
        float w = texelFetch(weights_tex, ivec2(j, 0), 0).r;

        // ALP value
        float p_lm = texelFetch(alp_tex, ivec2(j, y_alp_int), 0).r;

        // Compute the 1D DFT for this latitude at the specific m
        vec2 f_m = vec2(0.0);

        for (int k = 0; k < n_lon; k++) {
            // Sample physical value f(theta_j, phi_k)
            float f_val = texelFetch(physical_data_tex, ivec2(k, j), 0).r;

            // DFT basis e^{-i m phi_k}
            // Note: dft_fwd_tex size is (n_lon, num_m). width=n_lon, height=num_m
            vec2 basis = texelFetch(dft_fwd_tex, ivec2(k, m_idx), 0).rg;

            // accumulate
            f_m += f_val * basis;
        }

        // Multiply by weight and ALP, and accumulate
        coeff += f_m * w * p_lm;
    }

    // Normalize DFT (numpy's rfft doesn't normalize by N, but numpy_sht multiplies by 2pi/N)
    coeff *= (2.0 * 3.141592653589793) / float(n_lon);

    // Output complex coefficient to R and G channels
    frag_color = vec4(coeff.x, coeff.y, 0.0, 1.0);
}
