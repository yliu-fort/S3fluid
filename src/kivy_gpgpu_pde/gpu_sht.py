import numpy as np
from kivy_gpgpu_pde.gl_utils import create_float_texture

class GPUSHT:
    """
    GPU-accelerated Spherical Harmonics Transform.
    Uses precomputed arrays from NumPySHT and encodes them into RGBA32F Kivy textures
    for use in WebGL/GLSL shader passes.
    """
    def __init__(self, numpy_sht):
        self.sht = numpy_sht

        self.l_max = self.sht.l_max
        self.n_lat = self.sht.n_lat
        self.n_lon = self.sht.n_lon
        self.num_m = self.sht.num_m

        # Prepare textures
        self._generate_textures()

    def _generate_textures(self):
        """
        Convert NumPySHT arrays into Float32 GPU Textures.
        """
        # 1. ALP Texture: Shape (num_m, l_max + 1, n_lat) -> 2D Texture
        # We'll pack it into a 2D texture where:
        # width = n_lat
        # height = num_m * (l_max + 1)
        # channel = R (but we upload as RGBA and just use R)

        alp_tex_height = self.num_m * (self.l_max + 1)
        alp_tex_width = self.n_lat
        alp_data = np.zeros((alp_tex_height, alp_tex_width, 4), dtype=np.float32)

        # We need to map (i_m, l, lat_idx) to (y, x)
        for i_m in range(self.num_m):
            for l in range(self.l_max + 1):
                y = i_m * (self.l_max + 1) + l
                alp_data[y, :, 0] = self.sht.alp[i_m, l, :]

        self.alp_texture = create_float_texture(alp_data, alp_tex_width, alp_tex_height)

        # 2. DALP (Derivative ALP) Texture
        dalp_data = np.zeros((alp_tex_height, alp_tex_width, 4), dtype=np.float32)
        for i_m in range(self.num_m):
            for l in range(self.l_max + 1):
                y = i_m * (self.l_max + 1) + l
                dalp_data[y, :, 0] = self.sht.dalp_dtheta[i_m, l, :]

        self.dalp_texture = create_float_texture(dalp_data, alp_tex_width, alp_tex_height)

        # 3. Weights Texture: Shape (n_lat,)
        weights_data = np.zeros((1, self.n_lat, 4), dtype=np.float32)
        weights_data[0, :, 0] = self.sht.weights
        self.weights_texture = create_float_texture(weights_data, self.n_lat, 1)

        # 4. DFT Matrix Texture (Forward)
        # We compute F_m = \sum_k f(phi_k) e^{-i m phi_k}
        # Real part: cos(m phi_k), Imag part: -sin(m phi_k)
        dft_fwd_data = np.zeros((self.num_m, self.n_lon, 4), dtype=np.float32)
        phi = self.sht.lons
        for i_m, m in enumerate(self.sht.m_values):
            dft_fwd_data[i_m, :, 0] = np.cos(m * phi)
            dft_fwd_data[i_m, :, 1] = -np.sin(m * phi)

        # For FloatFbo mapping, the shape passed to create_float_texture is (height, width, channels)
        # We want width=n_lon, height=num_m. Thus we need an array of shape (num_m, n_lon, 4)
        # where data[y, x, :] corresponds to m=y, lon=x.

        self.dft_fwd_texture = create_float_texture(dft_fwd_data, self.n_lon, self.num_m)

        # 5. DFT Matrix Texture (Inverse)
        # f(phi_k) = (1/n_lon) \sum_m F_m e^{i m phi_k}
        # Since rfft only contains positive m, we handle the reconstruction
        # (F_m * e^{i m phi_k}) + (F_m^* * e^{-i m phi_k})
        # = 2 * Re(F_m e^{i m phi_k}) for m>0, and Re(F_0) for m=0

        # We store e^{i m phi_k} basis in a texture
        dft_inv_data = np.zeros((self.num_m, self.n_lon, 4), dtype=np.float32)
        for i_m, m in enumerate(self.sht.m_values):
            # Scale by 2 for m>0 due to conjugate symmetry in real FFT
            scale = 1.0 if m == 0 else 2.0
            dft_inv_data[i_m, :, 0] = scale * np.cos(m * phi)
            dft_inv_data[i_m, :, 1] = scale * np.sin(m * phi) # We want to compute Re(F * basis) = Re(F)*Re(basis) - Im(F)*Im(basis)

        # The inverse DFT texture maps from m to phi.
        # Height = n_lon, Width = num_m might be better for cache, but let's stick to consistent sizes
        # For shader simplicity, height = n_lon, width = num_m
        dft_inv_data_transposed = np.zeros((self.n_lon, self.num_m, 4), dtype=np.float32)
        for i_m in range(self.num_m):
            dft_inv_data_transposed[:, i_m, 0] = dft_inv_data[i_m, :, 0]
            dft_inv_data_transposed[:, i_m, 1] = dft_inv_data[i_m, :, 1]

        self.dft_inv_texture = create_float_texture(dft_inv_data_transposed, self.num_m, self.n_lon)
