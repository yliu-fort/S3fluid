import numpy as np
from kivy.graphics import RenderContext, BindTexture, Rectangle
from kivy.graphics.texture import Texture
from .fbo_utils import GPGPUFbo
from .gl_utils import GL_RGBA32F, GL_RGBA, GL_FLOAT, _gl_lib, check_gl_error

def create_float32_texture(width, height, data=None):
    """
    Creates a float32 Kivy texture. If data is provided, it uploads the numpy array data.
    """
    tex = Texture.create(size=(width, height), colorfmt='rgba', bufferfmt='float')
    # Set filtering to NEAREST to avoid interpolating between math data
    tex.mag_filter = 'nearest'
    tex.min_filter = 'nearest'
    tex.wrap = 'clamp_to_edge'

    tex.bind()
    if _gl_lib:
        import ctypes
        data_ptr = None
        if data is not None:
            # ensure float32 flat buffer
            if not isinstance(data, np.ndarray) or data.dtype != np.float32:
                data = np.array(data, dtype=np.float32)
            data_ptr = data.ctypes.data_as(ctypes.c_void_p)

        _gl_lib.glTexImage2D(0x0DE1, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, data_ptr)
        check_gl_error()

    return tex


class SHT_GPU:
    def __init__(self, numpy_sht):
        """
        Initializes the GPU SHT components using precomputed matrices from NumPySHT.
        """
        self.nsht = numpy_sht
        self.n_lat = self.nsht.n_lat
        self.n_lon = self.nsht.n_lon
        self.num_m = self.nsht.num_m
        self.l_max = self.nsht.l_max

        # FBO for storing spectral coefficients
        # For simplicity, we can store spectra as a 2D float texture: width=l_max+1, height=num_m
        # Real and Imag parts will go into RG and BA or similar.
        self.spectrum_fbo = GPGPUFbo(size=(self.l_max + 1, self.num_m))
        self.grid_fbo = GPGPUFbo(size=(self.n_lon, self.n_lat))

        self._upload_matrices()

    def _upload_matrices(self):
        """
        Uploads Legendre polynomials, weights and their derivatives to float32 textures.
        """
        # Encode self.nsht.alp (num_m, l_max+1, n_lat) into a texture.
        # This requires flattening or mapping 3D into 2D texture map.
        # We can create a large 2D texture: width = n_lat, height = num_m * (l_max+1)
        w = self.n_lat
        h = self.num_m * (self.l_max + 1)

        # alp shape: (num_m, l_max+1, n_lat)
        # We'll store it in the R channel, and dalp_dtheta in the G channel, weights in B.
        data = np.zeros((h, w, 4), dtype=np.float32)

        # weights are 1D: size n_lat
        # Just tile it for every row in the texture for convenience, or upload as separate texture.
        # For memory efficiency, a separate 1D texture (or 2D 1xN) is better for weights.

        for i_m in range(self.num_m):
            for l in range(self.l_max + 1):
                row = i_m * (self.l_max + 1) + l
                data[row, :, 0] = self.nsht.alp[i_m, l, :]
                data[row, :, 1] = self.nsht.dalp_dtheta[i_m, l, :]
                data[row, :, 2] = self.nsht.weights[:]  # weights

        # To handle complex values seamlessly in shader, we ensure precise layout.
        self.tex_legendre = create_float32_texture(w, h, data)

    def forward_sht(self, grid_texture):
        """
        Pass 1: Forward SHT (Physics -> Spectrum).
        This is a placeholder for the actual shader dispatch.
        """
        # Set up a RenderContext for forward pass
        rc = RenderContext(use_parent_modelview=True, use_parent_projection=True)
        # Pseudo-shader logic for forward pass:
        # FFT along longitude, then Gauss-Legendre quadrature along latitude.
        # In actual implementation, we might need a multi-pass approach (FFT pass then Legendre pass).
        # We'll mock the shader for UT purposes to pass data structure mapping.

        rc.shader.fs = '''
        $HEADER$
        uniform sampler2D tex_grid;
        uniform sampler2D tex_legendre;
        void main(void) {
            // Placeholder: output mock coefficient based on texture fetch
            gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        }
        '''
        with rc:
            BindTexture(texture=grid_texture, index=1)
            rc['tex_grid'] = 1
            BindTexture(texture=self.tex_legendre, index=2)
            rc['tex_legendre'] = 2
            Rectangle(pos=(0, 0), size=self.spectrum_fbo.size)

        self.spectrum_fbo.add(rc)
        self.spectrum_fbo.draw()
        self.spectrum_fbo.remove(rc)
        return self.spectrum_fbo

    def inverse_sht(self, spectrum_texture):
        """
        Pass 2: Inverse SHT (Spectrum -> Physics).
        """
        rc = RenderContext(use_parent_modelview=True, use_parent_projection=True)
        rc.shader.fs = '''
        $HEADER$
        uniform sampler2D tex_spectrum;
        uniform sampler2D tex_legendre;
        void main(void) {
            // Placeholder: output mock grid based on texture fetch
            gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0);
        }
        '''
        with rc:
            BindTexture(texture=spectrum_texture, index=1)
            rc['tex_spectrum'] = 1
            BindTexture(texture=self.tex_legendre, index=2)
            rc['tex_legendre'] = 2
            Rectangle(pos=(0, 0), size=self.grid_fbo.size)

        self.grid_fbo.add(rc)
        self.grid_fbo.draw()
        self.grid_fbo.remove(rc)
        return self.grid_fbo

    def inverse_sht_grad(self, spectrum_texture):
        """
        Pass 3: Inverse SHT + Gradient
        """
        rc = RenderContext(use_parent_modelview=True, use_parent_projection=True)
        rc.shader.fs = '''
        $HEADER$
        uniform sampler2D tex_spectrum;
        uniform sampler2D tex_legendre;
        void main(void) {
            // Placeholder: output mock gradient
            gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0);
        }
        '''
        with rc:
            BindTexture(texture=spectrum_texture, index=1)
            rc['tex_spectrum'] = 1
            BindTexture(texture=self.tex_legendre, index=2)
            rc['tex_legendre'] = 2
            Rectangle(pos=(0, 0), size=self.grid_fbo.size)

        self.grid_fbo.add(rc)
        self.grid_fbo.draw()
        self.grid_fbo.remove(rc)
        return self.grid_fbo
