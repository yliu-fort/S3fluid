from kivy.graphics import Fbo
from .gl_utils import read_fbo_pixels_float32

class GPGPUFbo(Fbo):
    """
    A wrapper around Kivy's Fbo that uses colorfmt='rgba32f' for high dynamic range
    floating point textures, necessary for PDE solvers.
    """
    def __init__(self, size, **kwargs):
        # We explicitly request rgba32f to hold the 32-bit floating point data.
        kwargs['colorfmt'] = 'rgba32f'
        kwargs.setdefault('with_depthbuffer', False)
        kwargs.setdefault('with_stencilbuffer', False)
        super().__init__(size=size, **kwargs)
        # Store colorfmt manually because Fbo in older Kivy versions might not expose it as a property
        self.colorfmt = 'rgba32f'
        # Ensure underlying texture has the correct float storage since some versions of Kivy fall back to rgba
        from .gl_utils import GL_RGBA32F, GL_RGBA, GL_FLOAT, _gl_lib, check_gl_error
        self.texture.bind()
        if _gl_lib:
            # Override internal format to GL_RGBA32F
            _gl_lib.glTexImage2D(0x0DE1, 0, GL_RGBA32F, size[0], size[1], 0, GL_RGBA, GL_FLOAT, None)
            check_gl_error()
        self.bind()
        self.release()

    def read_pixels_float32(self):
        """
        Reads the FBO's pixels into a float32 numpy array.
        This must be called when the FBO is bound.
        """
        self.bind()
        try:
            width, height = self.size
            return read_fbo_pixels_float32(0, 0, width, height)
        finally:
            self.release()

    def get_texture(self):
        """
        Returns the FBO's texture.
        """
        return self.texture
