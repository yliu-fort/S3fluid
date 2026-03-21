import ctypes
import ctypes.util
import numpy as np
from kivy.graphics import Fbo
from kivy.graphics.texture import Texture

# OpenGL constants
GL_RGBA = 0x1908
GL_FLOAT = 0x1406
GL_RGBA32F = 0x8814
GL_TEXTURE_2D = 0x0DE1

# Load OpenGL library
if ctypes.util.find_library('GL'):
    gl_lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library('GL'))
elif ctypes.util.find_library('opengl32'):
    gl_lib = ctypes.windll.LoadLibrary('opengl32')
else:
    raise RuntimeError("Could not find OpenGL library")

_glReadPixels = gl_lib.glReadPixels
_glReadPixels.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
_glReadPixels.restype = None

_glTexImage2D = gl_lib.glTexImage2D
_glTexImage2D.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
_glTexImage2D.restype = None

def glReadPixels_float(x, y, width, height):
    """
    Read pixels from the current framebuffer as a flat float32 numpy array.
    """
    buffer = np.zeros(width * height * 4, dtype=np.float32)
    _glReadPixels(x, y, width, height, GL_RGBA, GL_FLOAT, buffer.ctypes.data_as(ctypes.c_void_p))
    return buffer

def glTexImage2D_float(texture, data, width, height):
    """
    Uploads float32 numpy array to the bound texture with GL_RGBA32F internal format.
    """
    texture.bind()
    _glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, data.ctypes.data_as(ctypes.c_void_p))

class FloatFbo(Fbo):
    """
    Custom Fbo wrapper that strictly ensures GL_RGBA32F precision.
    Reads and writes data as flat numpy float32 arrays without clamping to [0, 1].
    """
    def __init__(self, size, **kwargs):
        kwargs['size'] = size
        kwargs['with_depthbuffer'] = False
        # Kivy 2.3+ might not support 'rgba32f' directly in colorfmt, we use 'rgba' and then manually upgrade
        kwargs.setdefault('colorfmt', 'rgba')
        super().__init__(**kwargs)

        # Upgrade texture to 32-bit float
        empty_data = np.zeros(size[0] * size[1] * 4, dtype=np.float32)
        glTexImage2D_float(self.texture, empty_data, size[0], size[1])

    def read_pixels_float(self):
        """
        Bind the FBO and read its contents back into CPU memory as a numpy array.
        Returns array of shape (height, width, 4) in float32.
        """
        self.bind()
        width, height = self.size
        # Use our custom ctypes call
        data = glReadPixels_float(0, 0, width, height)
        self.release()

        # Reshape to (height, width, 4)
        reshaped = data.reshape((height, width, 4))
        return np.ascontiguousarray(reshaped)

    def write_pixels_float(self, data):
        """
        Write a numpy array of shape (height, width, 4) or flat to this FBO's texture.
        """
        data = np.asarray(data, dtype=np.float32)
        if data.size != self.size[0] * self.size[1] * 4:
            raise ValueError(f"Data size {data.size} does not match FBO size {self.size[0] * self.size[1] * 4}")

        # For pure array mapping in texelFetch without Kivy's projection matrices interfering,
        # we actually DO NOT WANT to flip here, otherwise texelFetch(..., ivec2(x, y)) reads
        # upside-down relative to the logical numpy array coordinates.

        height, width = self.size[1], self.size[0]
        reshaped = data.reshape((height, width, 4))
        contiguous_data = np.ascontiguousarray(reshaped, dtype=np.float32)

        glTexImage2D_float(self.texture, contiguous_data, width, height)

def create_float_texture(data, width, height):
    """
    Helper to create a standalone RGBA32F Kivy Texture from a numpy array.
    """
    # Create an Fbo to ensure the texture gets a proper framebuffer binding
    # and RGBA32F allocation without being discarded by Kivy's rendering queue.
    fbo = FloatFbo(size=(width, height))
    fbo.texture.mag_filter = 'nearest'
    fbo.texture.min_filter = 'nearest'

    data = np.asarray(data, dtype=np.float32)
    fbo.write_pixels_float(data)

    # Return the FBO's texture
    return fbo.texture
