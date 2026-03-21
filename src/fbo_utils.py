from kivy.graphics.fbo import Fbo
from kivy.graphics.texture import Texture
import numpy as np
import ctypes
from .gl_utils import gl_lib

GL_FLOAT = 0x1406
GL_RGBA32F = 0x8814
GL_RGBA = 0x1908
GL_TEXTURE_2D = 0x0DE1

def create_float_fbo(width, height):
    """
    Creates an FBO with a 32-bit float texture.
    """
    tex = Texture.create(size=(width, height), colorfmt='rgba', bufferfmt='float')
    if gl_lib:
        tex.bind()
        # Kivy might not use GL_RGBA32F internally, which clips colors to 0-1.
        # We manually re-allocate the texture with GL_RGBA32F internal format.
        # void glTexImage2D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels);
        gl_lib.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, None)

    fbo = Fbo(size=(width, height), texture=tex, with_depthbuffer=False)
    # The Fbo will attach the texture using glFramebufferTexture2D.
    return fbo

def fbo_to_ndarray(fbo):
    if gl_lib is None:
        raise RuntimeError("OpenGL library could not be loaded via ctypes. Cannot read float pixels.")

    width, height = fbo.size
    data = np.zeros((height, width, 4), dtype=np.float32)
    fbo.bind()
    gl_lib.glReadPixels(
        ctypes.c_int(0),
        ctypes.c_int(0),
        ctypes.c_int(width),
        ctypes.c_int(height),
        ctypes.c_uint(GL_RGBA),
        ctypes.c_uint(GL_FLOAT),
        data.ctypes.data_as(ctypes.c_void_p)
    )
    fbo.release()
    return data
