import ctypes
import ctypes.util
import os
import sys

# Try to find and load OpenGL library
_gl_lib = None
if sys.platform.startswith('win'):
    _gl_lib = ctypes.windll.opengl32
elif sys.platform.startswith('darwin'):
    # macOS
    from ctypes.macholib.dyld import framework_find
    path = framework_find('OpenGL')
    if path:
        _gl_lib = ctypes.cdll.LoadLibrary(path)
else:
    # Linux
    path = ctypes.util.find_library('GL')
    if not path:
        path = 'libGL.so.1'
    try:
        _gl_lib = ctypes.cdll.LoadLibrary(path)
    except OSError:
        pass

if not _gl_lib:
    raise RuntimeError("Could not load OpenGL library")

# OpenGL Constants
GL_RGBA = 0x1908
GL_FLOAT = 0x1406
GL_RGBA32F = 0x8814
GL_TEXTURE_2D = 0x0DE1
GL_COLOR_ATTACHMENT0 = 0x8CE0
GL_FRAMEBUFFER_COMPLETE = 0x8CD5
GL_FRAMEBUFFER = 0x8D40

# Setup ctypes signatures
try:
    _glTexImage2D = _gl_lib.glTexImage2D
    _glTexImage2D.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
    _glTexImage2D.restype = None

    _glReadPixels = _gl_lib.glReadPixels
    _glReadPixels.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
    _glReadPixels.restype = None

    _glGetError = _gl_lib.glGetError
    _glGetError.argtypes = []
    _glGetError.restype = ctypes.c_uint
except AttributeError:
    pass


def check_gl_error():
    if _gl_lib:
        err = _gl_lib.glGetError()
        if err != 0:
            raise RuntimeError(f"OpenGL error: {err}")


def read_fbo_pixels_float32(x, y, width, height):
    """
    Reads pixels from the currently bound framebuffer into a numpy array as float32.
    It reads GL_RGBA format and GL_FLOAT type.
    """
    import numpy as np

    # 4 channels: R, G, B, A
    buffer_size = width * height * 4
    buffer = (ctypes.c_float * buffer_size)()

    if _gl_lib:
        _gl_lib.glReadPixels(x, y, width, height, GL_RGBA, GL_FLOAT, ctypes.byref(buffer))
        check_gl_error()
    else:
        raise RuntimeError("OpenGL library not loaded")

    # Convert to numpy array and reshape
    arr = np.frombuffer(buffer, dtype=np.float32)
    # The image is usually upside down in OpenGL compared to typical image coordinates,
    # but for pure data we might just reshape it.
    arr = arr.reshape((height, width, 4))

    return arr
