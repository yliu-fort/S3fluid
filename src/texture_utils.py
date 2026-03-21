import numpy as np
from kivy.graphics.texture import Texture
from .gl_utils import gl_lib
from .fbo_utils import GL_FLOAT, GL_RGBA32F, GL_RGBA, GL_TEXTURE_2D

def ndarray_to_texture(data):
    """
    Converts a NumPy array into a 32-bit Kivy Float Texture.
    If the data is 2D (height, width), it will be duplicated across RGB and A=1.
    If the data is 3D (height, width, channels), it will be padded to RGBA.
    """
    if len(data.shape) == 2:
        height, width = data.shape
        rgba_data = np.zeros((height, width, 4), dtype=np.float32)
        rgba_data[..., 0] = data
        rgba_data[..., 1] = data
        rgba_data[..., 2] = data
        rgba_data[..., 3] = 1.0
    elif len(data.shape) == 3:
        height, width, channels = data.shape
        rgba_data = np.zeros((height, width, 4), dtype=np.float32)
        rgba_data[..., :channels] = data
        if channels < 4:
            rgba_data[..., 3] = 1.0
    else:
        raise ValueError("Data must be 2D or 3D numpy array")

    rgba_data = rgba_data.astype(np.float32)

    tex = Texture.create(size=(width, height), colorfmt='rgba', bufferfmt='float')

    # To bypass Kivy's clamping or internal representation to 0-1 ubytes on some drivers,
    # we explicitly use PyOpenGL/ctypes to upload the float32 array as GL_RGBA32F
    if gl_lib:
        tex.bind()
        # Upload data to the texture
        import ctypes
        gl_lib.glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA32F,
            width,
            height,
            0,
            GL_RGBA,
            GL_FLOAT,
            rgba_data.ctypes.data_as(ctypes.c_void_p)
        )
    else:
        # Fallback to standard kivy blit_buffer (which might clamp precision if not backed by proper GL context defaults)
        tex.blit_buffer(rgba_data.tobytes(), colorfmt='rgba', bufferfmt='float')

    # Nearest neighbor filtering is critical for data textures to prevent interpolation
    tex.mag_filter = 'nearest'
    tex.min_filter = 'nearest'

    return tex
