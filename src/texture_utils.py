import numpy as np
import kivy
from kivy.graphics.texture import Texture

kivy.require('2.3.0')

def create_float_texture(width, height, data=None):
    """
    Creates a Kivy GL_RGBA32F texture and optionally fills it with numpy data.
    The `data` should be a numpy float32 array of shape (height, width, 4).
    """
    texture = Texture.create(size=(width, height), colorfmt='rgba', bufferfmt='float')
    texture.mag_filter = 'nearest'
    texture.min_filter = 'nearest'

    if data is not None:
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        if data.shape != (height, width, 4):
            raise ValueError(f"Expected data shape ({height}, {width}, 4), got {data.shape}")

        # blit_buffer expects a flat bytes-like object
        # Note: Kivy blit_buffer may be tricky with formats depending on GL backend
        # For floating point data, we strictly ensure we pass float32 bytes
        texture.blit_buffer(data.tobytes(), colorfmt='rgba', bufferfmt='float')

    return texture
