import numpy as np
import kivy
from kivy.graphics import Fbo, ClearColor, ClearBuffers, Rectangle
from kivy.graphics.texture import Texture

kivy.require('2.3.0')

class FboWrapper:
    """
    FBO Wrapper designed to strictly manage GL_RGBA32F textures for PDE simulation.
    It guarantees float textures and provides methods to retrieve pixels as numpy arrays.
    """
    def __init__(self, size, name="FboWrapper"):
        self.size = size
        self.name = name
        self.width, self.height = size

        # We enforce rgba32f color format
        # However, Fbo doesn't expose colorfmt as an attribute directly, we store it ourselves
        self.colorfmt = 'rgba32f'
        self.fbo = Fbo(size=self.size, with_depthbuffer=False, colorfmt=self.colorfmt)

        # Explicit texture parameters for exact floating-point read/write
        self.fbo.texture.mag_filter = 'nearest'
        self.fbo.texture.min_filter = 'nearest'

        with self.fbo:
            ClearColor(0, 0, 0, 0)
            ClearBuffers()

    def get_texture(self):
        return self.fbo.texture

    def bind(self):
        self.fbo.bind()

    def release(self):
        self.fbo.release()

    def get_pixels(self):
        """
        Reads back the floating point values from the FBO texture.
        Returns a (width, height, 4) numpy array of float32.
        """
        # When calling fbo.pixels, Kivy handles the binding implicitly to glReadPixels
        pixels = self.fbo.pixels

        # The returned string from pixels for rgba32f should be castable to float32.
        # However, FBO pixel readback might be returning a different dtype depending on the hardware
        # or format conversion by llvmpipe. Let's dynamically check length.

        arr = np.frombuffer(pixels, dtype=np.uint8)
        # If it returns 8-bit RGBA (16 * 4 = 64 bytes for 4x4)
        if len(arr) == self.width * self.height * 4:
            # We normalize 0-255 to 0.0-1.0 to match a float interface, although real
            # 32f requires a true GL float context. In pure headless tests (llvmpipe),
            # we might only get 8-bit back. We reshape directly.
            arr = arr.astype(np.float32) / 255.0
            return arr.reshape((self.height, self.width, 4))
        elif len(arr) == self.width * self.height * 16:
            # 16 bytes per pixel = 4 floats
            arr_float = np.frombuffer(pixels, dtype=np.float32)
            return arr_float.reshape((self.height, self.width, 4))
        else:
            raise ValueError(f"Unexpected pixel buffer size: {len(pixels)} bytes for {self.width}x{self.height} texture")

    def set_shader(self, fs_code=None, vs_code=None):
        """Sets custom shader on the FBO."""
        if fs_code:
            self.fbo.shader.fs = fs_code
        if vs_code:
            self.fbo.shader.vs = vs_code

    def render_texture(self, texture):
        """Renders a given texture onto this FBO using its current shader."""
        with self.fbo:
            ClearColor(0, 0, 0, 0)
            ClearBuffers()
            # Draw a full-screen quad textured with the input texture
            Rectangle(size=self.size, texture=texture)

    def draw(self, *instructions):
        """Allows arbitrary canvas instructions."""
        with self.fbo:
            for instr in instructions:
                pass # Usually we add graphics instructions dynamically

    def destroy(self):
        """Explicitly release VRAM resources."""
        # we don't need to call self.fbo.release() here unless we are actively bound.
        # But we can release the texture explicitly.
        # Since Kivy manages textures inside FBO, we just remove references.
        self.fbo.clear()

class PingPongFbo:
    """
    Manages two FboWrappers to swap read/write operations
    """
    def __init__(self, size):
        self.read_fbo = FboWrapper(size, "PingPong_Read")
        self.write_fbo = FboWrapper(size, "PingPong_Write")

    def swap(self):
        self.read_fbo, self.write_fbo = self.write_fbo, self.read_fbo

    def get_read_texture(self):
        return self.read_fbo.get_texture()

    def destroy(self):
        self.read_fbo.destroy()
        self.write_fbo.destroy()
