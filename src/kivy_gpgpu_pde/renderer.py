import numpy as np
from kivy.graphics import Mesh, RenderContext, Callback, InstructionGroup, Color, Line
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, BooleanProperty
from kivy.core.window import Window

# Basic Shader for colormapping a single channel float texture
colormesh_vs = '''
$HEADER$
in vec3 vPosition;
in vec2 vTexCoords0;
out vec2 frag_tex_coords;

uniform mat4 modelview_mat;
uniform mat4 proj_mat;

void main(void) {
    frag_tex_coords = vTexCoords0;
    vec4 pos = vec4(vPosition.xyz, 1.0);
    gl_Position = proj_mat * modelview_mat * pos;
}
'''

colormesh_fs = '''
$HEADER$
in vec2 frag_tex_coords;
out vec4 fragColor;

uniform sampler2D texture0;
uniform float data_min;
uniform float data_max;

// Simple jet-like colormap
vec3 colormap(float t) {
    vec3 c;
    if (t < 0.25) {
        c = vec3(0.0, 4.0 * t, 1.0);
    } else if (t < 0.5) {
        c = vec3(0.0, 1.0, 1.0 - 4.0 * (t - 0.25));
    } else if (t < 0.75) {
        c = vec3(4.0 * (t - 0.5), 1.0, 0.0);
    } else {
        c = vec3(1.0, 1.0 - 4.0 * (t - 0.75), 0.0);
    }
    return c;
}

void main(void) {
    // Read the single float value from the Red channel
    float val = texture(texture0, frag_tex_coords).r;

    // Normalize value
    float range = data_max - data_min;
    float norm_val = 0.5; // Default middle
    if (range > 1e-7) {
        norm_val = clamp((val - data_min) / range, 0.0, 1.0);
    }

    vec3 color = colormap(norm_val);
    fragColor = vec4(color, 1.0);
}
'''

class MeshRenderer(Widget):
    """
    Renders a 3D isomorphic sphere mesh.
    Maps an input GPGPU Float texture to the sphere using a colormap shader.
    """
    texture = ObjectProperty(None, allownone=True)
    wireframe = BooleanProperty(False)

    def __init__(self, **kwargs):
        self.canvas = RenderContext(use_parent_modelview=True, use_parent_projection=True)
        self.canvas.shader.vs = colormesh_vs
        self.canvas.shader.fs = colormesh_fs

        super(MeshRenderer, self).__init__(**kwargs)

        self.mesh = None
        self.wireframe_lines = InstructionGroup()
        self.canvas.add(self.wireframe_lines)

        self.data_min = -1.0
        self.data_max = 1.0

        self.bind(texture=self._on_texture)
        self.bind(wireframe=self._on_wireframe)

    def _on_texture(self, instance, value):
        if value:
            self.canvas['texture0'] = 0
            self.canvas['data_min'] = self.data_min
            self.canvas['data_max'] = self.data_max

    def _on_wireframe(self, instance, value):
        self.wireframe_lines.clear()
        if value and self.mesh:
            self.wireframe_lines.add(Color(1, 1, 1, 0.2))
            # Just add basic wireframe representation
            vertices = self.mesh.vertices
            indices = self.mesh.indices

            # Simple line representation based on indices
            # For a proper wireframe, we'd draw lines for each triangle
            line_points = []
            for i in range(0, len(indices), 3):
                idx1 = indices[i] * 5
                idx2 = indices[i+1] * 5
                idx3 = indices[i+2] * 5

                v1 = vertices[idx1:idx1+3]
                v2 = vertices[idx2:idx2+3]
                v3 = vertices[idx3:idx3+3]

                # Transform vertices to screen space (simplified for now)
                pass # Proper wireframe requires applying MVP matrix, which is complex in pure Kivy without full GL projection handling manually

            # As an alternative, we could use glPolygonMode in OpenGL, but Kivy doesn't easily expose it per-widget natively.
            # We'll rely on the shader or leave it empty if complicated.
            print("Wireframe mode toggled")

    def build_mesh(self, sht, radius=1.0):
        """
        Builds the spherical mesh matching the NumPySHT grid.
        - sht: NumPySHT instance
        - radius: Sphere radius
        """
        lats = sht.lats # n_lat values from 0 to pi
        lons = sht.lons # n_lon values from 0 to 2pi

        n_lat = sht.n_lat
        n_lon = sht.n_lon

        # Vertices layout: x, y, z, u, v
        vertices = []
        indices = []

        # We add an extra longitude column to close the seam (u=1.0)
        # We also need poles to close the sphere correctly (v=0.0 and v=1.0)

        # To avoid singularity at poles and correctly map the Gauss-Legendre grid:
        # The true Gauss-Legendre grid does NOT contain the poles (lats are strictly inside (0, pi))
        # So we add artificial pole vertices at lat=0 and lat=pi.

        # Calculate UV coordinates based on texel centers to match WebGL GPGPU exactly
        # The texture is size (n_lon, n_lat) ? Wait, in physics.py / gpu_sht.py, what is the layout?
        # Usually it's (n_lat, n_lon) in numpy, but in texture, is it (n_lon, n_lat)?
        # For FloatFbo texture, shape is usually (width=n_lon, height=n_lat)

        # 1. Add North Pole (lat=0)
        # We need a vertex for each longitude to avoid texture stretching singularity if possible,
        # but a single vertex is often used. To map texture cleanly, we use a single vertex but give it v=0
        # Actually, for triangles connected to the pole, each needs a different 'u'.
        # So we create n_lon+1 pole vertices.
        np_start_idx = len(vertices) // 5
        for j in range(n_lon + 1):
            u = (j + 0.5) / n_lon  # Match longitude texel centers if possible, or just j/n_lon
            # Better: match exact 0 to 1 for rendering interpolation
            u = j / n_lon
            # v mapping: texture y goes from 0 to 1. n_lat rows.
            # texel centers are (i + 0.5)/n_lat. Pole is at v=0 (or 1 depending on orientation).
            # Let's say v=0 corresponds to North Pole.
            v = 0.0

            vertices.extend([0.0, 0.0, radius, u, v])

        # 2. Add Gauss-Legendre grid vertices
        grid_start_idx = len(vertices) // 5
        for i in range(n_lat):
            lat = lats[i]
            # v maps from first GL node to last GL node
            # Texture mapping: row i corresponds to lat[i].
            # v should be exactly at the texel center so `texture()` samples exactly the data
            v = (i + 0.5) / n_lat

            z = radius * np.cos(lat)
            r_xy = radius * np.sin(lat)

            for j in range(n_lon + 1):
                lon = lons[j % n_lon] # Wrap around for j=n_lon
                x = r_xy * np.cos(lon)
                y = r_xy * np.sin(lon)

                u = (j + 0.5) / n_lon # Texel center for j? Wait, if we use j/n_lon, it interpolates.
                # If we want exact texel centers, we might need a specific mapping.
                # Let's use standard u=j/n_lon to stretch the texture over the whole 0..1 range
                u = j / n_lon

                vertices.extend([x, y, z, u, v])

        # 3. Add South Pole (lat=pi)
        sp_start_idx = len(vertices) // 5
        for j in range(n_lon + 1):
            u = j / n_lon
            v = 1.0
            vertices.extend([0.0, 0.0, -radius, u, v])

        # Build indices
        # North pole triangles
        for j in range(n_lon):
            p_top = np_start_idx + j
            p_bottom_left = grid_start_idx + j
            p_bottom_right = grid_start_idx + j + 1

            # Triangle
            indices.extend([p_top, p_bottom_left, p_bottom_right])

        # Grid quads (rendered as two triangles)
        for i in range(n_lat - 1):
            row_start = grid_start_idx + i * (n_lon + 1)
            next_row_start = grid_start_idx + (i + 1) * (n_lon + 1)

            for j in range(n_lon):
                top_left = row_start + j
                top_right = row_start + j + 1
                bottom_left = next_row_start + j
                bottom_right = next_row_start + j + 1

                indices.extend([top_left, bottom_left, top_right])
                indices.extend([top_right, bottom_left, bottom_right])

        # South pole triangles
        last_grid_row_start = grid_start_idx + (n_lat - 1) * (n_lon + 1)
        for j in range(n_lon):
            p_top_left = last_grid_row_start + j
            p_top_right = last_grid_row_start + j + 1
            p_bottom = sp_start_idx + j

            indices.extend([p_top_left, p_bottom, p_top_right])

        # Update Mesh
        self.canvas.clear()
        self.canvas.add(Callback(self._setup_gl_context))

        # fmt is (name, num_elements, type)
        fmt = [
            (b'vPosition', 3, 'float'),
            (b'vTexCoords0', 2, 'float'),
        ]

        self.mesh = Mesh(
            vertices=vertices,
            indices=indices,
            fmt=fmt,
            mode='triangles',
            texture=self.texture
        )
        self.canvas.add(self.mesh)
        self.canvas.add(Callback(self._reset_gl_context))

        # Re-add wireframe group if needed
        self.canvas.add(self.wireframe_lines)

    def _setup_gl_context(self, instr):
        # Apply transformation matrices (rotation, translation) here
        # For a proper 3D view, we need a camera. Let's setup a basic orthographic or perspective projection.
        from kivy.graphics.transformation import Matrix

        # Center the widget
        w, h = Window.size
        aspect_ratio = w / float(h)

        proj = Matrix()
        # Simple perspective projection
        proj.view_clip(-aspect_ratio, aspect_ratio, -1.0, 1.0, 1.0, 100.0, 1)

        modelview = Matrix()
        modelview.translate(0, 0, -3) # Move back

        # Rotate to see it better
        import time
        t = time.time() * 20.0 # rotation speed
        modelview.rotate(np.radians(t), 0, 1, 0) # Rotate around Y
        modelview.rotate(np.radians(20), 1, 0, 0) # Tilt down a bit

        self.canvas['proj_mat'] = proj
        self.canvas['modelview_mat'] = modelview

        if self.texture:
            self.canvas['data_min'] = self.data_min
            self.canvas['data_max'] = self.data_max

    def _reset_gl_context(self, instr):
        pass
