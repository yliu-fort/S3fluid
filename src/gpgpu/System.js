export class GPGPUSystem {
  constructor(canvas) {
    // Phase 1 Development Principle: Explicit Error Visibility. Fail fast.
    const _canvas = canvas || document.createElement('canvas');

    // Request WebGL2 Context
    this.gl = _canvas.getContext('webgl2', {
      antialias: false,
      depth: false,
      alpha: false,
      preserveDrawingBuffer: true // useful for debugging occasionally, though we mostly use gl.readPixels on FBOs
    });

    if (!this.gl) {
      throw new Error('FATAL: WebGL 2.0 is not supported on this device/browser.');
    }

    // Must have float texture support for GPGPU.
    // In WebGL2, OES_texture_float is core, but for rendering TO a float texture (MRT/FBO),
    // we explicitly need EXT_color_buffer_float.
    this.extColorBufferFloat = this.gl.getExtension('EXT_color_buffer_float');
    if (!this.extColorBufferFloat) {
      throw new Error('FATAL: EXT_color_buffer_float extension is not supported. Cannot render to float textures.');
    }

    // Also get float linear filtering just in case it's needed for sampling
    this.extTextureFloatLinear = this.gl.getExtension('OES_texture_float_linear');
    if (!this.extTextureFloatLinear) {
      console.warn('OES_texture_float_linear is not supported. Float texture linear interpolation will fallback to nearest.');
    }

    // Default states for GPGPU
    this.gl.disable(this.gl.DEPTH_TEST);
    this.gl.disable(this.gl.BLEND);

    // Create a full-screen quad for executing fragment shaders
    this.quadVAO = this.gl.createVertexArray();
    this.gl.bindVertexArray(this.quadVAO);

    const positionBuffer = this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
    // Two triangles to cover [-1, 1] NDC
    const positions = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
      -1,  1,
       1, -1,
       1,  1
    ]);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

    // Position is location 0
    this.gl.enableVertexAttribArray(0);
    this.gl.vertexAttribPointer(0, 2, this.gl.FLOAT, false, 0, 0);

    this.gl.bindVertexArray(null);
  }

  /**
   * Run a shader pass over the full screen quad.
   * Assume program is already bound, FBO is bound, and uniforms are set.
   */
  runPass() {
    this.gl.bindVertexArray(this.quadVAO);
    this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
    this.gl.bindVertexArray(null);
  }
}
