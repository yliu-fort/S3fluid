export class GPGPU {
  constructor(glContext = null) {
    if (glContext) {
      this.gl = glContext;
    } else {
      const canvas = document.createElement('canvas');
      // Request WebGL 2.0
      this.gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
      if (!this.gl) {
        throw new Error("WebGL 2.0 is not supported");
      }
    }

    this.extFloat = this.gl.getExtension('EXT_color_buffer_float');
    if (!this.extFloat) {
      throw new Error("EXT_color_buffer_float is not supported");
    }

    this.gl.disable(this.gl.DEPTH_TEST);
    this.gl.disable(this.gl.BLEND);
  }

  createShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`Shader compilation error: ${info}`);
    }
    return shader;
  }

  createProgram(vertSource, fragSource) {
    const gl = this.gl;
    const program = gl.createProgram();
    const vertShader = this.createShader(gl.VERTEX_SHADER, vertSource);
    const fragShader = this.createShader(gl.FRAGMENT_SHADER, fragSource);

    gl.attachShader(program, vertShader);
    gl.attachShader(program, fragShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new Error(`Program linking error: ${info}`);
    }

    gl.deleteShader(vertShader);
    gl.deleteShader(fragShader);

    return program;
  }

  createTexture(width, height, data = null) {
    const gl = this.gl;
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);

    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    // Using RGBA32F for general computation
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA32F,
      width,
      height,
      0,
      gl.RGBA,
      gl.FLOAT,
      data
    );

    return texture;
  }

  createFBO(texture) {
    const gl = this.gl;
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      texture,
      0
    );

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`Framebuffer incomplete: ${status}`);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return fbo;
  }

  readPixels(fbo, width, height) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    const data = new Float32Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, data);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return data;
  }
}

export class ShaderPass {
  constructor(gpgpu, fragSource) {
    this.gpgpu = gpgpu;
    this.gl = gpgpu.gl;

    const vertSource = `#version 300 es
      in vec2 a_position;
      out vec2 v_uv;
      void main() {
        v_uv = a_position * 0.5 + 0.5;
        gl_Position = vec4(a_position, 0.0, 1.0);
      }
    `;

    this.program = this.gpgpu.createProgram(vertSource, fragSource);

    // Create a full-screen quad
    this.quadBuffer = this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
    const quad = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
      -1,  1,
       1, -1,
       1,  1
    ]);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, quad, this.gl.STATIC_DRAW);

    this.positionLocation = this.gl.getAttribLocation(this.program, 'a_position');
  }

  render(targetFBO, width, height, uniforms = {}, textures = {}) {
    const gl = this.gl;
    gl.useProgram(this.program);

    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFBO);
    gl.viewport(0, 0, width, height);

    // Set uniforms
    for (const [name, value] of Object.entries(uniforms)) {
      const location = gl.getUniformLocation(this.program, name);
      if (location !== null) {
        if (typeof value === 'number') {
          gl.uniform1f(location, value);
        } else if (Array.isArray(value) && value.length === 2) {
          gl.uniform2fv(location, value);
        }
      }
    }

    // Bind textures
    let textureUnit = 0;
    for (const [name, texture] of Object.entries(textures)) {
      const location = gl.getUniformLocation(this.program, name);
      if (location !== null) {
        gl.activeTexture(gl.TEXTURE0 + textureUnit);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.uniform1i(location, textureUnit);
        textureUnit++;
      }
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.enableVertexAttribArray(this.positionLocation);
    gl.vertexAttribPointer(this.positionLocation, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.disableVertexAttribArray(this.positionLocation);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
}

export class PingPongFBO {
  constructor(gpgpu, width, height, initialData = null) {
    this.width = width;
    this.height = height;

    this.tex1 = gpgpu.createTexture(width, height, initialData);
    this.fbo1 = gpgpu.createFBO(this.tex1);

    this.tex2 = gpgpu.createTexture(width, height, initialData);
    this.fbo2 = gpgpu.createFBO(this.tex2);

    this.readIdx = 1;
  }

  get read() {
    return this.readIdx === 1 ? this.tex1 : this.tex2;
  }

  get readFBO() {
    return this.readIdx === 1 ? this.fbo1 : this.fbo2;
  }

  get write() {
    return this.readIdx === 1 ? this.tex2 : this.tex1;
  }

  get writeFBO() {
    return this.readIdx === 1 ? this.fbo2 : this.fbo1;
  }

  swap() {
    this.readIdx = this.readIdx === 1 ? 2 : 1;
  }
}
