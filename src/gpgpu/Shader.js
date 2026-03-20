// A minimal vertex shader for a full-screen quad pass
export const DEFAULT_VS = `#version 300 es
layout(location = 0) in vec2 position;
out vec2 vUv;
void main() {
    vUv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
`;

export class GPGPUShader {
  /**
   * Compiles and links a WebGL2 Shader Program.
   * Enforces strict shader error checking (Explicit Error Visibility Principle).
   */
  constructor(gl, fragmentShaderSource, vertexShaderSource = DEFAULT_VS) {
    this.gl = gl;
    this.program = gl.createProgram();

    // Explicitly prefix fragment shader with version 300 es if not present
    let fsSrc = fragmentShaderSource.trim();
    if (!fsSrc.startsWith('#version 300 es')) {
      fsSrc = '#version 300 es\nprecision highp float;\n' + fsSrc;
    }

    const vs = this._compileShader(gl.VERTEX_SHADER, vertexShaderSource);
    const fs = this._compileShader(gl.FRAGMENT_SHADER, fsSrc);

    gl.attachShader(this.program, vs);
    gl.attachShader(this.program, fs);
    gl.linkProgram(this.program);

    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(this.program);
      gl.deleteProgram(this.program);
      throw new Error('FATAL: WebGL Program Link Error: \n' + info);
    }

    // After linking, we can delete the attached shaders to save memory
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    // Cache uniform locations for performance
    this.uniforms = {};
    const numUniforms = gl.getProgramParameter(this.program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < numUniforms; i++) {
      const activeInfo = gl.getActiveUniform(this.program, i);
      if (activeInfo) {
        // Handle array names like 'uArray[0]'
        const name = activeInfo.name.replace('\\[0\\]', '');
        this.uniforms[name] = gl.getUniformLocation(this.program, activeInfo.name);
      }
    }
  }

  _compileShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      const typeStr = type === gl.VERTEX_SHADER ? 'Vertex Shader' : 'Fragment Shader';
      throw new Error(`FATAL: WebGL ${typeStr} Compilation Error:\n${info}\n\nSource code:\n${this._formatSource(source)}`);
    }

    return shader;
  }

  _formatSource(source) {
    const lines = source.split('\n');
    return lines.map((line, i) => `${String(i + 1).padStart(4, '0')}: ${line}`).join('\n');
  }

  use() {
    this.gl.useProgram(this.program);
  }

  setUniform1i(name, value) {
    if (this.uniforms[name] !== undefined) {
      this.gl.uniform1i(this.uniforms[name], value);
    }
  }

  setUniform1f(name, value) {
    if (this.uniforms[name] !== undefined) {
      this.gl.uniform1f(this.uniforms[name], value);
    }
  }

  setUniform2f(name, x, y) {
    if (this.uniforms[name] !== undefined) {
      this.gl.uniform2f(this.uniforms[name], x, y);
    }
  }
}
