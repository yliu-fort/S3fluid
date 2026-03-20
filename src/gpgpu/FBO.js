export class GPGPUFBO {
  /**
   * Creates a Framebuffer Object with an attached 32-bit floating point texture (RGBA32F).
   */
  constructor(gl, width, height, initialData = null) {
    this.gl = gl;
    this.width = width;
    this.height = height;

    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.texture);

    // GPGPU requires exact pixel lookups, but we might want linear sampling later
    // depending on the physical field mapping. For compute, nearest is safer.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // Provide initial data if any
    let dataPtr = null;
    if (initialData) {
      if (!(initialData instanceof Float32Array)) {
        throw new Error("GPGPUFBO initialData must be a Float32Array");
      }
      if (initialData.length !== width * height * 4) {
        throw new Error(`GPGPUFBO initialData length ${initialData.length} does not match expected ${width * height * 4}`);
      }
      dataPtr = initialData;
    }

    // Allocate RGBA32F texture
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA32F,
      width,
      height,
      0,
      gl.RGBA,
      gl.FLOAT,
      dataPtr
    );

    this.framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      this.texture,
      0
    );

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`FATAL: Framebuffer is incomplete. Status: ${status}`);
    }

    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  bind() {
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.framebuffer);
    this.gl.viewport(0, 0, this.width, this.height);
  }

  unbind() {
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
  }
}

export class PingPongFBO {
  /**
   * Manages two FBOs for iterative passes (e.g. RK4 steps, diffusion, etc).
   */
  constructor(gl, width, height, initialData = null) {
    this.gl = gl;
    this.width = width;
    this.height = height;

    this.readFBO = new GPGPUFBO(gl, width, height, initialData);
    this.writeFBO = new GPGPUFBO(gl, width, height);
  }

  /**
   * Swaps the read and write FBOs.
   */
  swap() {
    const temp = this.readFBO;
    this.readFBO = this.writeFBO;
    this.writeFBO = temp;
  }

  get read() {
    return this.readFBO;
  }

  get write() {
    return this.writeFBO;
  }
}
