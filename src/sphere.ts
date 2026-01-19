/**
 * sphere.ts - Interactive Sphere Renderer
 *
 * This module renders a draggable sphere that interacts with the water simulation.
 * The sphere can be moved by the user and optionally affected by physics (gravity
 * and buoyancy).
 *
 * Key features:
 * - Procedurally generated sphere geometry using octahedron subdivision
 * - Caustic lighting effects when underwater
 * - Ambient occlusion based on proximity to pool walls
 * - Underwater color tinting
 */

/**
 * Renders an interactive sphere with realistic underwater lighting.
 *
 * The Sphere class creates a subdivided octahedron geometry for smooth rendering
 * and applies dynamic lighting based on water state. The shader handles:
 * - Refracted sunlight through water surface
 * - Caustic patterns when underwater
 * - Distance-based darkening near pool edges
 * - Underwater color tinting
 */
export class Sphere {
  /** WebGPU device for creating GPU resources */
  private device: GPUDevice;

  /** Texture format matching the canvas (e.g., 'bgra8unorm') */
  private format: GPUTextureFormat;

  /** Uniform buffer containing view-projection matrix and eye position */
  private commonUniformBuffer: GPUBuffer;

  /** Uniform buffer containing sphere position and radius */
  private sphereUniformBuffer: GPUBuffer;

  /** Uniform buffer containing light direction vector */
  private lightUniformBuffer: GPUBuffer;

  /** Vertex buffer containing sphere vertex positions (unit sphere) */
  private positionBuffer!: GPUBuffer;

  /** Index buffer for indexed drawing */
  private indexBuffer!: GPUBuffer;

  /** Number of indices to draw */
  private vertexCount!: number;

  /** The render pipeline for sphere rendering */
  private pipeline!: GPURenderPipeline;

  /**
   * Creates a new Sphere renderer.
   *
   * @param device - WebGPU device for resource creation
   * @param format - Canvas texture format
   * @param uniformBuffer - Buffer with view-projection matrix and eye position
   * @param lightUniformBuffer - Buffer with light direction
   * @param sphereUniformBuffer - Buffer for sphere position and radius
   */
  constructor(
    device: GPUDevice,
    format: GPUTextureFormat,
    uniformBuffer: GPUBuffer,
    lightUniformBuffer: GPUBuffer,
    sphereUniformBuffer: GPUBuffer
  ) {
    this.device = device;
    this.format = format;
    this.commonUniformBuffer = uniformBuffer;
    this.sphereUniformBuffer = sphereUniformBuffer;
    this.lightUniformBuffer = lightUniformBuffer;

    this.createGeometry();
    this.createPipeline();
  }

  /**
   * Updates the sphere's position and size in the uniform buffer.
   *
   * Called each frame when the sphere moves (via physics or user interaction).
   *
   * @param center - The sphere center position [x, y, z]
   * @param radius - The sphere radius
   */
  update(center: number[], radius: number): void {
    const data = new Float32Array([...center, radius]);
    this.device.queue.writeBuffer(this.sphereUniformBuffer, 0, data);
  }

  /**
   * Creates the sphere geometry using octahedron subdivision.
   *
   * This technique produces a more uniform triangle distribution than
   * latitude/longitude sphere generation. The algorithm:
   * 1. Starts with 8 octants of a unit cube
   * 2. Subdivides each octant into triangles
   * 3. Projects vertices onto a unit sphere
   *
   * The `detail` parameter controls the subdivision level:
   * - detail=1: 8 triangles (octahedron)
   * - detail=10: 800 triangles (smooth sphere)
   */
  private createGeometry(): void {
    const detail = 10; // Subdivision level for smooth sphere

    /**
     * Helper class to deduplicate vertices.
     * Vertices at octant boundaries would be duplicated without this.
     */
    class Indexer {
      /** Array of unique vertex positions */
      unique: number[][];
      /** Map from position string to index */
      map: Map<string, number>;

      constructor() {
        this.unique = [];
        this.map = new Map();
      }

      /**
       * Adds a vertex, returning its index.
       * If the vertex already exists, returns the existing index.
       */
      add(v: number[]): number {
        const key = v.join(',');
        if (!this.map.has(key)) {
          this.map.set(key, this.unique.length);
          this.unique.push(v);
        }
        return this.map.get(key)!;
      }
    }

    /**
     * Returns the sign multipliers for an octant (0-7).
     * Each bit controls the sign of one axis:
     * - Bit 0: X sign
     * - Bit 1: Y sign
     * - Bit 2: Z sign
     */
    function pickOctant(i: number): [number, number, number] {
      return [
        (i & 1) * 2 - 1,
        (i & 2) - 1,
        (i & 4) / 2 - 1
      ];
    }

    /**
     * Applies a smoothing function to make triangles more uniform.
     * Without this, triangles near octant corners would be smaller.
     */
    function fix(x: number): number { return x + (x - x * x) / 2; }

    const indexer = new Indexer();
    const finalIndices: number[] = [];

    // Process each of the 8 octants
    for (let octant = 0; octant < 8; octant++) {
      const scale = pickOctant(octant);

      // Determine triangle winding based on octant orientation
      const flip = scale[0] * scale[1] * scale[2] > 0;
      const data: number[] = [];

      // Generate vertices for this octant using barycentric subdivision
      for (let i = 0; i <= detail; i++) {
        for (let j = 0; i + j <= detail; j++) {
          // Barycentric coordinates (a, b, c) where a + b + c = 1
          const a = i / detail;
          const b = j / detail;
          const c = (detail - i - j) / detail;

          // Apply smoothing and create position
          const v = [fix(a), fix(b), fix(c)];

          // Normalize to project onto unit sphere
          const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
          const pos = [
            (v[0] / len) * scale[0],
            (v[1] / len) * scale[1],
            (v[2] / len) * scale[2]
          ];

          data.push(indexer.add(pos));
        }
      }

      // Generate triangle indices for this octant
      for (let i = 0; i <= detail; i++) {
        if (i > 0) {
          for (let j = 0; i + j <= detail; j++) {
            // Calculate vertex indices in the triangular grid
            const a = (i - 1) * (detail + 1) + ((i - 1) - (i - 1) * (i - 1)) / 2 + j;
            const b = i * (detail + 1) + (i - i * i) / 2 + j;

            // Add triangle with correct winding order
            if (flip) {
              finalIndices.push(data[a], data[b], data[a + 1]);
            } else {
              finalIndices.push(data[a], data[a + 1], data[b]);
            }

            // Add second triangle for quad (except at octant edge)
            if (i + j < detail) {
              if (flip) {
                finalIndices.push(data[b], data[b + 1], data[a + 1]);
              } else {
                finalIndices.push(data[b], data[a + 1], data[b + 1]);
              }
            }
          }
        }
      }
    }

    this.vertexCount = finalIndices.length;

    // Flatten vertex positions into a single array
    const finalPositions: number[] = [];
    for (const p of indexer.unique) {
      finalPositions.push(...p);
    }

    // Create and populate vertex buffer
    this.positionBuffer = this.device.createBuffer({
      label: 'Sphere Vertex Buffer',
      size: finalPositions.length * 4,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.positionBuffer.getMappedRange()).set(finalPositions);
    this.positionBuffer.unmap();

    // Create and populate index buffer
    this.indexBuffer = this.device.createBuffer({
      label: 'Sphere Index Buffer',
      size: finalIndices.length * 4,
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true,
    });
    new Uint32Array(this.indexBuffer.getMappedRange()).set(finalIndices);
    this.indexBuffer.unmap();
  }

  /**
   * Creates the render pipeline with WGSL shaders.
   *
   * The shader implements:
   * - Unit sphere scaled and translated by uniform values
   * - Distance-based darkening near pool walls/floor
   * - Refracted light and caustic sampling when underwater
   * - Underwater color tinting
   */
  private createPipeline(): void {
    const shaderModule = this.device.createShaderModule({
      label: 'Sphere Shader',
      code: `
        // Camera uniforms: view-projection matrix
        struct CommonUniforms {
          viewProjectionMatrix : mat4x4f,
        }
        @binding(0) @group(0) var<uniform> commonUniforms : CommonUniforms;

        // Sphere transform: center position and radius
        struct SphereUniforms {
          center : vec3f,
          radius : f32,
        }
        @binding(1) @group(0) var<uniform> sphereUniforms : SphereUniforms;

        // Light direction for shading calculations
        struct LightUniforms {
           direction : vec3f,
        }
        @binding(2) @group(0) var<uniform> light : LightUniforms;

        // Textures for water state and caustics
        @binding(3) @group(0) var waterSampler : sampler;
        @binding(4) @group(0) var waterTexture : texture_2d<f32>;
        @binding(5) @group(0) var causticTexture : texture_2d<f32>;

        struct VertexOutput {
          @builtin(position) position : vec4f,
          @location(0) localPos : vec3f,  // Position on unit sphere
          @location(1) worldPos : vec3f,  // Position in world space
        }

        @vertex
        fn vs_main(@location(0) position : vec3f) -> VertexOutput {
          var output : VertexOutput;

          // Transform unit sphere vertex to world space
          let worldPos = sphereUniforms.center + position * sphereUniforms.radius;
          output.position = commonUniforms.viewProjectionMatrix * vec4f(worldPos, 1.0);
          output.localPos = position;
          output.worldPos = worldPos;
          return output;
        }

        @fragment
        fn fs_main(@location(0) localPos : vec3f, @location(1) worldPos : vec3f) -> @location(0) vec4f {
          // Physical constants for light refraction
          let IOR_AIR = 1.0;
          let IOR_WATER = 1.333;

          // Base sphere color (gray)
          var color = vec3f(0.5);

          let sphereRadius = sphereUniforms.radius;
          let point = worldPos;

          // Distance-based darkening near pool boundaries
          // Creates ambient occlusion effect near walls and floor
          let dist_x = (1.0 + sphereRadius - abs(point.x)) / sphereRadius;
          let dist_z = (1.0 + sphereRadius - abs(point.z)) / sphereRadius;
          let dist_y = (point.y + 1.0 + sphereRadius) / sphereRadius;

          // Apply inverse-cube falloff for soft shadows
          color *= 1.0 - 0.9 / pow(max(0.1, dist_x), 3.0);
          color *= 1.0 - 0.9 / pow(max(0.1, dist_z), 3.0);
          color *= 1.0 - 0.9 / pow(max(0.1, dist_y), 3.0);

          // Calculate refracted light direction (Snell's law)
          let refractedLight = refract(-light.direction, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
          let sphereNormal = normalize(localPos);

          // Basic diffuse lighting
          var diffuse = max(0.0, dot(-refractedLight, sphereNormal)) * 0.5;

          // Sample water height at sphere's XZ position
          let waterInfo = textureSampleLevel(waterTexture, waterSampler, point.xz * 0.5 + 0.5, 0.0);

          // Apply caustics when underwater
          if (point.y < waterInfo.r) {
             // Project caustic UV based on refracted light direction
             let causticUV = 0.75 * (point.xz - point.y * refractedLight.xz / refractedLight.y) * 0.5 + 0.5;
             let caustic = textureSampleLevel(causticTexture, waterSampler, causticUV, 0.0);
             diffuse *= caustic.r * 4.0; // Amplify caustic brightness
          }

          color += diffuse;

          // Apply underwater color tint
          if (point.y < waterInfo.r) {
             let underwaterColor = vec3f(0.4, 0.9, 1.0);
             color *= underwaterColor * 1.2;
          }

          return vec4f(color, 1.0);
        }
      `
    });

    // Create the render pipeline
    this.pipeline = this.device.createRenderPipeline({
      label: 'Sphere Pipeline',
      layout: 'auto',
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 3 * 4, // 3 floats per vertex (x, y, z)
          attributes: [{
            shaderLocation: 0,
            offset: 0,
            format: 'float32x3'
          }]
        }]
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{ format: this.format }]
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'back', // Back-face culling (sphere is closed)
      },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth24plus',
      }
    });
  }

  /**
   * Renders the sphere to the current render pass.
   *
   * Creates a new bind group each frame to incorporate dynamic textures
   * (water height and caustics that change every frame).
   *
   * @param passEncoder - The active render pass encoder
   * @param waterTexture - Current water simulation texture (height/normals)
   * @param waterSampler - Sampler for water texture
   * @param causticsTexture - Pre-computed caustic pattern texture
   */
  render(passEncoder: GPURenderPassEncoder, waterTexture: GPUTexture, waterSampler: GPUSampler, causticsTexture: GPUTexture): void {
    // Create bind group with all required resources
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.commonUniformBuffer } },
        { binding: 1, resource: { buffer: this.sphereUniformBuffer } },
        { binding: 2, resource: { buffer: this.lightUniformBuffer } },
        { binding: 3, resource: waterSampler },
        { binding: 4, resource: waterTexture.createView() },
        { binding: 5, resource: causticsTexture.createView() }
      ]
    });

    // Issue draw commands
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setVertexBuffer(0, this.positionBuffer);
    passEncoder.setIndexBuffer(this.indexBuffer, 'uint32');
    passEncoder.drawIndexed(this.vertexCount);
  }
}
