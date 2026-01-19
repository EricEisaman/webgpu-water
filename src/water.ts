/**
 * water.ts - Water Simulation and Rendering System
 *
 * This is the core module that implements the interactive water simulation.
 * It handles both the physics simulation and the visual rendering of water.
 *
 * The simulation uses a heightfield approach where water height is stored
 * in a 2D texture. The simulation runs on the GPU using render passes.
 *
 * Key components:
 * - Physics simulation: Wave propagation using neighboring height values
 * - Drop creation: Adding ripples from user interaction
 * - Sphere interaction: Water displacement from moving sphere
 * - Normal calculation: Computing surface normals for lighting
 * - Surface rendering: Reflections and refractions using ray tracing
 * - Caustics: Light patterns on pool floor from water surface refraction
 */

import type { PipelineConfig } from './types';

/**
 * Main water simulation and rendering class.
 *
 * The Water class manages:
 * 1. Two ping-pong textures for double-buffered simulation
 * 2. Multiple compute pipelines for different simulation steps
 * 3. Surface mesh for rendering the water from above and below
 * 4. Caustics texture for underwater light patterns
 */
export class Water {
  /** WebGPU device for all GPU operations */
  private device: GPUDevice;

  /** Width of the simulation texture in pixels */
  private width: number;

  /** Height of the simulation texture in pixels */
  private height: number;

  // --- External Resources ---
  // These buffers and textures are passed in from main.ts

  /** Common uniform buffer (view-projection matrix, eye position) */
  private commonUniformBuffer: GPUBuffer;

  /** Light direction uniform buffer */
  private lightUniformBuffer: GPUBuffer;

  /** Sphere position and radius uniform buffer */
  private sphereUniformBuffer: GPUBuffer;

  /** Shadow toggle flags uniform buffer */
  private shadowUniformBuffer: GPUBuffer;

  /** Pool tile texture for refracted view */
  private tileTexture: GPUTexture;

  /** Sampler for tile texture */
  private tileSampler: GPUSampler;

  /** Skybox cubemap texture for reflections */
  private skyTexture: GPUTexture;

  /** Sampler for skybox texture */
  private skySampler: GPUSampler;

  // --- Physics State ---
  // Double-buffered textures for ping-pong rendering

  /**
   * Primary simulation texture (current state).
   * RGBA channels store:
   * - R: Water height
   * - G: Water velocity
   * - B: Surface normal X component
   * - A: Surface normal Z component
   */
  textureA: GPUTexture;

  /**
   * Secondary simulation texture (next state).
   * Swapped with textureA after each simulation step.
   */
  textureB: GPUTexture;

  /**
   * Caustics texture storing light intensity patterns.
   * Higher resolution (1024x1024) for visual detail.
   * - R: Light intensity
   * - G: Sphere shadow factor
   */
  causticsTexture: GPUTexture;

  /** Sampler for simulation textures (linear filtering, clamp edges) */
  sampler: GPUSampler;

  // --- Simulation Pipelines ---
  // Each pipeline performs one step of the simulation

  /** Pipeline for adding water drops (ripples) */
  private dropPipeline!: PipelineConfig;

  /** Pipeline for wave propagation physics */
  private updatePipeline!: PipelineConfig;

  /** Pipeline for computing surface normals */
  private normalPipeline!: PipelineConfig;

  /** Pipeline for sphere-water interaction */
  private spherePipeline!: PipelineConfig;

  // --- Surface Rendering ---

  /** Vertex buffer for water surface mesh */
  private positionBuffer!: GPUBuffer;

  /** Index buffer for water surface mesh */
  private indexBuffer!: GPUBuffer;

  /** Number of indices in the surface mesh */
  private vertexCount!: number;

  /** Bind group layout for surface rendering (shared by both pipelines) */
  private surfaceBindGroupLayout!: GPUBindGroupLayout;

  /** Pipeline for rendering water surface from above */
  private surfacePipelineAbove!: GPURenderPipeline;

  /** Pipeline for rendering water surface from below */
  private surfacePipelineUnder!: GPURenderPipeline;

  // --- Caustics ---

  /** Pipeline for rendering caustic light patterns */
  private causticsPipeline!: GPURenderPipeline;

  /**
   * Creates a new Water simulation system.
   *
   * @param device - WebGPU device
   * @param width - Simulation texture width
   * @param height - Simulation texture height
   * @param uniformBuffer - Common uniforms (matrices, eye position)
   * @param lightUniformBuffer - Light direction buffer
   * @param sphereUniformBuffer - Sphere position/radius buffer
   * @param shadowUniformBuffer - Shadow toggle flags buffer
   * @param tileTexture - Pool tile texture
   * @param tileSampler - Tile texture sampler
   * @param skyTexture - Skybox cubemap texture
   * @param skySampler - Skybox sampler
   */
  constructor(
    device: GPUDevice,
    width: number,
    height: number,
    uniformBuffer: GPUBuffer,
    lightUniformBuffer: GPUBuffer,
    sphereUniformBuffer: GPUBuffer,
    shadowUniformBuffer: GPUBuffer,
    tileTexture: GPUTexture,
    tileSampler: GPUSampler,
    skyTexture: GPUTexture,
    skySampler: GPUSampler
  ) {
    this.device = device;
    this.width = width;
    this.height = height;

    // Store external resources
    this.commonUniformBuffer = uniformBuffer;
    this.lightUniformBuffer = lightUniformBuffer;
    this.sphereUniformBuffer = sphereUniformBuffer;
    this.shadowUniformBuffer = shadowUniformBuffer;
    this.tileTexture = tileTexture;
    this.tileSampler = tileSampler;
    this.skyTexture = skyTexture;
    this.skySampler = skySampler;

    // Create double-buffered simulation textures
    this.textureA = this.createTexture();
    this.textureB = this.createTexture();

    // Caustics texture (higher resolution for detail)
    this.causticsTexture = this.device.createTexture({
      size: [1024, 1024],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // Create sampler with linear filtering and edge clamping
    this.sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    // Initialize all pipelines
    this.createPipelines();
    this.createSurfaceMesh();
    this.createSurfacePipeline();
    this.createCausticsPipeline();
  }

  /**
   * Creates a simulation texture with appropriate format.
   *
   * Uses float32 if available (higher precision), otherwise float16.
   * The texture stores height, velocity, and normal data in RGBA channels.
   */
  private createTexture(): GPUTexture {
    const format = this.device.features.has('float32-filterable') ? 'rgba32float' : 'rgba16float';
    return this.device.createTexture({
      size: [this.width, this.height],
      format: format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  /**
   * Creates all simulation pipelines (drop, update, normal, sphere).
   *
   * Each pipeline renders a fullscreen quad that processes every pixel
   * of the simulation texture. The output is written to textureB,
   * then textures are swapped.
   */
  private createPipelines(): void {
    const format: GPUTextureFormat = this.device.features.has('float32-filterable') ? 'rgba32float' : 'rgba16float';

    // Common fullscreen quad vertex shader
    // Generates 6 vertices for 2 triangles covering the screen
    const fullscreenQuadVS = `
      struct VertexOutput {
        @builtin(position) position : vec4f,
        @location(0) uv : vec2f,
      }

      @vertex
      fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
        var pos = array<vec2f, 6>(
          vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
          vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
        );
        var output : VertexOutput;
        output.position = vec4f(pos[vertexIndex], 0.0, 1.0);
        output.uv = vec2f((pos[vertexIndex].x + 1.0) * 0.5, (1.0 - pos[vertexIndex].y) * 0.5);
        return output;
      }
    `;

    // --- Drop Pipeline ---
    // Adds circular ripples to the water at a given position
    // Uses cosine falloff for smooth drop shape
    this.dropPipeline = this.createPipeline('Drop', fullscreenQuadVS, `
      @group(0) @binding(0) var waterTexture : texture_2d<f32>;
      @group(0) @binding(1) var waterSampler : sampler;

      struct DropUniforms {
        center : vec2f,    // Drop position in [-1, 1] range
        radius : f32,      // Drop radius
        strength : f32,    // Drop intensity (positive or negative)
      }
      @group(0) @binding(2) var<uniform> u : DropUniforms;

      @fragment
      fn fs_main(@location(0) uv : vec2f) -> @location(0) vec4f {
        var info = textureSample(waterTexture, waterSampler, uv);

        // Calculate distance from drop center with cosine falloff
        let drop = max(0.0, 1.0 - length(u.center * 0.5 + 0.5 - uv) / u.radius);
        let dropVal = 0.5 - cos(drop * 3.14159265) * 0.5;

        // Add drop height to water surface
        info.r += dropVal * u.strength;

        return info;
      }
    `, 32, format);

    // --- Update Pipeline ---
    // Propagates waves using a simple finite difference scheme
    // Height moves toward neighbor average, velocity carries momentum
    this.updatePipeline = this.createPipeline('Update', fullscreenQuadVS, `
      @group(0) @binding(0) var waterTexture : texture_2d<f32>;
      @group(0) @binding(1) var waterSampler : sampler;

      struct UpdateUniforms {
        delta : vec2f,  // Texel size (1/width, 1/height)
      }
      @group(0) @binding(2) var<uniform> u : UpdateUniforms;

      @fragment
      fn fs_main(@location(0) uv : vec2f) -> @location(0) vec4f {
        var info = textureSample(waterTexture, waterSampler, uv);

        // Sample neighboring heights
        let dx = vec2f(u.delta.x, 0.0);
        let dy = vec2f(0.0, u.delta.y);

        let average = (
          textureSample(waterTexture, waterSampler, uv - dx).r +
          textureSample(waterTexture, waterSampler, uv - dy).r +
          textureSample(waterTexture, waterSampler, uv + dx).r +
          textureSample(waterTexture, waterSampler, uv + dy).r
        ) * 0.25;

        // Update velocity based on difference from average
        info.g += (average - info.r) * 2.0;
        // Apply damping to prevent perpetual waves
        info.g *= 0.995;
        // Update height based on velocity
        info.r += info.g;

        return info;
      }
    `, 16, format);

    // --- Normal Pipeline ---
    // Computes surface normals from height differences
    // Normals are stored in BA channels for lighting calculations
    this.normalPipeline = this.createPipeline('Normal', fullscreenQuadVS, `
      @group(0) @binding(0) var waterTexture : texture_2d<f32>;
      @group(0) @binding(1) var waterSampler : sampler;

      struct NormalUniforms {
        delta : vec2f,  // Texel size (1/width, 1/height)
      }
      @group(0) @binding(2) var<uniform> u : NormalUniforms;

      @fragment
      fn fs_main(@location(0) uv : vec2f) -> @location(0) vec4f {
        var info = textureSample(waterTexture, waterSampler, uv);

        // Sample neighboring heights to compute gradient
        let val_dx = textureSample(waterTexture, waterSampler, vec2f(uv.x + u.delta.x, uv.y)).r;
        let val_dy = textureSample(waterTexture, waterSampler, vec2f(uv.x, uv.y + u.delta.y)).r;

        // Create tangent vectors from height differences
        let dx = vec3f(u.delta.x, val_dx - info.r, 0.0);
        let dy = vec3f(0.0, val_dy - info.r, u.delta.y);

        // Normal is cross product of tangent vectors
        let normal = normalize(cross(dy, dx));
        info.b = normal.x;  // Store X component
        info.a = normal.z;  // Store Z component

        return info;
      }
    `, 16, format);

    // --- Sphere Interaction Pipeline ---
    // Displaces water based on sphere movement
    // Adds volume where sphere leaves, removes where it enters
    this.spherePipeline = this.createPipeline('Sphere', fullscreenQuadVS, `
      @group(0) @binding(0) var waterTexture : texture_2d<f32>;
      @group(0) @binding(1) var waterSampler : sampler;

      struct SphereUniforms {
        oldCenter : vec3f,  // Previous sphere position
        radius : f32,       // Sphere radius
        newCenter : vec3f,  // Current sphere position
        padding : f32,      // Alignment padding
      }
      @group(0) @binding(2) var<uniform> u : SphereUniforms;

      // Calculates the volume of sphere intersecting the water at a UV position
      fn volumeInSphere(center : vec3f, uv : vec2f, radius : f32) -> f32 {
        let p = vec3f(uv.x * 2.0 - 1.0, 0.0, uv.y * 2.0 - 1.0);
        let dist = length(p - center);
        let t = dist / radius;

        // Gaussian-like falloff for smooth interaction
        let dy = exp(-pow(t * 1.5, 6.0));
        let ymin = min(0.0, center.y - dy);
        let ymax = min(max(0.0, center.y + dy), ymin + 2.0 * dy);
        return (ymax - ymin) * 0.1;
      }

      @fragment
      fn fs_main(@location(0) uv : vec2f) -> @location(0) vec4f {
        var info = textureSample(waterTexture, waterSampler, uv);

        // Water rises where sphere was, falls where sphere is now
        info.r += volumeInSphere(u.oldCenter, uv, u.radius);
        info.r -= volumeInSphere(u.newCenter, uv, u.radius);

        return info;
      }
    `, 32, format);
  }

  /**
   * Helper to create a simulation pipeline.
   *
   * @param label - Debug label for the pipeline
   * @param vsCode - Vertex shader WGSL code
   * @param fsCode - Fragment shader WGSL code
   * @param uniformSize - Size of the uniform buffer in bytes
   * @param format - Texture format for output
   * @returns PipelineConfig with pipeline and uniform buffer
   */
  private createPipeline(label: string, vsCode: string, fsCode: string, uniformSize: number, format: GPUTextureFormat): PipelineConfig {
    const module = this.device.createShaderModule({
      label: label + ' Module',
      code: vsCode + fsCode
    });

    const pipeline = this.device.createRenderPipeline({
      label: label + ' Pipeline',
      layout: 'auto',
      vertex: {
        module: module,
        entryPoint: 'vs_main',
      },
      fragment: {
        module: module,
        entryPoint: 'fs_main',
        targets: [{ format: format }]
      },
      primitive: {
        topology: 'triangle-list',
      }
    });

    return {
      pipeline,
      uniformSize,
      uniformBuffer: this.device.createBuffer({
        size: uniformSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
    };
  }

  /**
   * Executes a simulation pipeline pass.
   *
   * Renders textureA through the pipeline to textureB,
   * then swaps the textures for double-buffering.
   *
   * @param pipelineObj - The pipeline configuration to run
   * @param uniformsData - Uniform data to upload
   */
  private runPipeline(pipelineObj: PipelineConfig, uniformsData: Float32Array<ArrayBuffer>): void {
    // Upload uniforms
    this.device.queue.writeBuffer(pipelineObj.uniformBuffer, 0, uniformsData);

    // Create bind group with input texture and uniforms
    const bindGroup = this.device.createBindGroup({
      layout: pipelineObj.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.textureA.createView() },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: pipelineObj.uniformBuffer } }
      ]
    });

    // Execute render pass
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.textureB.createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 0 }
      }]
    });

    pass.setPipeline(pipelineObj.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6); // Fullscreen quad (2 triangles)
    pass.end();

    this.device.queue.submit([encoder.finish()]);

    // Swap textures for double-buffering
    const temp = this.textureA;
    this.textureA = this.textureB;
    this.textureB = temp;
  }

  /**
   * Adds a circular ripple to the water surface.
   *
   * @param x - X position in [-1, 1] range
   * @param y - Y position in [-1, 1] range
   * @param radius - Radius of the ripple
   * @param strength - Intensity (positive = up, negative = down)
   */
  addDrop(x: number, y: number, radius: number, strength: number): void {
    const data = new Float32Array(4);
    data[0] = x; data[1] = y;
    data[2] = radius;
    data[3] = strength;
    this.runPipeline(this.dropPipeline, data);
  }

  /**
   * Advances the water simulation by one time step.
   *
   * Should be called multiple times per frame for smoother simulation.
   */
  stepSimulation(): void {
    const data = new Float32Array(2);
    data[0] = 1.0 / this.width;
    data[1] = 1.0 / this.height;
    this.runPipeline(this.updatePipeline, data);
  }

  /**
   * Recomputes surface normals from current height data.
   *
   * Should be called after simulation steps, before rendering.
   */
  updateNormals(): void {
    const data = new Float32Array(2);
    data[0] = 1.0 / this.width;
    data[1] = 1.0 / this.height;
    this.runPipeline(this.normalPipeline, data);
  }

  /**
   * Updates water displacement based on sphere movement.
   *
   * @param oldCenter - Previous sphere position [x, y, z]
   * @param newCenter - Current sphere position [x, y, z]
   * @param radius - Sphere radius
   */
  moveSphere(oldCenter: number[], newCenter: number[], radius: number): void {
    const data = new Float32Array(8);
    data[0] = oldCenter[0]; data[1] = oldCenter[1]; data[2] = oldCenter[2];
    data[3] = radius;
    data[4] = newCenter[0]; data[5] = newCenter[1]; data[6] = newCenter[2];
    data[7] = 0; // padding
    this.runPipeline(this.spherePipeline, data);
  }

  // =========================================================================
  // Surface Rendering
  // =========================================================================

  /**
   * Creates the water surface mesh as a subdivided plane.
   *
   * The plane spans from -1 to 1 on X and Z axes.
   * Higher detail (200x200) provides smooth displacement from wave heights.
   */
  private createSurfaceMesh(): void {
    const detail = 200; // Grid resolution
    const positions: number[] = [];
    const indices: number[] = [];

    // Generate vertex grid from -1 to 1 on X and Z
    for (let z = 0; z <= detail; z++) {
      const t = z / detail;
      for (let x = 0; x <= detail; x++) {
        const s = x / detail;
        // Store as XY initially (Z will be sampled from texture)
        positions.push(2 * s - 1, 2 * t - 1, 0);
      }
    }

    // Generate triangle indices
    for (let z = 0; z < detail; z++) {
      for (let x = 0; x < detail; x++) {
        const i = x + z * (detail + 1);
        // Two triangles per quad
        indices.push(i, i + 1, i + detail + 1);
        indices.push(i + detail + 1, i + 1, i + detail + 2);
      }
    }

    this.vertexCount = indices.length;

    // Create vertex buffer
    this.positionBuffer = this.device.createBuffer({
      label: 'Water Surface Vertices',
      size: positions.length * 4,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.positionBuffer.getMappedRange()).set(positions);
    this.positionBuffer.unmap();

    // Create index buffer
    this.indexBuffer = this.device.createBuffer({
      label: 'Water Surface Indices',
      size: indices.length * 4,
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true,
    });
    new Uint32Array(this.indexBuffer.getMappedRange()).set(indices);
    this.indexBuffer.unmap();
  }

  /**
   * Creates the water surface rendering pipelines.
   *
   * Two pipelines are created:
   * - Above: For viewing water from above (culls front faces)
   * - Under: For viewing water from below (culls back faces)
   *
   * The shader implements ray tracing for reflections and refractions,
   * with Fresnel blending between them.
   */
  private createSurfacePipeline(): void {
    /**
     * Generates shader code for water surface rendering.
     * The isUnderwater parameter controls the perspective and Fresnel calculations.
     */
    const shaderCode = (isUnderwater: boolean): string => `
        // Common camera uniforms
        struct CommonUniforms {
          viewProjectionMatrix : mat4x4f,
          eyePosition : vec3f,
        }
        @binding(0) @group(0) var<uniform> commonUniforms : CommonUniforms;

        // Light direction
        struct LightUniforms {
           direction : vec3f,
        }
        @binding(1) @group(0) var<uniform> light : LightUniforms;

        // Sphere for ray intersection
        struct SphereUniforms {
          center : vec3f,
          radius : f32,
        }
        @binding(2) @group(0) var<uniform> sphere : SphereUniforms;

        // Textures for rendering
        @binding(3) @group(0) var tileSampler : sampler;
        @binding(4) @group(0) var tileTexture : texture_2d<f32>;
        @binding(5) @group(0) var waterSampler : sampler;
        @binding(6) @group(0) var waterTexture : texture_2d<f32>;
        @binding(7) @group(0) var skySampler : sampler;
        @binding(8) @group(0) var skyTexture : texture_cube<f32>;
        @binding(9) @group(0) var causticTexture : texture_2d<f32>;

        // Physical constants
        const IOR_AIR : f32 = 1.0;
        const IOR_WATER : f32 = 1.333;
        const ABOVewaterColor : vec3f = vec3f(0.25, 1.0, 1.25);
        const UNDERwaterColor : vec3f = vec3f(0.4, 0.9, 1.0);

        struct VertexOutput {
          @builtin(position) position : vec4f,
          @location(0) worldPos : vec3f,
        }

        @vertex
        fn vs_main(@location(0) position : vec3f) -> VertexOutput {
          var output : VertexOutput;

          // Sample water height at this vertex position
          let uv = position.xy * 0.5 + 0.5;
          let info = textureSampleLevel(waterTexture, waterSampler, uv, 0.0);

          // Transform from XY plane to XZ plane with height from texture
          var pos = position.xzy;
          pos.y = info.r;

          output.worldPos = pos;
          output.position = commonUniforms.viewProjectionMatrix * vec4f(pos, 1.0);

          return output;
        }

        // Ray-box intersection for pool walls
        fn intersectCube(origin: vec3f, ray: vec3f, cubeMin: vec3f, cubeMax: vec3f) -> vec2f {
          let tMin = (cubeMin - origin) / ray;
          let tMax = (cubeMax - origin) / ray;
          let t1 = min(tMin, tMax);
          let t2 = max(tMin, tMax);
          let tNear = max(max(t1.x, t1.y), t1.z);
          let tFar = min(min(t2.x, t2.y), t2.z);
          return vec2f(tNear, tFar);
        }

        // Ray-sphere intersection
        fn intersectSphere(origin: vec3f, ray: vec3f, sphereCenter: vec3f, sphereRadius: f32) -> f32 {
            let toSphere = origin - sphereCenter;
            let a = dot(ray, ray);
            let b = 2.0 * dot(toSphere, ray);
            let c = dot(toSphere, toSphere) - sphereRadius * sphereRadius;
            let discriminant = b*b - 4.0*a*c;
            if (discriminant > 0.0) {
              let t = (-b - sqrt(discriminant)) / (2.0 * a);
              if (t > 0.0) { return t; }
            }
            return 1.0e6; // No hit
        }

        // Calculates sphere color at hit point (same as sphere.ts shader)
        fn getSphereColor(point: vec3f, IOR_AIR: f32, IOR_WATER: f32) -> vec3f {
            var color = vec3f(0.5);
            let sphereRadius = sphere.radius;

            // Distance-based darkening near pool walls
            color *= 1.0 - 0.9 / pow((1.0 + sphereRadius - abs(point.x)) / sphereRadius, 3.0);
            color *= 1.0 - 0.9 / pow((1.0 + sphereRadius - abs(point.z)) / sphereRadius, 3.0);
            color *= 1.0 - 0.9 / pow((point.y + 1.0 + sphereRadius) / sphereRadius, 3.0);

            // Diffuse lighting with caustics
            let sphereNormal = (point - sphere.center) / sphereRadius;
            let refractedLight = refract(-light.direction, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
            var diffuse = max(0.0, dot(-refractedLight, sphereNormal)) * 0.5;

            let info = textureSampleLevel(waterTexture, waterSampler, point.xz * 0.5 + 0.5, 0.0);
            if (point.y < info.r) {
                let causticUV = 0.75 * (point.xz - point.y * refractedLight.xz / refractedLight.y) * 0.5 + 0.5;
                let caustic = textureSampleLevel(causticTexture, waterSampler, causticUV, 0.0);
                diffuse *= caustic.r * 4.0;
            }
            color += diffuse;
            return color;
        }

        // Calculates pool wall color at hit point
        fn getWallColor(point: vec3f, IOR_AIR: f32, IOR_WATER: f32, poolHeight: f32) -> vec3f {
            var wallColor : vec3f;
            var normal = vec3f(0.0, 1.0, 0.0);

            // Sample tile texture based on wall orientation
            if (abs(point.x) > 0.999) {
                wallColor = textureSampleLevel(tileTexture, tileSampler, point.yz * 0.5 + vec2f(1.0, 0.5), 0.0).rgb;
                normal = vec3f(-point.x, 0.0, 0.0);
            } else if (abs(point.z) > 0.999) {
                wallColor = textureSampleLevel(tileTexture, tileSampler, point.yx * 0.5 + vec2f(1.0, 0.5), 0.0).rgb;
                normal = vec3f(0.0, 0.0, -point.z);
            } else {
                wallColor = textureSampleLevel(tileTexture, tileSampler, point.xz * 0.5 + 0.5, 0.0).rgb;
            }

            // Ambient occlusion
            var scale = 0.5;
            scale /= length(point);
            scale *= 1.0 - 0.9 / pow(length(point - sphere.center) / sphere.radius, 4.0);

            // Lighting with caustics or rim shadow
            let refractedLight = -refract(-light.direction, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
            var diffuse = max(0.0, dot(refractedLight, normal));

            let info = textureSampleLevel(waterTexture, waterSampler, point.xz * 0.5 + 0.5, 0.0);
            if (point.y < info.r) {
                let causticUV = 0.75 * (point.xz - point.y * refractedLight.xz / refractedLight.y) * 0.5 + 0.5;
                let caustic = textureSampleLevel(causticTexture, waterSampler, causticUV, 0.0);
                scale += diffuse * caustic.r * 2.0 * caustic.g;
            } else {
                let t = intersectCube(point, refractedLight, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
                diffuse *= 1.0 / (1.0 + exp(-200.0 / (1.0 + 10.0 * (t.y - t.x)) * (point.y + refractedLight.y * t.y - 2.0 / 12.0)));
                scale += diffuse * 0.5;
            }

            return wallColor * scale;
        }

        // Traces a ray from water surface to find color
        fn getSurfaceRayColor(origin: vec3f, ray: vec3f, waterColor: vec3f) -> vec3f {
            var color : vec3f;
            let poolHeight = 1.0;

            // Check sphere intersection first
            let q = intersectSphere(origin, ray, sphere.center, sphere.radius);
            if (q < 1.0e6) {
                color = getSphereColor(origin + ray * q, IOR_AIR, IOR_WATER);
            } else if (ray.y < 0.0) {
                // Ray going down - hit pool walls/floor
                let t = intersectCube(origin, ray, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
                color = getWallColor(origin + ray * t.y, IOR_AIR, IOR_WATER, poolHeight);
            } else {
                // Ray going up - hit walls or sky
                let t = intersectCube(origin, ray, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
                let hit = origin + ray * t.y;
                if (hit.y < 2.0 / 12.0) {
                    color = getWallColor(hit, IOR_AIR, IOR_WATER, poolHeight);
                } else {
                    // Sample skybox
                    color = textureSampleLevel(skyTexture, skySampler, ray, 0.0).rgb;
                    // Add sun specular highlight
                    let sunDir = normalize(light.direction);
                    let spec = pow(max(0.0, dot(sunDir, ray)), 5000.0);
                    color += vec3f(spec) * vec3f(10.0, 8.0, 6.0);
                }
            }

            // Apply underwater tint for downward rays
            if (ray.y < 0.0) {
                color *= waterColor;
            }
            return color;
        }

        @fragment
        fn fs_main(@location(0) worldPos : vec3f) -> @location(0) vec4f {
            // Sample normal with UV refinement for smooth appearance
            var uv = worldPos.xz * 0.5 + 0.5;
            var info = textureSampleLevel(waterTexture, waterSampler, uv, 0.0);

            // Iteratively refine UV based on normal offset
            for (var i = 0; i < 5; i++) {
                uv += info.ba * 0.005;
                info = textureSampleLevel(waterTexture, waterSampler, uv, 0.0);
            }

            // Reconstruct normal from BA channels
            let ba = vec2f(info.b, info.a);
            var normal = vec3f(info.b, sqrt(max(0.0, 1.0 - dot(ba, ba))), info.a);

            // Ray from camera to water surface
            let incomingRay = normalize(worldPos - commonUniforms.eyePosition);

            ${isUnderwater ? `
            // UNDERWATER VIEW: Looking up at water surface
            normal = -normal; // Flip normal for underwater
            let reflectedRay = reflect(incomingRay, normal);
            let refractedRay = refract(incomingRay, normal, IOR_WATER / IOR_AIR);
            let fresnel = mix(0.5, 1.0, pow(1.0 - dot(normal, -incomingRay), 3.0));

            let reflectedColor = getSurfaceRayColor(worldPos, reflectedRay, UNDERwaterColor);
            let refractedColor = getSurfaceRayColor(worldPos, refractedRay, vec3f(1.0)) * vec3f(0.8, 1.0, 1.1);

            let finalColor = mix(reflectedColor, refractedColor, (1.0 - fresnel) * length(refractedRay));
            ` : `
            // ABOVE WATER VIEW: Looking down at water surface
            let reflectedRay = reflect(incomingRay, normal);
            let refractedRay = refract(incomingRay, normal, IOR_AIR / IOR_WATER);
            let fresnel = mix(0.25, 1.0, pow(1.0 - dot(normal, -incomingRay), 3.0));

            let reflectedColor = getSurfaceRayColor(worldPos, reflectedRay, ABOVewaterColor);
            let refractedColor = getSurfaceRayColor(worldPos, refractedRay, ABOVewaterColor);

            let finalColor = mix(refractedColor, reflectedColor, fresnel);
            `}

            return vec4f(finalColor, 1.0);
        }
        `;

    // Create bind group layout (shared by both pipelines)
    this.surfaceBindGroupLayout = this.device.createBindGroupLayout({
      label: 'Water Surface BindGroupLayout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 5, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 6, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: 'cube' } },
        { binding: 9, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      ],
    });

    const surfacePipelineLayout = this.device.createPipelineLayout({
      label: 'Water Surface PipelineLayout',
      bindGroupLayouts: [this.surfaceBindGroupLayout],
    });

    /**
     * Helper to create a surface pipeline with specific settings.
     */
    const createSurfacePipeline = (label: string, isUnderwater: boolean, cullMode: GPUCullMode): GPURenderPipeline => {
      const shaderModule = this.device.createShaderModule({
        label: `${label} Shader`,
        code: shaderCode(isUnderwater),
      });

      return this.device.createRenderPipeline({
        label,
        layout: surfacePipelineLayout,
        vertex: {
          module: shaderModule,
          entryPoint: 'vs_main',
          buffers: [{
            arrayStride: 3 * 4,
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
          targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }]
        },
        primitive: {
          topology: 'triangle-list',
          cullMode,
        },
        depthStencil: {
          depthWriteEnabled: true,
          depthCompare: 'less',
          format: 'depth24plus',
        }
      });
    };

    // Create both pipelines
    this.surfacePipelineAbove = createSurfacePipeline(
      'Water Surface Above Pipeline',
      false,
      'front' // Cull front faces (see back face = top of water)
    );
    this.surfacePipelineUnder = createSurfacePipeline(
      'Water Surface Under Pipeline',
      true,
      'back' // Cull back faces (see front face = bottom of water)
    );
  }

  /**
   * Renders the water surface to the current render pass.
   *
   * Renders twice: once for above-water view, once for underwater view.
   * The appropriate pipeline is selected based on face culling.
   *
   * @param passEncoder - The active render pass encoder
   */
  renderSurface(passEncoder: GPURenderPassEncoder): void {
    // Create bind group with all required resources
    const bindGroup = this.device.createBindGroup({
      layout: this.surfaceBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.commonUniformBuffer } },
        { binding: 1, resource: { buffer: this.lightUniformBuffer } },
        { binding: 2, resource: { buffer: this.sphereUniformBuffer } },
        { binding: 3, resource: this.tileSampler },
        { binding: 4, resource: this.tileTexture.createView() },
        { binding: 5, resource: this.sampler },
        { binding: 6, resource: this.textureA.createView() },
        { binding: 7, resource: this.skySampler },
        { binding: 8, resource: this.skyTexture.createView({ dimension: 'cube' }) },
        { binding: 9, resource: this.causticsTexture.createView() }
      ]
    });

    // Render water surface from above
    passEncoder.setPipeline(this.surfacePipelineAbove);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setVertexBuffer(0, this.positionBuffer);
    passEncoder.setIndexBuffer(this.indexBuffer, 'uint32');
    passEncoder.drawIndexed(this.vertexCount);

    // Render water surface from below (same geometry, different shader)
    passEncoder.setPipeline(this.surfacePipelineUnder);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.drawIndexed(this.vertexCount);
  }

  // =========================================================================
  // Caustics Rendering
  // =========================================================================

  /**
   * Creates the caustics rendering pipeline.
   *
   * Caustics are the light patterns on the pool floor caused by
   * refraction through the water surface. The algorithm:
   * 1. For each water surface vertex, trace refracted light ray to pool floor
   * 2. Compare old position (flat water) to new position (displaced water)
   * 3. Light intensity is proportional to area ratio (convergence = brighter)
   *
   * Uses additive blending to accumulate light from multiple rays.
   */
  private createCausticsPipeline(): void {
    const shaderModule = this.device.createShaderModule({
      label: 'Caustics Shader',
      code: `
        // Light direction for refraction calculation
        struct LightUniforms {
           direction : vec3f,
        }
        @binding(0) @group(0) var<uniform> light : LightUniforms;

        // Sphere for shadow calculation
        struct SphereUniforms {
          center : vec3f,
          radius : f32,
        }
        @binding(1) @group(0) var<uniform> sphere : SphereUniforms;

        // Shadow toggle flags
        struct ShadowUniforms {
            rim : f32,
            sphere : f32,
            ao : f32,
        }
        @binding(4) @group(0) var<uniform> shadows : ShadowUniforms;

        // Water simulation texture
        @binding(2) @group(0) var waterSampler : sampler;
        @binding(3) @group(0) var waterTexture : texture_2d<f32>;

        struct VertexOutput {
          @builtin(position) position : vec4f,
          @location(0) oldPos : vec3f,  // Where ray would hit with flat water
          @location(1) newPos : vec3f,  // Where ray hits with displaced water
          @location(2) ray : vec3f,     // Refracted ray direction
        }

        // Ray-box intersection
        fn intersectCube(origin: vec3f, ray: vec3f, cubeMin: vec3f, cubeMax: vec3f) -> vec2f {
          let tMin = (cubeMin - origin) / ray;
          let tMax = (cubeMax - origin) / ray;
          let t1 = min(tMin, tMax);
          let t2 = max(tMin, tMax);
          let tNear = max(max(t1.x, t1.y), t1.z);
          let tFar = min(min(t2.x, t2.y), t2.z);
          return vec2f(tNear, tFar);
        }

        // Projects ray from water surface to pool floor
        fn project(origin: vec3f, ray: vec3f, refractedLight: vec3f) -> vec3f {
            let poolHeight = 1.0;
            var point = origin;

            // First find where ray exits pool volume
            let tcube = intersectCube(origin, ray, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
            point += ray * tcube.y;

            // Then project down to floor plane (y = -1)
            let tplane = (-point.y - 1.0) / refractedLight.y;
            return point + refractedLight * tplane;
        }

        @vertex
        fn vs_main(@location(0) position : vec3f) -> VertexOutput {
          var output : VertexOutput;
          let uv = position.xy * 0.5 + 0.5;

          // Sample water height and normal
          let info = textureSampleLevel(waterTexture, waterSampler, uv, 0.0);

          // Reconstruct normal (scaled down for stability)
          let ba = info.ba * 0.5;
          let normal = vec3f(ba.x, sqrt(max(0.0, 1.0 - dot(ba, ba))), ba.y);

          // Calculate refracted light directions
          let IOR_AIR = 1.0;
          let IOR_WATER = 1.333;
          let lightDir = normalize(light.direction);

          // Flat water refraction (reference)
          let refractedLight = refract(-lightDir, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
          // Displaced water refraction (actual)
          let ray = refract(-lightDir, normal, IOR_AIR / IOR_WATER);

          // Water surface position
          let pos = vec3f(position.x, 0.0, position.y);

          // Project both rays to pool floor
          output.oldPos = project(pos, refractedLight, refractedLight);
          output.newPos = project(pos + vec3f(0.0, info.r, 0.0), ray, refractedLight);
          output.ray = ray;

          // Position in caustics texture space
          let projectedPos = 0.75 * (output.newPos.xz - output.newPos.y * refractedLight.xz / refractedLight.y);
          output.position = vec4f(projectedPos.x, -projectedPos.y, 0.0, 1.0);

          return output;
        }

        @fragment
        fn fs_main(@location(0) oldPos : vec3f, @location(1) newPos : vec3f, @location(2) ray : vec3f) -> @location(0) vec4f {
            // Calculate intensity from area ratio using screen-space derivatives
            // Light converges where triangles shrink, diverges where they grow
            let oldArea = length(dpdx(oldPos)) * length(dpdy(oldPos));
            let newArea = length(dpdx(newPos)) * length(dpdy(newPos));

            var intensity = oldArea / newArea * 0.2;

            // Calculate sphere shadow
            let IOR_AIR = 1.0;
            let IOR_WATER = 1.333;
            let lightDir = normalize(light.direction);
            let refractedLight = refract(-lightDir, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);

            // Sphere shadow using distance to ray
            let dir = (sphere.center - newPos) / sphere.radius;
            let area = cross(dir, refractedLight);
            var shadow = dot(area, area);
            let dist = dot(dir, -refractedLight);

            shadow = 1.0 + (shadow - 1.0) / (0.05 + dist * 0.025);
            shadow = clamp(1.0 / (1.0 + exp(-shadow)), 0.0, 1.0);
            shadow = mix(1.0, shadow, clamp(dist * 2.0, 0.0, 1.0));
            shadow = mix(1.0, shadow, shadows.sphere);

            // Rim shadow at pool edges
            let poolHeight = 1.0;
            let t = intersectCube(newPos, -refractedLight, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
            let rimShadow = 1.0 / (1.0 + exp(-200.0 / (1.0 + 10.0 * (t.y - t.x)) * (newPos.y - refractedLight.y * t.y - 2.0 / 12.0)));
            intensity *= mix(1.0, rimShadow, shadows.rim);

            // R = caustic intensity, G = sphere shadow factor
            return vec4f(intensity, shadow, 0.0, 1.0);
        }
        `
    });

    this.causticsPipeline = this.device.createRenderPipeline({
      label: 'Caustics Pipeline',
      layout: 'auto',
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 3 * 4,
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
        targets: [{
          format: 'rgba8unorm',
          // Additive blending: multiple rays contribute to same pixel
          blend: {
            color: {
              operation: 'add',
              srcFactor: 'one',
              dstFactor: 'one',
            },
            alpha: {
              operation: 'add',
              srcFactor: 'one',
              dstFactor: 'one',
            }
          }
        }]
      },
      primitive: {
        topology: 'triangle-list',
      }
    });
  }

  /**
   * Updates the caustics texture.
   *
   * Should be called after water simulation and normal updates.
   * The caustics texture is then used by pool and sphere shaders.
   */
  updateCaustics(): void {
    const bindGroup = this.device.createBindGroup({
      layout: this.causticsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.lightUniformBuffer } },
        { binding: 1, resource: { buffer: this.sphereUniformBuffer } },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: this.textureA.createView() },
        { binding: 4, resource: { buffer: this.shadowUniformBuffer } }
      ]
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.causticsTexture.createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 0 }
      }]
    });

    pass.setPipeline(this.causticsPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, this.positionBuffer);
    pass.setIndexBuffer(this.indexBuffer, 'uint32');
    pass.drawIndexed(this.vertexCount);
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }
}
