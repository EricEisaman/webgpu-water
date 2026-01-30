/**
 * cubemap.ts - Cubemap Texture Loader for Skybox Rendering
 *
 * This module handles loading six individual images and combining them into
 * a WebGPU cube texture. The cubemap is used for:
 * - Skybox rendering (background environment)
 * - Environment reflections on the water surface
 *
 * Cubemap faces are arranged as:
 * - xpos/xneg: Right/Left faces
 * - ypos/yneg: Top/Bottom faces
 * - zpos/zneg: Front/Back faces
 */

import type { CubemapURLs } from './types';

/**
 * Loader class for creating WebGPU cubemap textures from six image files.
 *
 * Usage:
 * ```typescript
 * const cubemap = new Cubemap(device);
 * const texture = await cubemap.load({
 *   xpos: 'right.jpg', xneg: 'left.jpg',
 *   ypos: 'top.jpg',   yneg: 'bottom.jpg',
 *   zpos: 'front.jpg', zneg: 'back.jpg'
 * });
 * ```
 */
export class Cubemap {
  /** WebGPU device for creating textures */
  private device: GPUDevice;

  /**
   * Creates a new Cubemap loader.
   * @param device - The WebGPU device to use for texture creation
   */
  constructor(device: GPUDevice) {
    this.device = device;
  }

  /**
   * Loads six images and creates a cubemap texture.
   *
   * The images are loaded in parallel for efficiency, then copied to
   * the appropriate layers of a 2D array texture with 6 layers.
   *
   * @param urls - Object containing URLs for each cubemap face
   * @returns Promise resolving to the created GPUTexture
   */
  async load(urls: CubemapURLs): Promise<GPUTexture> {
    // Standard cubemap face order expected by WebGPU
    const faces: (keyof CubemapURLs)[] = ['xpos', 'xneg', 'ypos', 'yneg', 'zpos', 'zneg'];

    // Load all six face images in parallel
    const images = await Promise.all(
      faces.map((face) =>
        fetch(urls[face])
          .then((r) => r.blob())
          .then((b) => createImageBitmap(b))
      )
    );

    // All faces must be the same size; use first image dimensions
    const { width, height } = images[0];

    // Create a cube texture (2D array with 6 layers)
    const texture = this.device.createTexture({
      size: [width, height, 6],
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // Copy each face image to its corresponding texture layer
    images.forEach((image, i) => {
      this.device.queue.copyExternalImageToTexture(
        { source: image, flipY: true }, // Flip Y to match WebGPU coordinate system
        { texture, origin: [0, 0, i] }, // Layer index determines cube face
        { width, height }
      );
    });

    return texture;
  }
}
