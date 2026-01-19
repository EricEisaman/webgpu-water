/**
 * types.ts - Shared TypeScript type definitions for the WebGPU Water simulation
 *
 * This file contains interfaces, types, and enums used across multiple modules
 * to ensure type safety and consistency throughout the application.
 */

import type { Mat4 } from 'wgpu-matrix';

/**
 * Configuration for a WebGPU render pipeline used in water simulation passes.
 * Each simulation step (drop, update, normal, sphere) uses this structure.
 */
export interface PipelineConfig {
  /** The compiled render pipeline for this simulation pass */
  pipeline: GPURenderPipeline;
  /** Size of the uniform buffer in bytes */
  uniformSize: number;
  /** GPU buffer containing shader uniforms for this pass */
  uniformBuffer: GPUBuffer;
}

/**
 * URLs for the six faces of a cubemap texture.
 * Used for loading skybox textures for environment reflections.
 */
export interface CubemapURLs {
  /** Positive X face (right) */
  xpos: string;
  /** Negative X face (left) */
  xneg: string;
  /** Positive Y face (top) */
  ypos: string;
  /** Negative Y face (bottom) */
  yneg: string;
  /** Positive Z face (front) */
  zpos: string;
  /** Negative Z face (back) */
  zneg: string;
}

/**
 * Pair of camera matrices used for 3D rendering.
 * These matrices transform world coordinates to clip space.
 */
export interface MatricesPair {
  /** Perspective projection matrix (defines field of view, aspect ratio, near/far planes) */
  projectionMatrix: Mat4;
  /** View matrix (defines camera position and orientation in world space) */
  viewMatrix: Mat4;
}

/**
 * Enumeration of mouse interaction modes.
 * Determines how mouse drag events are interpreted.
 */
export const enum InteractionMode {
  /** No active interaction */
  None = -1,
  /** Drawing ripples on the water surface */
  AddDrops = 0,
  /** Rotating the camera around the scene */
  OrbitCamera = 1,
  /** Dragging the sphere object */
  MoveSphere = 2,
}

/**
 * Viewport dimensions as a tuple: [x, y, width, height]
 * Used for ray casting and coordinate transformations.
 */
export type Viewport = [number, number, number, number];
