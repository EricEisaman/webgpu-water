/**
 * lightgl.ts - 3D Math Utilities for Ray Casting and Vector Operations
 *
 * Ported from lightgl.js (http://github.com/evanw/lightgl.js/) and adapted
 * for use with wgpu-matrix. Provides essential 3D math classes for:
 * - Vector operations (add, subtract, dot product, cross product, etc.)
 * - Ray casting from screen coordinates to world space
 * - Hit testing for sphere intersection (used for mouse interaction)
 */

import { mat4, vec3 } from 'wgpu-matrix';
import type { Mat4 } from 'wgpu-matrix';
import type { Viewport } from './types';

/**
 * A 3D vector class with common mathematical operations.
 * Used throughout the application for positions, directions, and physics calculations.
 */
export class Vector {
  x: number;
  y: number;
  z: number;

  /**
   * Creates a new Vector with the given coordinates.
   * @param x - X component (default: 0)
   * @param y - Y component (default: 0)
   * @param z - Z component (default: 0)
   */
  constructor(x = 0, y = 0, z = 0) {
    this.x = x;
    this.y = y;
    this.z = z;
  }

  /**
   * Returns a new vector pointing in the opposite direction.
   */
  negative(): Vector {
    return new Vector(-this.x, -this.y, -this.z);
  }

  /**
   * Adds a vector or scalar to this vector.
   * @param v - Vector to add, or scalar to add to all components
   */
  add(v: Vector | number): Vector {
    if (v instanceof Vector) {
      return new Vector(this.x + v.x, this.y + v.y, this.z + v.z);
    }
    return new Vector(this.x + v, this.y + v, this.z + v);
  }

  /**
   * Subtracts a vector or scalar from this vector.
   * @param v - Vector to subtract, or scalar to subtract from all components
   */
  subtract(v: Vector | number): Vector {
    if (v instanceof Vector) {
      return new Vector(this.x - v.x, this.y - v.y, this.z - v.z);
    }
    return new Vector(this.x - v, this.y - v, this.z - v);
  }

  /**
   * Multiplies this vector by another vector (component-wise) or a scalar.
   * @param v - Vector for component-wise multiplication, or scalar multiplier
   */
  multiply(v: Vector | number): Vector {
    if (v instanceof Vector) {
      return new Vector(this.x * v.x, this.y * v.y, this.z * v.z);
    }
    return new Vector(this.x * v, this.y * v, this.z * v);
  }

  /**
   * Divides this vector by another vector (component-wise) or a scalar.
   * @param v - Vector for component-wise division, or scalar divisor
   */
  divide(v: Vector | number): Vector {
    if (v instanceof Vector) {
      return new Vector(this.x / v.x, this.y / v.y, this.z / v.z);
    }
    return new Vector(this.x / v, this.y / v, this.z / v);
  }

  /**
   * Computes the dot product of this vector with another.
   * The dot product equals |a||b|cos(θ) where θ is the angle between vectors.
   * @param v - The other vector
   */
  dot(v: Vector): number {
    return this.x * v.x + this.y * v.y + this.z * v.z;
  }

  /**
   * Computes the cross product of this vector with another.
   * The result is perpendicular to both input vectors.
   * @param v - The other vector
   */
  cross(v: Vector): Vector {
    return new Vector(
      this.y * v.z - this.z * v.y,
      this.z * v.x - this.x * v.z,
      this.x * v.y - this.y * v.x
    );
  }

  /**
   * Returns the length (magnitude) of this vector.
   */
  length(): number {
    return Math.sqrt(this.dot(this));
  }

  /**
   * Returns a normalized (unit length) version of this vector.
   */
  unit(): Vector {
    return this.divide(this.length());
  }

  /**
   * Returns the minimum component value.
   */
  min(): number {
    return Math.min(this.x, this.y, this.z);
  }

  /**
   * Returns the maximum component value.
   */
  max(): number {
    return Math.max(this.x, this.y, this.z);
  }

  /**
   * Converts this vector to a number array [x, y, z].
   */
  toArray(): number[] {
    return [this.x, this.y, this.z];
  }

  /**
   * Creates a copy of this vector.
   */
  clone(): Vector {
    return new Vector(this.x, this.y, this.z);
  }

  /**
   * Creates a vector from spherical coordinates (angles).
   * @param theta - Azimuthal angle (rotation around Y axis)
   * @param phi - Polar angle (elevation from XZ plane)
   */
  static fromAngles(theta: number, phi: number): Vector {
    return new Vector(
      Math.cos(phi) * Math.cos(theta),
      Math.sin(phi),
      Math.cos(phi) * Math.sin(theta)
    );
  }

  /**
   * Linearly interpolates between two vectors.
   * @param a - Start vector (t=0)
   * @param b - End vector (t=1)
   * @param t - Interpolation factor [0, 1]
   */
  static lerp(a: Vector, b: Vector, t: number): Vector {
    return a.add(b.subtract(a).multiply(t));
  }

  /**
   * Returns a vector with the minimum components from two vectors.
   */
  static min(a: Vector, b: Vector): Vector {
    return new Vector(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.min(a.z, b.z));
  }

  /**
   * Returns a vector with the maximum components from two vectors.
   */
  static max(a: Vector, b: Vector): Vector {
    return new Vector(Math.max(a.x, b.x), Math.max(a.y, b.y), Math.max(a.z, b.z));
  }
}

/**
 * Result of a ray-object intersection test.
 * Contains information about where and how a ray hit an object.
 */
export class HitTest {
  /** Distance along the ray to the hit point */
  t: number;
  /** World-space position of the hit */
  hit: Vector;
  /** Surface normal at the hit point */
  normal: Vector;

  constructor(t: number, hit: Vector, normal: Vector) {
    this.t = t;
    this.hit = hit;
    this.normal = normal;
  }
}

/**
 * Utility class for casting rays from screen coordinates into 3D world space.
 * Used for mouse interaction - determining what the user clicked on.
 *
 * The raytracer pre-computes corner rays for efficient per-pixel ray generation
 * using bilinear interpolation.
 */
export class Raytracer {
  /** Camera position in world space */
  eye: Vector;

  /** Viewport dimensions [x, y, width, height] */
  private viewport: Viewport;
  /** Inverse of view-projection matrix for unprojecting screen coordinates */
  private invViewProj: Mat4;
  /** Pre-computed ray at top-left corner */
  private ray00: Vector;
  /** Pre-computed ray at top-right corner */
  private ray10: Vector;
  /** Pre-computed ray at bottom-left corner */
  private ray01: Vector;
  /** Pre-computed ray at bottom-right corner */
  private ray11: Vector;

  /**
   * Creates a new Raytracer for the given camera configuration.
   * @param viewMatrix - Camera view matrix (world to camera transform)
   * @param projectionMatrix - Camera projection matrix (camera to clip transform)
   * @param viewport - Screen viewport [x, y, width, height]
   */
  constructor(viewMatrix: Mat4, projectionMatrix: Mat4, viewport: Viewport) {
    this.viewport = viewport;

    // Calculate eye position by transforming origin through inverse view matrix
    const invView = mat4.invert(viewMatrix);
    const eyeVec = vec3.transformMat4([0, 0, 0], invView);
    this.eye = new Vector(eyeVec[0], eyeVec[1], eyeVec[2]);

    // Pre-compute inverse view-projection for unprojecting screen coordinates
    this.invViewProj = mat4.invert(mat4.multiply(projectionMatrix, viewMatrix));

    // Pre-compute corner rays for efficient interpolation
    const [minX, minY, width, height] = viewport;
    const maxX = minX + width;
    const maxY = minY + height;

    // Unproject corners at far plane (z=1) and compute ray directions
    this.ray00 = this.unProject(minX, minY, 1).subtract(this.eye);
    this.ray10 = this.unProject(maxX, minY, 1).subtract(this.eye);
    this.ray01 = this.unProject(minX, maxY, 1).subtract(this.eye);
    this.ray11 = this.unProject(maxX, maxY, 1).subtract(this.eye);
  }

  /**
   * Unprojects a screen coordinate to world space.
   * @param winX - Screen X coordinate
   * @param winY - Screen Y coordinate (0 = top)
   * @param winZ - Depth value [0=near, 1=far]
   */
  private unProject(winX: number, winY: number, winZ: number): Vector {
    const [vx, vy, vw, vh] = this.viewport;

    // Convert screen coordinates to normalized device coordinates (NDC)
    // NDC range: x,y = [-1, 1], z = [0, 1] for WebGPU
    const x = ((winX - vx) / vw) * 2 - 1;
    const y = (1 - (winY - vy) / vh) * 2 - 1; // Flip Y: screen Y=0 is top

    // Transform NDC to world space using inverse view-projection
    const world = vec3.transformMat4([x, y, winZ], this.invViewProj);
    return new Vector(world[0], world[1], world[2]);
  }

  /**
   * Gets a normalized ray direction for a screen pixel.
   * Uses bilinear interpolation of pre-computed corner rays for efficiency.
   * @param x - Screen X coordinate
   * @param y - Screen Y coordinate
   */
  getRayForPixel(x: number, y: number): Vector {
    const [vx, vy, vw, vh] = this.viewport;

    // Calculate interpolation factors
    const u = (x - vx) / vw; // 0 = left, 1 = right
    const v = (y - vy) / vh; // 0 = top, 1 = bottom

    // Bilinear interpolation of corner rays
    const rayTop = Vector.lerp(this.ray00, this.ray10, u);
    const rayBottom = Vector.lerp(this.ray01, this.ray11, u);

    return Vector.lerp(rayTop, rayBottom, v).unit();
  }

  /**
   * Tests if a ray intersects a sphere.
   * Uses the quadratic formula to solve the ray-sphere intersection equation.
   *
   * @param origin - Ray origin point
   * @param ray - Normalized ray direction
   * @param center - Sphere center position
   * @param radius - Sphere radius
   * @returns HitTest result if intersection found, null otherwise
   */
  static hitTestSphere(origin: Vector, ray: Vector, center: Vector, radius: number): HitTest | null {
    // Solve: |origin + t*ray - center|^2 = radius^2
    // Expanding: at^2 + bt + c = 0
    const offset = origin.subtract(center);
    const a = ray.dot(ray);
    const b = 2 * ray.dot(offset);
    const c = offset.dot(offset) - radius * radius;
    const discriminant = b * b - 4 * a * c;

    if (discriminant > 0) {
      // Take the closer intersection (smaller t)
      const t = (-b - Math.sqrt(discriminant)) / (2 * a);
      const hit = origin.add(ray.multiply(t));
      const normal = hit.subtract(center).divide(radius);
      return new HitTest(t, hit, normal);
    }

    return null;
  }
}
