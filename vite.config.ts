import { defineConfig } from 'vite';
import glsl from 'vite-plugin-glsl';

export default defineConfig({
  base: '/webgpu-water/',
  plugins: [
    glsl({
      include: ['**/*.wgsl', '**/*.vert', '**/*.frag'],
      warnDuplicatedImports: true,
    }),
  ],
});
