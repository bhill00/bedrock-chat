import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

// Fix: @aws-amplify packages need xstate v4, but the app uses xstate v5.
// We install xstate v4 as "xstate-v4" (npm alias) and redirect @aws-amplify
// imports to it. This avoids Rollup's module resolution issues entirely.
function amplifyXstateFixPlugin(): Plugin {
  return {
    name: 'amplify-xstate-fix',
    enforce: 'pre',
    async resolveId(source, importer, options) {
      if (source !== 'xstate' || !importer) return null;

      // Redirect xstate imports from @aws-amplify packages to xstate-v4
      if (importer.includes('@aws-amplify') || importer.includes('@xstate')) {
        const result = await this.resolve('xstate-v4', importer, {
          ...options,
          skipSelf: true,
        });
        return result;
      }
      return null;
    },
  };
}

// https://vitejs.dev/config/
export default defineConfig({
  resolve: { alias: { './runtimeConfig': './runtimeConfig.browser' } },
  plugins: [
    amplifyXstateFixPlugin(),
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      devOptions: {
        enabled: true,
      },
      injectRegister: 'auto',
      workbox: {
        maximumFileSizeToCacheInBytes: 4 * 1024 * 1024,
      },
      manifest: {
        name: 'Bedrock Chat',
        short_name: 'Bedrock Chat',
        description: 'AWS-native chatbot using Bedrock',
        start_url: '/index.html',
        display: 'standalone',
        theme_color: '#232F3E',
        icons: [
          {
            src: '/images/bedrock_icon_72.png',
            sizes: '72x72',
            type: 'image/png',
          },
          {
            src: '/images/bedrock_icon_96.png',
            sizes: '96x96',
            type: 'image/png',
          },
          {
            src: '/images/bedrock_icon_128.png',
            sizes: '128x128',
            type: 'image/png',
          },
          {
            src: '/images/bedrock_icon_144.png',
            sizes: '144x144',
            type: 'image/png',
          },
          {
            src: '/images/bedrock_icon_152.png',
            sizes: '152x152',
            type: 'image/png',
          },
          {
            src: '/images/bedrock_icon_192.png',
            sizes: '192x192',
            type: 'image/png',
          },
          {
            src: '/images/bedrock_icon_384.png',
            sizes: '384x384',
            type: 'image/png',
          },
          {
            src: '/images/bedrock_icon_512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'maskable',
          },
          {
            src: '/images/bedrock_icon_512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any',
          },
        ],
      },
    }),
  ],
  server: { host: true },
});
