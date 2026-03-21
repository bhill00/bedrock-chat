import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

// Rollup plugin: resolve xstate to v4 when imported from @aws-amplify packages.
// @aws-amplify/ui uses xstate v4 API (named export "actions"), but the top-level
// xstate is v5 which removed that export. This plugin forces Amplify's imports
// to resolve to the nested v4 copy that npm installs via overrides.
function amplifyXstateFixPlugin(): Plugin {
  return {
    name: 'amplify-xstate-fix',
    enforce: 'pre',
    resolveId(source, importer) {
      if (
        source === 'xstate' &&
        importer &&
        importer.includes('@aws-amplify')
      ) {
        // Resolve to the v4 copy nested under @aws-amplify/ui-react
        return this.resolve(
          source,
          importer.replace(
            /node_modules\/@aws-amplify\/.*$/,
            'node_modules/@aws-amplify/ui-react/node_modules/xstate/lib/index.js'
          ),
          { skipSelf: true }
        );
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
