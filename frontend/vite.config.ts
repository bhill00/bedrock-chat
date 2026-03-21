import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';
import path from 'path';
import fs from 'fs';

// Fix: @aws-amplify/ui and @xstate/react v3 (nested under @aws-amplify)
// need xstate v4, but the top-level xstate is v5. Rollup resolves bare
// 'xstate' imports to the hoisted v5 regardless of npm overrides.
//
// This plugin intercepts 'xstate' imports from @aws-amplify packages and
// rewrites them to point at the nested v4 copy's ESM entry.
function amplifyXstateFixPlugin(): Plugin {
  let xstateV4Entry: string | null = null;

  return {
    name: 'amplify-xstate-fix',
    enforce: 'pre',

    configResolved() {
      // Find the xstate v4 ESM entry — npm installs it under
      // @aws-amplify/ui-react/node_modules/xstate (via overrides)
      const candidates = [
        'node_modules/@aws-amplify/ui-react/node_modules/xstate',
        'node_modules/@aws-amplify/ui-react-core/node_modules/xstate',
      ];
      for (const candidate of candidates) {
        const abs = path.resolve(candidate);
        if (fs.existsSync(abs)) {
          // xstate v4 has es/ directory with ESM build
          const esEntry = path.join(abs, 'es', 'index.js');
          const libEntry = path.join(abs, 'lib', 'index.js');
          if (fs.existsSync(esEntry)) {
            xstateV4Entry = esEntry;
          } else if (fs.existsSync(libEntry)) {
            xstateV4Entry = libEntry;
          } else {
            // Fallback: use package.json main/module field
            xstateV4Entry = abs;
          }
          break;
        }
      }
    },

    resolveId(source, importer) {
      if (source !== 'xstate' || !importer || !xstateV4Entry) return null;

      // Redirect xstate imports from @aws-amplify packages to v4
      if (importer.includes('@aws-amplify')) {
        return xstateV4Entry;
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
