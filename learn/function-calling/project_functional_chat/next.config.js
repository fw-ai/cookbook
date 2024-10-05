/* eslint-disable eslint-comments/disable-enable-pair */
/* eslint-disable node/no-unpublished-require */
/* eslint-disable unicorn/prefer-module */
/* eslint-disable @typescript-eslint/no-var-requires */
/* eslint-disable @typescript-eslint/no-require-imports */
/* eslint-disable import/no-commonjs */
const webpack = require('webpack');
const withBundleAnalyzer = require('@next/bundle-analyzer');
const { createVanillaExtractPlugin } = require('@vanilla-extract/next-plugin');

const withAnalyzer = withBundleAnalyzer({
  enabled: process.env.ANALYZE === 'true',
});

const withVanillaExtract = createVanillaExtractPlugin();

const securityHeaders = [
  {
    key: 'Strict-Transport-Security',
    value: 'max-age=63072000; includeSubDomains; preload',
  },
  {
    key: 'X-Content-Type-Options',
    value: 'nosniff',
  },
  {
    key: 'X-Permitted-Cross-Domain-Policies',
    value: 'none',
  },
  {
    key: 'Referrer-Policy',
    value: 'no-referrer',
  },
  {
    key: 'X-DNS-Prefetch-Control',
    value: 'on',
  },
  {
    key: 'X-Frame-Options',
    value: 'DENY',
  },
];

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    externalDir: true,
    typedRoutes: true,
    esmExternals: true,
  },
  webpack: (config, { isServer, dev }) => {
    config.plugins.unshift(
      new webpack.DefinePlugin({
        __DEV__: JSON.stringify(dev),
        'process.env.NODE_ENV': JSON.stringify(dev ? 'development' : 'production'),
        'process.env.FIREWORKS_ENV': JSON.stringify(isServer ? 'node' : 'web'),
      }),
    );
    if (!config.externalsPresets?.node) {
      // It seems like NextJS creates 3 webpack configs: 1/ server, 2/ client, and 3/ server side rendering of client code.
      // 1/ has config.externalsPresets.node set to true, so all node modules are excluded from the bundle.
      // 2/ and 3/ have it set to false, to we manually add node modules to exclude here.
      config.externals.push('v8', 'http2', 'child_process', 'fs');
    }

    return config;
  },
  poweredByHeader: false,
  async headers() {
    return [
      {
        source: '/:path*',
        headers: securityHeaders,
      },
    ];
  },
  async rewrites() {
    return [
      {
        source: '/accounts/:accountID/grafana',
        destination: '/api/accounts/:accountID/grafana',
      },
      {
        source: '/accounts/:accountID/grafana/:path*',
        destination: '/api/accounts/:accountID/grafana/:path*',
      },
    ];
  },
  modularizeImports: {
    '@mui/material': {
      transform: '@mui/material/{{member}}',
    },
    '@mui/icons-material': {
      transform: '@mui/icons-material/{{member}}',
    },
  },
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'storage.googleapis.com',
        port: '',
        pathname: '*',
      },
      {
        protocol: 'https',
        hostname: 'cdn-images-1.medium.com',
        port: '',
        pathname: '*',
      },
      {
        protocol: 'https',
        hostname: 'medium.com',
        port: '',
        pathname: '*',
      },
      {
        protocol: 'https',
        hostname: 'files.readme.io',
        port: '',
        pathname: '/**',
      },
    ],
  },
};

module.exports = withAnalyzer(withVanillaExtract(nextConfig));
