// @ts-check

const config = {
  title: 'Hybridizer Documentation',
  tagline: 'Performance everywhere, same code.',
  favicon: 'img/favicon.png',

  url: 'https://docs.hybridizer.io',
  baseUrl: '/',

  organizationName: 'hybridizer',
  projectName: 'hybridizer-docs',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },
  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],
  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */ ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/hybridizer-io/hybridizer-io.github.io/edit/master/',
          routeBasePath: '/'
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig: /** @type {import('@docusaurus/preset-classic').ThemeConfig} */ ({
    navbar: {

      logo: {
        alt: 'Hybridizer Logo',
        src: 'img/logo.svg',
        href: 'https://hybridizer.io'
      },
      items: [

        { href: 'https://docs.hybridizer.io', label: 'Legacy Docs', position: 'right' }
      ],
    },
    footer: {
      style: 'dark',
      links: [
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Hybridizer. Built with Love in Paris.`,
    },
    prism: {
      theme: require('prism-react-renderer').themes.github,
      darkTheme: require('prism-react-renderer').themes.dracula,
      additionalLanguages: ['csharp']
    },
  }),
};

module.exports = config;
