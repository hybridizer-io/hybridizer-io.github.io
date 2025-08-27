// @ts-check

const config = {
  title: 'Hybridizer Documentation',
  tagline: 'Performance everywhere, same code.',
  favicon: 'img/favicon.ico',

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

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */ ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/hybridizer-io/hybridizer-docs/edit/main/',
          routeBasePath: '/docs'
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
      title: 'Hybridizer',
      logo: {
        alt: 'Hybridizer Logo',
        src: 'img/logo.svg',
      },
      items: [
        {type: 'docSidebar', sidebarId: 'docsSidebar', position: 'left', label: 'Docs'},
        {href: 'https://docs.hybridizer.io', label: 'Legacy Docs', position: 'right'}
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Getting Started', to: '/docs'},
          ],
        },
        {
          title: 'Community',
          items: [
            {label: 'GitHub', href: 'https://github.com/hybridizer-io'}
          ],
        },
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
