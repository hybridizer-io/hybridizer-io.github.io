/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docsSidebar: [
    {
      type: 'category',
      label: 'Welcome',
      collapsed: false,
      items: [
        'index',
        'overview/what-is-hybridizer',
        'overview/value-proposition',
        'overview/architecture',
        'overview/when-to-use'
      ],
    },
    {
      type: 'category',
      label: 'Quickstart',
      collapsed: false,
      items: [
        'quickstart/install',
        'quickstart/hello-gpu',
        'quickstart/run-and-debug',
        'quickstart/faq-troubleshooting'
      ],
    },
    {
      type: 'category',
      label: 'Programming Guide',
      items: [
        'guide/concepts',
        'guide/compilation-pipeline',
        'guide/invoke-generated-code',
        'guide/generated-code-layout',
        'guide/data-marshalling',
        'guide/intrinsics-builtins',
        'guide/generics-virtuals-delegates',
        'guide/line-info-and-debug'
      ],
    },
    {
      type: 'category',
      label: 'CUDA Basics',
      items: [
        'cuda/basics-threading',
        'cuda/memory-and-profiling',
        'cuda/perf-metrics'
      ],
    },
    {
      type: 'category',
      label: 'Platforms & Flavors',
      items: [
        'platforms/overview',
        'platforms/cuda',
        'platforms/omp-cuda',
        'platforms/vector-avx-neon'
      ],
    },
    {
      type: 'category',
      label: 'How-To Recipes',
      items: [
        'howto/port-existing-code',
        'howto/optimize-kernels',
        'howto/manage-memory',
        'howto/use-libraries',
        'howto/ci-cd-integration'
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/attributes-and-annotations',
        'reference/cli-options',
        'reference/api-index',
        'reference/glossary'
      ],
    },
    {
      type: 'category',
      label: 'LLM Guide',
      items: [
        'llm/overview',
        'llm/task-examples',
        'llm/terminology',
        'llm/code-patterns'
      ],
    }
  ],
};

module.exports = sidebars;
