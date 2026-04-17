/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docsSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'index',
        'overview/what-is-hybridizer',
        'overview/when-to-use'
      ],
    },
    {
      type: 'category',
      label: 'Tutorials',
      collapsed: false,
      link: {
        type: 'generated-index',
        title: 'Tutorials',
        description: 'Step-by-step tutorials from your first kernel to advanced GPU patterns.',
      },
      items: [
        'tutorials/setup',
        'tutorials/first-kernel',
        'tutorials/understanding-result',
        'tutorials/cpu-to-gpu',
        'tutorials/working-with-images',
        'tutorials/shared-memory-reduction',
      ],
    },
    {
      type: 'category',
      label: 'Concepts',
      items: [
        'overview/architecture',
        'guide/concepts',
        'guide/compilation-pipeline',
        'guide/data-marshalling',
        'guide/invoke-generated-code',
        'guide/generated-code-layout',
        'guide/intrinsics-builtins',
        'guide/generics-virtuals-delegates',
        'guide/line-info-and-debug',
      ],
    },
    {
      type: 'category',
      label: 'CUDA Background',
      collapsed: true,
      link: {
        type: 'generated-index',
        title: 'CUDA Background',
        description: 'Optional deep-dive into CUDA GPU concepts. Skip this if you already know CUDA.',
      },
      items: [
        'cuda/basics-threading',
        'cuda/functions',
        'cuda/memory-and-profiling',
        'cuda/perf-metrics',
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
      label: 'Examples',
      link: {
        type: 'generated-index',
        title: 'Code Examples',
        description: 'Complete working examples from the hybridizer-basic-samples repository.',
      },
      items: [
        'examples/hello-world',
        'examples/mandelbrot',
        'examples/reduction',
        'examples/sobel-filter',
        'examples/black-scholes',
        'examples/streams',
        'examples/constant-memory',
        'examples/generic-reduction',
        'examples/lambda-reduction',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/attributes-and-annotations',
        'reference/cli-options',
        'reference/api-index',
        'reference/glossary',
        'quickstart/run-and-debug',
        'quickstart/faq-troubleshooting',
      ],
    },
    {
      type: 'category',
      label: 'LLM Guide',
      collapsed: true,
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
