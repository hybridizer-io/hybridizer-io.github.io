import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';

export default function Home() {
  return (
    <Layout title="Hybridizer Docs" description="Performance everywhere, same code">
      <main style={{padding: '3rem 1rem'}}>
        <h1>Hybridizer</h1>
        <p>Write .NET or Java code once. Run fast everywhere: NVIDIA GPUs, x86/AVX, POWER, NEON and more. Hybridizer compiles MSIL/bytecode to optimized native backends.</p>
        <div style={{display:'flex', gap:'1rem', flexWrap:'wrap'}}>
          <Link className="button button--primary" to="/docs/quickstart/install">Get Started</Link>
          <Link className="button button--secondary" to="/docs/overview/what-is-hybridizer">What is Hybridizer?</Link>
          <Link className="button" to="/docs/guide/concepts">Programming Guide</Link>
        </div>
      </main>
    </Layout>
  );
}
