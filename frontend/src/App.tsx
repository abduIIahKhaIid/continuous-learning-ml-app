import { DataSubmissionForm } from './features/data/DataSubmissionForm'

function App() {
  return (
    <main className="page-shell">
      <section className="card" aria-labelledby="page-title">
        <p className="eyebrow">Phase 1</p>
        <h1 id="page-title">Submit sample data</h1>
        <p className="intro">
          Send three demonstration features and an optional label to the
          FastAPI validation endpoint.
        </p>
        <DataSubmissionForm />
      </section>
    </main>
  )
}

export default App
