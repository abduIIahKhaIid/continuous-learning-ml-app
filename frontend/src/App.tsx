import { DataSubmissionForm } from './features/data/DataSubmissionForm'
import { ModelStatus } from './features/prediction/ModelStatus'
import { PredictionForm } from './features/prediction/PredictionForm'

function App() {
  return (
    <main className="page-shell">
      <div className="dashboard">
        <header className="page-header">
          <p className="eyebrow">Phase 4</p>
          <h1 id="page-title">ML samples and predictions</h1>
          <ModelStatus />
        </header>

        <div className="content-grid">
          <section className="card" aria-labelledby="prediction-title">
            <h2 id="prediction-title">Make a prediction</h2>
            <p className="intro">
              Run three features through the current trained model.
            </p>
            <PredictionForm />
          </section>

          <section className="card" aria-labelledby="sample-title">
            <h2 id="sample-title">Submit labelled sample data</h2>
            <p className="intro">
              Store demonstration features and an optional ground-truth label.
            </p>
            <DataSubmissionForm />
          </section>
        </div>
      </div>
    </main>
  )
}

export default App
