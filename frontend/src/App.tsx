import { useState } from 'react'

import { DataSubmissionForm } from './features/data/DataSubmissionForm'
import { ModelStatus } from './features/prediction/ModelStatus'
import { PredictionForm } from './features/prediction/PredictionForm'
import { PredictionHistory } from './features/prediction/PredictionHistory'
import { ContinuousTrainingStatus } from './features/training/ContinuousTrainingStatus'

function App() {
  const [historyRefreshToken, setHistoryRefreshToken] = useState(0)

  return (
    <main className="page-shell">
      <div className="dashboard">
        <header className="page-header">
          <p className="eyebrow">Phase 6</p>
          <h1 id="page-title">ML samples and predictions</h1>
          <ModelStatus />
        </header>

        <div className="content-grid">
          <section className="card" aria-labelledby="prediction-title">
            <h2 id="prediction-title">Make a prediction</h2>
            <p className="intro">
              Run three features through the current trained model.
            </p>
            <PredictionForm
              onPredictionCreated={() =>
                setHistoryRefreshToken((current) => current + 1)
              }
            />
          </section>

          <section className="card" aria-labelledby="sample-title">
            <h2 id="sample-title">Submit labelled sample data</h2>
            <p className="intro">
              Store demonstration features and an optional ground-truth label.
            </p>
            <DataSubmissionForm />
          </section>
        </div>

        <ContinuousTrainingStatus />

        <section className="card history-card" aria-labelledby="history-title">
          <h2 id="history-title">Prediction history and ground truth</h2>
          <p className="intro">
            Add a verified actual result once it is known. Saved ground truth is
            immutable here and is never copied from the model prediction.
          </p>
          <PredictionHistory refreshToken={historyRefreshToken} />
        </section>
      </div>
    </main>
  )
}

export default App
