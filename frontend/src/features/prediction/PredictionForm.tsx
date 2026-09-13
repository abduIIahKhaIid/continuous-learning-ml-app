import { FormEvent, useState } from 'react'

import { submitPrediction } from '../../lib/api/predictions'
import type { PredictionResponse } from '../../types/prediction'

export function PredictionForm() {
  const [feature1, setFeature1] = useState('')
  const [feature2, setFeature2] = useState('')
  const [feature3, setFeature3] = useState('')
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const features = [Number(feature1), Number(feature2), Number(feature3)]
    if (!features.every(Number.isFinite)) {
      setError('All three features must be finite numbers.')
      setResult(null)
      return
    }

    setIsSubmitting(true)
    setError(null)
    setResult(null)
    try {
      const prediction = await submitPrediction({
        feature_1: features[0],
        feature_2: features[1],
        feature_3: features[2],
      })
      setResult(prediction)
    } catch (submissionError) {
      setError(
        submissionError instanceof Error
          ? submissionError.message
          : 'Unable to make a prediction.',
      )
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <>
      <form className="data-form" onSubmit={handleSubmit}>
        <label htmlFor="prediction-feature-1">Feature 1</label>
        <input
          id="prediction-feature-1"
          name="feature_1"
          type="number"
          step="any"
          required
          value={feature1}
          onChange={(event) => setFeature1(event.target.value)}
          placeholder="2.5"
        />

        <label htmlFor="prediction-feature-2">Feature 2</label>
        <input
          id="prediction-feature-2"
          name="feature_2"
          type="number"
          step="any"
          required
          value={feature2}
          onChange={(event) => setFeature2(event.target.value)}
          placeholder="4.1"
        />

        <label htmlFor="prediction-feature-3">Feature 3</label>
        <input
          id="prediction-feature-3"
          name="feature_3"
          type="number"
          step="any"
          required
          value={feature3}
          onChange={(event) => setFeature3(event.target.value)}
          placeholder="6.7"
        />

        <button type="submit" disabled={isSubmitting}>
          {isSubmitting ? 'Predicting…' : 'Make prediction'}
        </button>
      </form>

      {error && (
        <p className="message error" role="alert">
          {error}
        </p>
      )}

      {result && (
        <section className="response prediction-result" aria-live="polite">
          <h2>Prediction result</h2>
          <dl>
            <div>
              <dt>Predicted class</dt>
              <dd>{result.predicted_class}</dd>
            </div>
            <div>
              <dt>Class 1 probability</dt>
              <dd>
                {result.probability === null
                  ? 'Not available'
                  : `${(result.probability * 100).toFixed(2)}%`}
              </dd>
            </div>
            <div>
              <dt>Model version</dt>
              <dd>{result.model_version}</dd>
            </div>
            <div>
              <dt>Prediction ID</dt>
              <dd>{result.prediction_id}</dd>
            </div>
          </dl>
        </section>
      )}
    </>
  )
}
