import { FormEvent, useEffect, useState } from 'react'

import {
  fetchFeedbackSummary,
  fetchPredictions,
  submitPredictionFeedback,
} from '../../lib/api/predictions'
import type {
  PredictionFeedbackSummary,
  PredictionHistoryItem,
} from '../../types/prediction'

interface PredictionHistoryProps {
  refreshToken: number
}

type LabelChoice = '' | '0' | '1'

export function PredictionHistory({ refreshToken }: PredictionHistoryProps) {
  const [predictions, setPredictions] = useState<PredictionHistoryItem[]>([])
  const [summary, setSummary] = useState<PredictionFeedbackSummary | null>(null)
  const [choices, setChoices] = useState<Record<number, LabelChoice>>({})
  const [rowErrors, setRowErrors] = useState<Record<number, string>>({})
  const [submittingId, setSubmittingId] = useState<number | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    setIsLoading(true)
    setLoadError(null)
    Promise.all([
      fetchPredictions(controller.signal),
      fetchFeedbackSummary(controller.signal),
    ])
      .then(([history, feedbackSummary]) => {
        setPredictions(history)
        setSummary(feedbackSummary)
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === 'AbortError') {
          return
        }
        setLoadError(
          error instanceof Error
            ? error.message
            : 'Unable to load prediction history.',
        )
      })
      .finally(() => setIsLoading(false))

    return () => controller.abort()
  }, [refreshToken])

  async function handleFeedback(
    event: FormEvent<HTMLFormElement>,
    prediction: PredictionHistoryItem,
  ) {
    event.preventDefault()
    const choice = choices[prediction.id] ?? ''
    if (choice !== '0' && choice !== '1') {
      setRowErrors((current) => ({
        ...current,
        [prediction.id]: 'Select actual result 0 or 1.',
      }))
      return
    }

    setSubmittingId(prediction.id)
    setRowErrors((current) => ({ ...current, [prediction.id]: '' }))
    try {
      const saved = await submitPredictionFeedback(prediction.id, {
        actual_label: Number(choice) as 0 | 1,
      })
      setPredictions((current) =>
        current.map((item) =>
          item.id === prediction.id
            ? {
                ...item,
                actual_label: saved.actual_label,
                feedback_received: saved.feedback_received,
                updated_at: saved.updated_at,
              }
            : item,
        ),
      )
      setSummary(await fetchFeedbackSummary())
    } catch (error) {
      setRowErrors((current) => ({
        ...current,
        [prediction.id]:
          error instanceof Error
            ? error.message
            : 'Unable to submit actual result.',
      }))
    } finally {
      setSubmittingId(null)
    }
  }

  if (isLoading) {
    return <p className="message" aria-live="polite">Loading predictions…</p>
  }

  if (loadError) {
    return <p className="message error" role="alert">{loadError}</p>
  }

  return (
    <>
      {summary && (
        <div className="feedback-summary" aria-label="Verified feedback summary">
          <span>{summary.feedback_received} verified</span>
          <span>{summary.feedback_pending} pending</span>
          <span>
            Accuracy:{' '}
            {summary.verified_accuracy === null
              ? 'Not available'
              : `${(summary.verified_accuracy * 100).toFixed(1)}%`}
          </span>
        </div>
      )}

      {predictions.length === 0 ? (
        <p className="message">No predictions have been stored yet.</p>
      ) : (
        <div className="prediction-list">
          {predictions.map((prediction) => {
            const wasCorrect =
              prediction.actual_label === null
                ? null
                : prediction.predicted_class === prediction.actual_label
            return (
              <article className="prediction-row" key={prediction.id}>
                <div className="prediction-row-heading">
                  <strong>Prediction #{prediction.id}</strong>
                  <span>{prediction.model_version}</span>
                </div>
                <p>
                  Prediction: <strong>{prediction.predicted_class}</strong>
                  {prediction.prediction_probability === null
                    ? ''
                    : ` · Probability: ${(prediction.prediction_probability * 100).toFixed(1)}%`}
                </p>

                {prediction.feedback_received ? (
                  <p className="verified-result" aria-live="polite">
                    Actual result: <strong>{prediction.actual_label}</strong> ·
                    Correct: <strong>{wasCorrect ? 'Yes' : 'No'}</strong>
                  </p>
                ) : (
                  <form
                    className="feedback-form"
                    onSubmit={(event) => handleFeedback(event, prediction)}
                  >
                    <label htmlFor={`actual-label-${prediction.id}`}>
                      Provide Actual Result
                    </label>
                    <select
                      id={`actual-label-${prediction.id}`}
                      value={choices[prediction.id] ?? ''}
                      onChange={(event) =>
                        setChoices((current) => ({
                          ...current,
                          [prediction.id]: event.target.value as LabelChoice,
                        }))
                      }
                      disabled={submittingId === prediction.id}
                    >
                      <option value="">Select 0 or 1</option>
                      <option value="0">0</option>
                      <option value="1">1</option>
                    </select>
                    <button
                      type="submit"
                      disabled={submittingId === prediction.id}
                    >
                      {submittingId === prediction.id
                        ? 'Submitting…'
                        : 'Save ground truth'}
                    </button>
                  </form>
                )}

                {rowErrors[prediction.id] && (
                  <p className="row-error" role="alert">
                    {rowErrors[prediction.id]}
                  </p>
                )}
              </article>
            )
          })}
        </div>
      )}
    </>
  )
}
