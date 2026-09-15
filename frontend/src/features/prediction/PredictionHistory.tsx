import { FormEvent, useEffect, useState } from 'react'

import {
  fetchFeedbackSummary,
  fetchPredictions,
  submitPredictionFeedback,
} from '../../lib/api/predictions'
import { isAbortError } from '../../lib/api/client'
import {
  alertErrorClass,
  inputClass,
  primaryButtonClass,
} from '../../lib/ui'
import type {
  PredictionFeedbackSummary,
  PredictionHistoryItem,
} from '../../types/prediction'

interface PredictionHistoryProps {
  refreshToken: number
  onFeedbackSubmitted?: () => void
  onSummaryChange?: (summary: PredictionFeedbackSummary) => void
}

type LabelChoice = '' | '0' | '1'
const PAGE_SIZE = 20

export function PredictionHistory({
  refreshToken,
  onFeedbackSubmitted,
  onSummaryChange,
}: PredictionHistoryProps) {
  const [predictions, setPredictions] = useState<PredictionHistoryItem[]>([])
  const [summary, setSummary] = useState<PredictionFeedbackSummary | null>(null)
  const [choices, setChoices] = useState<Record<number, LabelChoice>>({})
  const [rowErrors, setRowErrors] = useState<Record<number, string>>({})
  const [submittingId, setSubmittingId] = useState<number | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [loadMoreError, setLoadMoreError] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    let isCurrent = true
    setIsLoading(true)
    setLoadError(null)
    setLoadMoreError(null)
    Promise.all([
      fetchPredictions(controller.signal),
      fetchFeedbackSummary(controller.signal),
    ])
      .then(([history, feedbackSummary]) => {
        if (!isCurrent) return
        setPredictions(history)
        setSummary(feedbackSummary)
        onSummaryChange?.(feedbackSummary)
      })
      .catch((error: unknown) => {
        if (!isCurrent || isAbortError(error)) return
        setLoadError(
          error instanceof Error
            ? error.message
            : 'Unable to load prediction history.',
        )
      })
      .finally(() => {
        if (isCurrent) setIsLoading(false)
      })

    return () => {
      isCurrent = false
      controller.abort()
    }
  }, [onSummaryChange, refreshToken])

  async function loadMore() {
    setIsLoadingMore(true)
    setLoadMoreError(null)
    try {
      const nextPage = await fetchPredictions(
        undefined,
        predictions.length,
        PAGE_SIZE,
      )
      setPredictions((current) => {
        const existingIds = new Set(current.map((item) => item.id))
        return [
          ...current,
          ...nextPage.filter((item) => !existingIds.has(item.id)),
        ]
      })
    } catch (error) {
      if (isAbortError(error)) return
      setLoadMoreError(
        error instanceof Error
          ? error.message
          : 'Unable to load more predictions.',
      )
    } finally {
      setIsLoadingMore(false)
    }
  }

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
      const feedbackSummary = await fetchFeedbackSummary()
      setSummary(feedbackSummary)
      onSummaryChange?.(feedbackSummary)
      onFeedbackSubmitted?.()
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
    return <div className="grid gap-3" aria-live="polite"><div className="h-24 animate-pulse rounded-2xl bg-slate-800/70 motion-reduce:animate-none" /><span className="sr-only">Loading predictions…</span></div>
  }

  if (loadError) {
    return <p className={alertErrorClass} role="alert">{loadError}</p>
  }

  return (
    <>
      {summary && (
        <div className="mb-6 grid gap-3 sm:grid-cols-3" aria-label="Verified feedback summary">
          <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4"><span className="text-xs font-semibold uppercase tracking-wider text-slate-500">Verified</span><strong className="mt-1 block text-2xl text-white">{summary.feedback_received}</strong></div>
          <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4"><span className="text-xs font-semibold uppercase tracking-wider text-slate-500">Pending</span><strong className="mt-1 block text-2xl text-white">{summary.feedback_pending}</strong></div>
          <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4"><span className="text-xs font-semibold uppercase tracking-wider text-slate-500">Accuracy</span><strong className="mt-1 block text-2xl text-white">
            {summary.verified_accuracy === null
              ? 'Not available'
              : `${(summary.verified_accuracy * 100).toFixed(1)}%`}
          </strong></div>
        </div>
      )}

      {predictions.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-slate-700 bg-slate-950/30 px-6 py-12 text-center"><span className="mx-auto grid size-11 place-items-center rounded-full bg-slate-800 text-slate-500"><svg aria-hidden="true" className="size-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 12h6M12 9v6" /><circle cx="12" cy="12" r="9" /></svg></span><p className="mt-3 text-sm font-semibold text-slate-300">No predictions stored yet</p><p className="mt-1 text-xs text-slate-500">New predictions will appear here.</p></div>
      ) : (
        <div className="grid gap-3">
          {predictions.map((prediction) => {
            const wasCorrect =
              prediction.actual_label === null
                ? null
                : prediction.predicted_class === prediction.actual_label
            return (
              <article className="rounded-2xl border border-white/10 bg-slate-950/40 p-4 transition hover:border-slate-700 sm:p-5" key={prediction.id}>
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <strong className="text-sm text-white">Prediction #{prediction.id}</strong>
                  <span className="rounded-full bg-slate-800 px-2.5 py-1 font-mono text-[10px] font-semibold text-slate-400">{prediction.model_version}</span>
                </div>
                <p className="mt-3 text-sm text-slate-400">
                  Prediction: <strong>{prediction.predicted_class}</strong>
                  {prediction.prediction_probability === null
                    ? ''
                    : ` · Probability: ${(prediction.prediction_probability * 100).toFixed(1)}%`}
                </p>

                {prediction.feedback_received ? (
                  <p className="mt-3 rounded-xl border border-emerald-400/20 bg-emerald-400/10 px-3 py-2.5 text-sm text-emerald-200" aria-live="polite">
                    Actual result: <strong>{prediction.actual_label}</strong> ·
                    Correct: <strong>{wasCorrect ? 'Yes' : 'No'}</strong>
                  </p>
                ) : (
                  <form
                    className="mt-4 grid items-end gap-3 sm:grid-cols-[minmax(9rem,1fr)_minmax(7rem,.5fr)_auto]"
                    onSubmit={(event) => handleFeedback(event, prediction)}
                  >
                    <label className="text-xs font-semibold text-slate-400 sm:col-span-3" htmlFor={`actual-label-${prediction.id}`}>
                      Provide Actual Result
                    </label>
                    <select
                      className={inputClass}
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
                      className={primaryButtonClass}
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
                  <p className="mt-2 text-xs font-medium text-rose-300" role="alert">
                    {rowErrors[prediction.id]}
                  </p>
                )}
              </article>
            )
          })}
          {summary && predictions.length < summary.total_predictions && (
            <div className="flex flex-col items-center gap-3 pt-3">
              <p className="text-xs text-slate-500">
                Showing {predictions.length} of {summary.total_predictions}{' '}
                predictions
              </p>
              <button
                className="rounded-xl border border-slate-700 bg-slate-800/70 px-5 py-2.5 text-sm font-semibold text-slate-200 transition hover:border-slate-600 hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
                disabled={isLoadingMore}
                onClick={() => void loadMore()}
                type="button"
              >
                {isLoadingMore ? 'Loading…' : 'Load more predictions'}
              </button>
            </div>
          )}
          {loadMoreError && (
            <p className="text-center text-xs font-medium text-rose-300" role="alert">
              {loadMoreError}
            </p>
          )}
        </div>
      )}
    </>
  )
}
