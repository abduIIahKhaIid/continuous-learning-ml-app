import { FormEvent, useState } from 'react'

import { submitPrediction } from '../../lib/api/predictions'
import {
  alertErrorClass,
  alertSuccessClass,
  inputClass,
  labelClass,
  primaryButtonClass,
} from '../../lib/ui'
import type { PredictionResponse } from '../../types/prediction'

interface PredictionFormProps {
  onPredictionCreated?: () => void
}

export function PredictionForm({
  onPredictionCreated,
}: PredictionFormProps) {
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
      onPredictionCreated?.()
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
      <form className="grid gap-4" onSubmit={handleSubmit}>
        <div className="grid gap-4 sm:grid-cols-3">
          <div className="grid gap-2"><label className={labelClass} htmlFor="prediction-feature-1">Feature 1</label><input className={inputClass} id="prediction-feature-1" name="feature_1" type="number" step="any" required value={feature1} onChange={(event) => setFeature1(event.target.value)} placeholder="2.50" /></div>
          <div className="grid gap-2"><label className={labelClass} htmlFor="prediction-feature-2">Feature 2</label><input className={inputClass} id="prediction-feature-2" name="feature_2" type="number" step="any" required value={feature2} onChange={(event) => setFeature2(event.target.value)} placeholder="4.10" /></div>
          <div className="grid gap-2"><label className={labelClass} htmlFor="prediction-feature-3">Feature 3</label><input className={inputClass} id="prediction-feature-3" name="feature_3" type="number" step="any" required value={feature3} onChange={(event) => setFeature3(event.target.value)} placeholder="6.70" /></div>
        </div>

        <button className={`${primaryButtonClass} mt-2 w-full`} type="submit" disabled={isSubmitting}>
          {isSubmitting && <span className="size-4 animate-spin rounded-full border-2 border-slate-900/30 border-t-slate-950 motion-reduce:animate-none" />}
          {isSubmitting ? 'Predicting…' : 'Make prediction'}
        </button>
      </form>

      {error && (
        <p className={alertErrorClass} role="alert">
          {error}
        </p>
      )}

      {result && (
        <section className={alertSuccessClass} aria-live="polite">
          <div className="mb-3 flex items-center justify-between"><h4 className="font-bold text-white">Prediction complete</h4><span className="rounded-full bg-emerald-300/15 px-2.5 py-1 text-xs font-bold">ID #{result.prediction_id}</span></div>
          <dl className="grid grid-cols-2 gap-2">
            <div className="rounded-lg bg-slate-950/40 p-3">
              <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Predicted class</dt>
              <dd className="mt-1 text-lg font-black text-white">{result.predicted_class}</dd>
            </div>
            <div className="rounded-lg bg-slate-950/40 p-3">
              <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Probability</dt>
              <dd className="mt-1 font-bold text-white">
                {result.probability === null
                  ? 'Not available'
                  : `${(result.probability * 100).toFixed(2)}%`}
              </dd>
            </div>
            <div className="col-span-2 rounded-lg bg-slate-950/40 p-3">
              <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Model version</dt>
              <dd className="mt-1 font-mono text-sm font-bold text-white">{result.model_version}</dd>
            </div>
          </dl>
        </section>
      )}
    </>
  )
}
