import { FormEvent, useState } from 'react'

import { submitData } from '../../lib/api/data'
import {
  alertErrorClass,
  alertSuccessClass,
  inputClass,
  labelClass,
  primaryButtonClass,
} from '../../lib/ui'
import type { DataRead } from '../../types/data'

interface DataSubmissionFormProps {
  onDataCreated?: () => void
}

export function DataSubmissionForm({ onDataCreated }: DataSubmissionFormProps) {
  const [feature1, setFeature1] = useState('')
  const [feature2, setFeature2] = useState('')
  const [feature3, setFeature3] = useState('')
  const [label, setLabel] = useState('')
  const [response, setResponse] = useState<DataRead | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setIsSubmitting(true)
    setError(null)
    setResponse(null)

    try {
      const result = await submitData({
        feature_1: Number(feature1),
        feature_2: Number(feature2),
        feature_3: Number(feature3),
        ...(label === '' ? {} : { label: Number(label) }),
      })
      setResponse(result)
      onDataCreated?.()
    } catch (submissionError) {
      setError(
        submissionError instanceof Error
          ? submissionError.message
          : 'Unable to submit data.',
      )
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <>
      <form className="grid gap-4" onSubmit={handleSubmit}>
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="grid gap-2"><label className={labelClass} htmlFor="feature-1">Feature 1</label><input className={inputClass} id="feature-1" name="feature_1" type="number" step="any" required value={feature1} onChange={(event) => setFeature1(event.target.value)} placeholder="1.25" /></div>
          <div className="grid gap-2"><label className={labelClass} htmlFor="feature-2">Feature 2</label><input className={inputClass} id="feature-2" name="feature_2" type="number" step="any" required value={feature2} onChange={(event) => setFeature2(event.target.value)} placeholder="2.50" /></div>
          <div className="grid gap-2"><label className={labelClass} htmlFor="feature-3">Feature 3</label><input className={inputClass} id="feature-3" name="feature_3" type="number" step="any" required value={feature3} onChange={(event) => setFeature3(event.target.value)} placeholder="-3.75" /></div>
          <div className="grid gap-2"><label className={labelClass} htmlFor="label">Binary label <span className="normal-case tracking-normal text-slate-600">(optional)</span></label><input className={inputClass} id="label" name="label" type="number" step="1" min="0" max="1" value={label} onChange={(event) => setLabel(event.target.value)} placeholder="0 or 1" /></div>
        </div>

        <button className={`${primaryButtonClass} mt-2 w-full`} type="submit" disabled={isSubmitting}>
          {isSubmitting && <span className="size-4 animate-spin rounded-full border-2 border-slate-900/30 border-t-slate-950 motion-reduce:animate-none" />}
          {isSubmitting ? 'Submitting…' : 'Submit data'}
        </button>
      </form>

      {error && (
        <p className={alertErrorClass} role="alert">
          {error}
        </p>
      )}

      {response && (
        <section className={alertSuccessClass} aria-live="polite">
          <div className="flex items-center justify-between gap-3"><h4 className="font-bold text-white">Sample saved</h4><span className="rounded-full bg-emerald-300/15 px-2.5 py-1 text-xs font-bold">ID #{response.id}</span></div>
          <p className="mt-2 text-xs text-emerald-200/70">Created {new Date(response.created_at).toLocaleString()}</p>
          <pre className="mt-3 overflow-x-auto rounded-lg bg-slate-950/50 p-3 text-xs leading-5 text-slate-300">{JSON.stringify(response, null, 2)}</pre>
        </section>
      )}
    </>
  )
}
