import { FormEvent, useState } from 'react'

import { submitData } from '../../lib/api/data'
import type { DataRead } from '../../types/data'

export function DataSubmissionForm() {
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
      <form className="data-form" onSubmit={handleSubmit}>
        <label htmlFor="feature-1">Feature 1</label>
        <input
          id="feature-1"
          name="feature_1"
          type="number"
          step="any"
          required
          value={feature1}
          onChange={(event) => setFeature1(event.target.value)}
          placeholder="1.25"
        />

        <label htmlFor="feature-2">Feature 2</label>
        <input
          id="feature-2"
          name="feature_2"
          type="number"
          step="any"
          required
          value={feature2}
          onChange={(event) => setFeature2(event.target.value)}
          placeholder="2.5"
        />

        <label htmlFor="feature-3">Feature 3</label>
        <input
          id="feature-3"
          name="feature_3"
          type="number"
          step="any"
          required
          value={feature3}
          onChange={(event) => setFeature3(event.target.value)}
          placeholder="-3.75"
        />

        <label htmlFor="label">Label (optional integer)</label>
        <input
          id="label"
          name="label"
          type="number"
          step="1"
          min="0"
          max="1"
          value={label}
          onChange={(event) => setLabel(event.target.value)}
          placeholder="1"
        />

        <button type="submit" disabled={isSubmitting}>
          {isSubmitting ? 'Submitting…' : 'Submit data'}
        </button>
      </form>

      {error && (
        <p className="message error" role="alert">
          {error}
        </p>
      )}

      {response && (
        <section className="response" aria-live="polite">
          <h2>Saved record</h2>
          <p>
            <strong>Record ID:</strong> {response.id}
          </p>
          <p>
            <strong>Created:</strong>{' '}
            {new Date(response.created_at).toLocaleString()}
          </p>
          <pre>{JSON.stringify(response, null, 2)}</pre>
        </section>
      )}
    </>
  )
}
