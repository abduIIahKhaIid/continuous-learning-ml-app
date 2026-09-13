import { useCallback, useEffect, useState } from 'react'

import { fetchTrainingStatus } from '../../lib/api/training'
import type { TrainingStatus } from '../../types/training'

export function ContinuousTrainingStatus() {
  const [status, setStatus] = useState<TrainingStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const loadStatus = useCallback(async (signal?: AbortSignal) => {
    setIsLoading(true)
    setError(null)
    try {
      setStatus(await fetchTrainingStatus(signal))
    } catch (requestError) {
      if (
        requestError instanceof DOMException &&
        requestError.name === 'AbortError'
      ) {
        return
      }
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'Unable to load continuous-training status.',
      )
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    void loadStatus(controller.signal)
    return () => controller.abort()
  }, [loadStatus])

  return (
    <section className="card training-status-card" aria-labelledby="training-status-title">
      <div className="section-heading">
        <div>
          <h2 id="training-status-title">Continuous training</h2>
          <p className="intro">
            Threshold-based full retraining using verified ground truth.
          </p>
        </div>
        <button
          className="secondary-button"
          type="button"
          onClick={() => void loadStatus()}
          disabled={isLoading}
        >
          {isLoading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>

      {error && <p className="message error" role="alert">{error}</p>}
      {!error && status && (
        <dl className="training-status-grid" aria-live="polite">
          <div>
            <dt>Automatic retraining</dt>
            <dd>{status.auto_retrain_enabled ? 'Enabled' : 'Disabled'}</dd>
          </div>
          <div>
            <dt>New verified samples</dt>
            <dd>{status.new_verified_samples} / {status.retrain_threshold}</dd>
          </div>
          <div>
            <dt>Training in progress</dt>
            <dd>{status.training_in_progress ? 'Yes' : 'No'}</dd>
          </div>
          <div>
            <dt>Active model</dt>
            <dd>{status.active_model_version ?? 'None'}</dd>
          </div>
          <div>
            <dt>Last run</dt>
            <dd>
              {status.last_training_run
                ? `${status.last_training_run.model_version} · ${status.last_training_run.status}`
                : 'No runs yet'}
            </dd>
          </div>
        </dl>
      )}
    </section>
  )
}
