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

  useEffect(() => {
    if (!status?.training_in_progress) return
    const timer = window.setInterval(() => void loadStatus(), 5000)
    return () => window.clearInterval(timer)
  }, [loadStatus, status?.training_in_progress])

  return (
    <section className="card training-status-card" aria-labelledby="training-status-title">
      <div className="section-heading">
        <div>
          <h2 id="training-status-title">Continuous training</h2>
          <p className="intro">
            Celery training queue using verified ground truth. Active jobs refresh every five seconds.
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
            <dt>Training infrastructure</dt>
            <dd>
              {status.redis_available && status.celery_worker_available
                ? 'Ready'
                : 'Degraded'}
            </dd>
          </div>
          <div>
            <dt>Queue</dt>
            <dd>{status.queued_jobs} queued · {status.running_jobs} running</dd>
          </div>
          <div>
            <dt>Active model</dt>
            <dd>{status.active_model_version ?? 'None'}</dd>
          </div>
          {status.current_training_run && (
            <div>
              <dt>Current job</dt>
              <dd>
                Run #{status.current_training_run.id} · {status.current_training_run.model_version}
                {' · '}{status.current_training_run.progress_stage}
                {status.current_training_run.is_stale ? ' · stale' : ''}
                {status.current_training_run.dispatch_error
                  ? ` · ${status.current_training_run.dispatch_error}`
                  : ''}
              </dd>
            </div>
          )}
          <div>
            <dt>Last run</dt>
            <dd>
              {status.last_training_run
                ? `${status.last_training_run.model_version} · ${status.last_training_run.status}`
                : 'No runs yet'}
            </dd>
          </div>
          <div>
            <dt>Latest result</dt>
            <dd>
              {status.latest_completed_run
                ? `${status.latest_completed_run.model_version} · ${status.latest_completed_run.status}`
                : 'No completed jobs yet'}
            </dd>
          </div>
        </dl>
      )}
    </section>
  )
}
