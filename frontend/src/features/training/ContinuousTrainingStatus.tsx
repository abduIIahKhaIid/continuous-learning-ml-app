import { useCallback, useEffect, useRef, useState } from 'react'

import { isAbortError } from '../../lib/api/client'
import { fetchTrainingStatus } from '../../lib/api/training'
import {
  alertErrorClass,
  panelClass,
  secondaryButtonClass,
  statusBaseClass,
  statusTone,
} from '../../lib/ui'
import type { TrainingStatus } from '../../types/training'

interface ContinuousTrainingStatusProps {
  refreshToken?: number
  onStatusChange?: (status: TrainingStatus) => void
}

const stages = [
  ['queued', 'Queued'],
  ['preparing_data', 'Preparing data'],
  ['training', 'Training model'],
  ['evaluating', 'Evaluating'],
  ['promoting', 'Final decision'],
] as const

function stageLabel(stage: string): string {
  return (
    stages.find(([value]) => value === stage)?.[1] ??
    stage.replaceAll('_', ' ')
  )
}

function formatDate(value: string): string {
  return new Date(value).toLocaleString()
}

export function ContinuousTrainingStatus({
  refreshToken = 0,
  onStatusChange,
}: ContinuousTrainingStatusProps) {
  const [status, setStatus] = useState<TrainingStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const activeRequest = useRef<AbortController | null>(null)
  const requestSequence = useRef(0)

  const loadStatus = useCallback(
    async (silent = false) => {
      activeRequest.current?.abort()
      const controller = new AbortController()
      activeRequest.current = controller
      const requestId = ++requestSequence.current
      if (!silent) setIsLoading(true)
      setError(null)
      try {
        const nextStatus = await fetchTrainingStatus(controller.signal)
        setStatus(nextStatus)
        onStatusChange?.(nextStatus)
      } catch (requestError) {
        if (isAbortError(requestError)) return
        setError(
          requestError instanceof Error
            ? requestError.message
            : 'Unable to load training status.',
        )
      } finally {
        if (requestId === requestSequence.current) {
          setIsLoading(false)
          activeRequest.current = null
        }
      }
    },
    [onStatusChange],
  )

  useEffect(() => {
    void loadStatus()
    return () => activeRequest.current?.abort()
  }, [loadStatus, refreshToken])

  useEffect(() => {
    if (!status?.training_in_progress) return
    const timer = window.setInterval(() => {
      if (document.visibilityState === 'visible') void loadStatus(true)
    }, 5000)
    return () => window.clearInterval(timer)
  }, [loadStatus, status?.training_in_progress])

  const servicesReady = Boolean(
    status?.redis_available && status?.celery_worker_available,
  )
  const currentJob = status?.current_training_run
  const currentStageIndex = currentJob
    ? stages.findIndex(([value]) => value === currentJob.progress_stage)
    : -1
  const state = !status
    ? 'Loading'
    : !servicesReady
      ? 'Needs attention'
      : status.training_in_progress
        ? currentJob?.status === 'queued'
          ? 'Queued'
          : 'Training in progress'
        : 'Idle'
  const stateTone = !servicesReady
    ? 'degraded'
    : status?.training_in_progress
      ? 'in progress'
      : 'idle'

  return (
    <section className={panelClass} aria-labelledby="training-status-title">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex items-start gap-4">
          <span className="grid size-11 shrink-0 place-items-center rounded-xl border border-amber-300/20 bg-amber-300/10 text-amber-300">
            <svg
              aria-hidden="true"
              className="size-5"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              viewBox="0 0 24 24"
            >
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
            </svg>
          </span>
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.18em] text-amber-300">
              Training operations
            </p>
            <h2
              className="mt-1 text-xl font-bold text-white"
              id="training-status-title"
            >
              Pipeline status
            </h2>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
              Worker availability, queue state, and the current training stage.
            </p>
          </div>
        </div>
        <button
          className={`${secondaryButtonClass} w-full sm:w-auto`}
          disabled={isLoading}
          onClick={() => void loadStatus()}
          type="button"
        >
          {isLoading ? 'Refreshing…' : 'Refresh status'}
        </button>
      </div>

      {error && <p className={alertErrorClass} role="alert">{error}</p>}

      {status && (
        <div aria-live="polite">
          <div className={`mt-7 rounded-2xl border p-5 ${status.training_in_progress ? 'border-amber-300/20 bg-amber-300/5' : servicesReady ? 'border-emerald-300/20 bg-emerald-300/5' : 'border-rose-300/20 bg-rose-300/5'}`}>
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="text-[10px] font-bold uppercase tracking-[0.16em] text-slate-500">
                  Current state
                </p>
                <p className="mt-1 text-lg font-bold text-white">
                  {state}
                </p>
                <p className="mt-1 text-xs leading-5 text-slate-400">
                  {currentJob
                    ? `Run #${currentJob.id} · ${currentJob.model_version} · ${stageLabel(currentJob.progress_stage)}`
                    : servicesReady
                      ? 'No training job is running. Data readiness is shown in the overview above.'
                      : 'Start Redis and the Celery worker before automatic training can run.'}
                </p>
              </div>
              <span className={`${statusBaseClass} ${statusTone(stateTone)}`}>
                {state}
              </span>
            </div>

            {currentJob && (
              <ol className="mt-5 grid gap-2 sm:grid-cols-5" aria-label="Training progress">
                {stages.map(([stage, label], index) => {
                  const isComplete = currentStageIndex > index
                  const isCurrent = currentStageIndex === index
                  return (
                    <li
                      aria-current={isCurrent ? 'step' : undefined}
                      className={`rounded-xl border px-3 py-2 text-xs font-semibold ${isCurrent ? 'border-amber-300/30 bg-amber-300/10 text-amber-200' : isComplete ? 'border-emerald-300/20 bg-emerald-300/5 text-emerald-300' : 'border-white/5 bg-slate-950/30 text-slate-600'}`}
                      key={stage}
                    >
                      <span className="mr-1.5">{isComplete ? '✓' : index + 1}</span>
                      {label}
                    </li>
                  )
                })}
              </ol>
            )}
          </div>

          <dl className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
              <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Automatic mode</dt>
              <dd className="mt-2 text-sm font-bold text-white">{status.auto_retrain_enabled ? 'Enabled' : 'Disabled'}</dd>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
              <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Queue</dt>
              <dd className="mt-2 text-sm font-bold text-white">{status.queued_jobs} queued <span className="text-slate-600">·</span> {status.running_jobs} running</dd>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
              <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Services</dt>
              <dd className="mt-2 flex flex-wrap gap-1.5">
                <span className={`${statusBaseClass} ${statusTone(status.redis_available ? 'ready' : 'degraded')}`}>Redis {status.redis_available ? 'ready' : 'down'}</span>
                <span className={`${statusBaseClass} ${statusTone(status.celery_worker_available ? 'ready' : 'degraded')}`}>Worker {status.celery_worker_available ? 'online' : 'offline'}</span>
              </dd>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
              <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Most recent run</dt>
              <dd className="mt-2 text-sm font-bold text-white">{status.last_training_run ? `${status.last_training_run.model_version} · ${status.last_training_run.status}` : 'No runs yet'}</dd>
              {status.last_training_run && <p className="mt-1 text-[11px] text-slate-500">{formatDate(status.last_training_run.created_at)}</p>}
            </div>
          </dl>

          {currentJob?.dispatch_error && (
            <p className={alertErrorClass} role="alert">
              Dispatch problem: {currentJob.dispatch_error}
            </p>
          )}
          {currentJob?.is_stale && (
            <p className={alertErrorClass} role="alert">
              This job has stopped reporting progress. Check the Celery worker logs.
            </p>
          )}
        </div>
      )}
    </section>
  )
}
