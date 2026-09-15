import { useCallback, useEffect, useRef, useState } from 'react'

import { fetchTrainingStatus } from '../../lib/api/training'
import { isAbortError } from '../../lib/api/client'
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

export function ContinuousTrainingStatus({
  refreshToken = 0,
  onStatusChange,
}: ContinuousTrainingStatusProps) {
  const [status, setStatus] = useState<TrainingStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const activeRequest = useRef<AbortController | null>(null)
  const requestSequence = useRef(0)

  const loadStatus = useCallback(async (silent = false) => {
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
          : 'Unable to load continuous-training status.',
      )
    } finally {
      if (requestId === requestSequence.current) {
        setIsLoading(false)
        activeRequest.current = null
      }
    }
  }, [onStatusChange])

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

  return (
    <section className={panelClass} aria-labelledby="training-status-title">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex items-start gap-4">
          <span className="grid size-11 shrink-0 place-items-center rounded-xl border border-amber-300/20 bg-amber-300/10 text-amber-300"><svg aria-hidden="true" className="size-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" /></svg></span>
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.18em] text-amber-300">Training pipeline</p>
            <h2 id="training-status-title" className="mt-1 text-xl font-bold text-white">Continuous training</h2>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">Celery queue powered by verified ground truth. Active jobs refresh every five seconds.</p>
          </div>
        </div>
        <button
          className={`${secondaryButtonClass} w-full sm:w-auto`}
          type="button"
          onClick={() => void loadStatus()}
          disabled={isLoading}
        >
          {isLoading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>

      {error && <p className={alertErrorClass} role="alert">{error}</p>}
      {!error && status && (
        <dl className="mt-7 grid gap-3 sm:grid-cols-2 lg:grid-cols-4" aria-live="polite">
          <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
            <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Automatic retraining</dt>
            <dd className="mt-2 text-sm font-bold text-white">{status.auto_retrain_enabled ? 'Enabled' : 'Disabled'}</dd>
          </div>
          <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
            <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">New verified feedback</dt>
            <dd className="mt-2 text-sm font-bold text-white">{status.new_verified_samples} <span className="font-normal text-slate-500">/ {status.retrain_threshold}</span></dd>
            <p className="mt-1 text-xs text-slate-500">{status.new_verified_samples_needed} still needed for trigger</p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
            <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Verified dataset</dt>
            <dd className="mt-2 text-sm font-bold text-white">{status.verified_feedback_samples} <span className="font-normal text-slate-500">/ {status.minimum_training_samples}</span></dd>
            <p className="mt-1 text-xs text-slate-500">{status.verified_samples_needed_for_minimum} still needed for minimum</p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
            <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Training state</dt>
            <dd className="mt-2"><span className={`${statusBaseClass} ${statusTone(status.training_in_progress ? 'in progress' : 'idle')}`}>
                {status.training_in_progress ? 'In progress' : 'Idle'}
            </span></dd>
          </div>
          <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
            <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Infrastructure</dt>
            <dd className="mt-2 flex flex-wrap gap-1.5">
              <span className={`${statusBaseClass} ${statusTone(status.redis_available ? 'ready' : 'degraded')}`}>Redis {status.redis_available ? 'ready' : 'down'}</span>
              <span className={`${statusBaseClass} ${statusTone(status.celery_worker_available ? 'ready' : 'degraded')}`}>Worker {status.celery_worker_available ? 'online' : 'offline'}</span>
            </dd>
          </div>
          <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
            <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Queue</dt>
            <dd className="mt-2 text-sm font-bold text-white">{status.queued_jobs} queued <span className="text-slate-600">·</span> {status.running_jobs} running</dd>
          </div>
          <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
            <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Active model</dt>
            <dd className="mt-2 truncate font-mono text-sm font-bold text-white">{status.active_model_version ?? 'None'}</dd>
          </div>
          {status.current_training_run && (
            <div className="rounded-2xl border border-amber-300/20 bg-amber-300/5 p-4 sm:col-span-2">
              <dt className="text-[10px] font-bold uppercase tracking-wider text-amber-300/70">Current job</dt>
              <dd className="mt-2 text-sm font-semibold text-amber-100">
                Run #{status.current_training_run.id} · {status.current_training_run.model_version}
                {' · '}{status.current_training_run.progress_stage}
                {status.current_training_run.is_stale ? ' · stale' : ''}
                {status.current_training_run.dispatch_error
                  ? ` · ${status.current_training_run.dispatch_error}`
                  : ''}
              </dd>
            </div>
          )}
          <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
            <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Last run</dt>
            <dd className="mt-2 text-sm font-bold text-white">
              {status.last_training_run
                ? `${status.last_training_run.model_version} · ${status.last_training_run.status}`
                : 'No runs yet'}
            </dd>
          </div>
          <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
            <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Latest result</dt>
            <dd className="mt-2 text-sm font-bold text-white">
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
