import { useCallback, useEffect, useState } from 'react'

import {
  fetchDriftReport,
  fetchModels,
  fetchMonitoringHealth,
  rollbackModel,
  runMonitoringCheck,
} from '../../lib/api/monitoring'
import { isAbortError } from '../../lib/api/client'
import {
  alertErrorClass,
  alertSuccessClass,
  panelClass,
  secondaryButtonClass,
  statusBaseClass,
  statusTone,
} from '../../lib/ui'
import type {
  DriftReport,
  ModelSummary,
  MonitoringHealth,
} from '../../types/monitoring'

interface ModelMonitoringProps {
  onActiveModelChanged: () => void
}

function formatMetric(value: number | null): string {
  return value === null ? '—' : value.toFixed(3)
}

function statusLabel(value: string): string {
  return value.replaceAll('_', ' ')
}

export function ModelMonitoring({ onActiveModelChanged }: ModelMonitoringProps) {
  const [health, setHealth] = useState<MonitoringHealth | null>(null)
  const [drift, setDrift] = useState<DriftReport | null>(null)
  const [models, setModels] = useState<ModelSummary[]>([])
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isWorking, setIsWorking] = useState(false)

  const load = useCallback(async (signal?: AbortSignal) => {
    setError(null)
    try {
      const [healthResult, driftResult, modelResult] = await Promise.all([
        fetchMonitoringHealth(signal),
        fetchDriftReport(signal),
        fetchModels(signal),
      ])
      setHealth(healthResult)
      setDrift(driftResult)
      setModels(modelResult)
    } catch (requestError) {
      if (isAbortError(requestError)) return
      setError(requestError instanceof Error ? requestError.message : 'Unable to load monitoring.')
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    void load(controller.signal)
    return () => controller.abort()
  }, [load])

  const persistCheck = async () => {
    setIsWorking(true)
    setMessage(null)
    setError(null)
    try {
      await runMonitoringCheck()
      setMessage('Monitoring snapshot saved.')
      await load()
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Monitoring check failed.')
    } finally {
      setIsWorking(false)
    }
  }

  const rollback = async (modelVersion: string) => {
    if (!window.confirm(`Activate ${modelVersion}? This changes the model used by new predictions.`)) return
    const reason = window.prompt('Optional rollback reason:', '')
    if (reason === null) return
    setIsWorking(true)
    setMessage(null)
    setError(null)
    try {
      const result = await rollbackModel(modelVersion, reason)
      setMessage(result.message)
      onActiveModelChanged()
      await load()
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Rollback failed.')
    } finally {
      setIsWorking(false)
    }
  }

  const metricCard = 'rounded-2xl border border-white/10 bg-slate-950/45 p-4'
  const metricLabel = 'text-[10px] font-bold uppercase tracking-wider text-slate-500'
  const metricValue = 'mt-2 text-sm font-bold text-white'
  const tableHeading = 'bg-slate-950/70 px-4 py-3 text-left text-[10px] font-bold uppercase tracking-wider text-slate-500'
  const tableCell = 'border-t border-white/5 px-4 py-3 text-sm text-slate-300'

  return (
    <section className={panelClass} aria-labelledby="monitoring-title">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex items-start gap-4">
          <span className="grid size-11 shrink-0 place-items-center rounded-xl border border-cyan-300/20 bg-cyan-300/10 text-cyan-300"><svg aria-hidden="true" className="size-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 3v18h18" /><path d="M7 15l4-4 3 3 5-7" /></svg></span>
          <div><p className="text-xs font-bold uppercase tracking-[0.18em] text-cyan-300">Observability</p><h2 id="monitoring-title" className="mt-1 text-xl font-bold text-white">Model monitoring</h2><p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">Live feature drift and verified performance health for the active model.</p></div>
        </div>
        <button className={`${secondaryButtonClass} w-full sm:w-auto`} disabled={isWorking} onClick={persistCheck} type="button">{isWorking ? 'Working…' : 'Save monitoring check'}</button>
      </div>

      {isLoading && !health && <div className="mt-7 h-28 animate-pulse rounded-2xl bg-slate-800/70 motion-reduce:animate-none" role="status"><span className="sr-only">Loading monitoring</span></div>}
      {error && <p className={alertErrorClass} role="alert">{error}</p>}
      {message && <p className={alertSuccessClass} role="status">{message}</p>}

      {health && (
        <dl className="mt-7 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className={metricCard}><dt className={metricLabel}>Overall health</dt><dd className="mt-2"><span className={`${statusBaseClass} ${statusTone(health.status)}`}>{statusLabel(health.status)}</span></dd></div>
          <div className={metricCard}><dt className={metricLabel}>Active model</dt><dd className={`${metricValue} truncate font-mono`}>{health.model_version ?? 'None'}</dd></div>
          <div className={metricCard}><dt className={metricLabel}>Data drift</dt><dd className="mt-2"><span className={`${statusBaseClass} ${statusTone(health.data_drift_status)}`}>{statusLabel(health.data_drift_status)}</span></dd></div>
          <div className={metricCard}><dt className={metricLabel}>Performance</dt><dd className="mt-2"><span className={`${statusBaseClass} ${statusTone(health.performance_status)}`}>{statusLabel(health.performance_status)}</span></dd></div>
          <div className={metricCard}><dt className={metricLabel}>Feedback coverage</dt><dd className={metricValue}>{(health.feedback_coverage * 100).toFixed(1)}%</dd></div>
          <div className={metricCard}><dt className={metricLabel}>Verified samples</dt><dd className={metricValue}>{health.new_verified_samples} <span className="font-normal text-slate-500">/ {health.retrain_threshold}</span></dd></div>
          <div className={metricCard}><dt className={metricLabel}>Last training</dt><dd className={metricValue}>{health.last_training_status ?? 'None'}</dd></div>
        </dl>
      )}

      <div className="mt-8"><div className="mb-3 flex items-center justify-between"><h3 className="font-bold text-white">Feature drift</h3><span className="text-xs text-slate-500">Population Stability Index</span></div>
        {drift && drift.features.length > 0 ? (
          <div className="overflow-x-auto rounded-2xl border border-white/10 focus:outline-none focus:ring-4 focus:ring-cyan-400/10" tabIndex={0} aria-label="Feature drift table">
            <table className="w-full min-w-[36rem] border-collapse"><thead><tr><th className={tableHeading}>Feature</th><th className={tableHeading}>PSI</th><th className={tableHeading}>Status</th><th className={tableHeading}>Current / reference</th></tr></thead>
              <tbody>{drift.features.map((feature) => <tr className="transition hover:bg-white/[0.025]" key={feature.feature_name}><td className={`${tableCell} font-mono font-semibold text-white`}>{feature.feature_name}</td><td className={tableCell}>{formatMetric(feature.psi)}</td><td className={tableCell}><span className={`${statusBaseClass} ${statusTone(feature.status)}`}>{statusLabel(feature.status)}</span></td><td className={tableCell}>{feature.current_samples} / {feature.reference_samples}</td></tr>)}</tbody>
            </table>
          </div>
        ) : !isLoading && <p className="rounded-2xl border border-dashed border-slate-700 px-5 py-8 text-center text-sm text-slate-500">No active model profile is available yet.</p>}
      </div>

      <div className="mt-8"><h3 className="font-bold text-white">Model registry</h3><p className="mt-1 text-sm text-slate-500">Switch to an earlier validated artifact without retraining.</p>
        <div className="mt-3 overflow-x-auto rounded-2xl border border-white/10 focus:outline-none focus:ring-4 focus:ring-cyan-400/10" tabIndex={0} aria-label="Model history table">
          <table className="w-full min-w-[54rem] border-collapse"><thead><tr><th className={tableHeading}>Version</th><th className={tableHeading}>Status</th><th className={tableHeading}>Active</th><th className={tableHeading}>F1</th><th className={tableHeading}>Samples</th><th className={tableHeading}>Created</th><th className={tableHeading}>Action</th></tr></thead>
            <tbody>{models.map((model) => <tr className="transition hover:bg-white/[0.025]" key={model.model_version}><td className={`${tableCell} font-mono font-semibold text-white`}>{model.model_version}</td><td className={tableCell}><span className={`${statusBaseClass} ${statusTone(model.status)}`}>{statusLabel(model.status)}</span></td><td className={tableCell}>{model.is_active ? <span className="text-emerald-300">● Yes</span> : 'No'}</td><td className={tableCell}>{formatMetric(model.f1_score)}</td><td className={tableCell}>{model.training_sample_count}</td><td className={tableCell}>{new Date(model.created_at).toLocaleString()}</td><td className={tableCell}>{!model.is_active && ['completed', 'promoted'].includes(model.status) ? <button className="rounded-lg border border-rose-400/30 bg-rose-400/10 px-3 py-1.5 text-xs font-bold text-rose-300 transition hover:bg-rose-400/20 disabled:opacity-50" disabled={isWorking} onClick={() => void rollback(model.model_version)} type="button">Rollback</button> : '—'}</td></tr>)}</tbody>
          </table>
          {!isLoading && models.length === 0 && <p className="px-5 py-10 text-center text-sm text-slate-500">No registered models yet. Train the first model to begin monitoring.</p>}
        </div>
      </div>
    </section>
  )
}
