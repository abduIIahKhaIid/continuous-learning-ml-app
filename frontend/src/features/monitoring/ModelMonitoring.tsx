import { useCallback, useEffect, useState } from 'react'

import {
  fetchDriftReport,
  fetchModels,
  fetchMonitoringHealth,
  rollbackModel,
  runMonitoringCheck,
} from '../../lib/api/monitoring'
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
      if (requestError instanceof DOMException && requestError.name === 'AbortError') return
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

  return (
    <section className="card monitoring-card" aria-labelledby="monitoring-title">
      <div className="section-heading">
        <div>
          <h2 id="monitoring-title">Model monitoring</h2>
          <p className="intro">Live drift and verified performance health for the active model.</p>
        </div>
        <button className="secondary-button" disabled={isWorking} onClick={persistCheck} type="button">
          {isWorking ? 'Working…' : 'Save monitoring check'}
        </button>
      </div>

      {isLoading && <p>Loading monitoring…</p>}
      {error && <p className="error" role="alert">{error}</p>}
      {message && <p className="verified-result" role="status">{message}</p>}

      {health && (
        <dl className="training-status-grid monitoring-summary">
          <div><dt>Health</dt><dd className={`status-${health.status}`}>{health.status}</dd></div>
          <div><dt>Active model</dt><dd>{health.model_version ?? 'None'}</dd></div>
          <div><dt>Data drift</dt><dd>{health.data_drift_status}</dd></div>
          <div><dt>Performance</dt><dd>{health.performance_status}</dd></div>
          <div><dt>Feedback coverage</dt><dd>{(health.feedback_coverage * 100).toFixed(1)}%</dd></div>
          <div><dt>New verified samples</dt><dd>{health.new_verified_samples} / {health.retrain_threshold}</dd></div>
          <div><dt>Last training</dt><dd>{health.last_training_status ?? 'None'}</dd></div>
        </dl>
      )}

      <h3>Feature drift (PSI)</h3>
      {drift && drift.features.length > 0 ? (
        <div className="table-scroll">
          <table className="monitoring-table">
            <thead><tr><th>Feature</th><th>PSI</th><th>Status</th><th>Current / reference</th></tr></thead>
            <tbody>
              {drift.features.map((feature) => (
                <tr key={feature.feature_name}>
                  <td>{feature.feature_name}</td><td>{formatMetric(feature.psi)}</td>
                  <td>{feature.status}</td><td>{feature.current_samples} / {feature.reference_samples}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : !isLoading && <p>No active model profile is available yet.</p>}

      <h3>Model history</h3>
      <p className="intro">Rollback is a confirmed development/admin action that switches the active artifact without retraining.</p>
      <div className="table-scroll">
        <table className="monitoring-table">
          <thead><tr><th>Version</th><th>Status</th><th>Active</th><th>F1</th><th>Training samples</th><th>Created</th><th>Action</th></tr></thead>
          <tbody>
            {models.map((model) => (
              <tr key={model.model_version}>
                <td>{model.model_version}</td><td>{model.status}</td>
                <td>{model.is_active ? 'Yes' : 'No'}</td><td>{formatMetric(model.f1_score)}</td>
                <td>{model.training_sample_count}</td><td>{new Date(model.created_at).toLocaleString()}</td>
                <td>
                  {!model.is_active && ['completed', 'promoted'].includes(model.status) ? (
                    <button className="rollback-button" disabled={isWorking} onClick={() => void rollback(model.model_version)} type="button">Rollback</button>
                  ) : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
