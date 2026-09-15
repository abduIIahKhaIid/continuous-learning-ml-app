import { apiRequest } from './client'
import type {
  DriftReport,
  ModelSummary,
  MonitoringHealth,
  RollbackResult,
} from '../../types/monitoring'

export function fetchMonitoringHealth(signal?: AbortSignal) {
  return apiRequest<MonitoringHealth>('/api/monitoring/health', { signal })
}

export function fetchDriftReport(signal?: AbortSignal) {
  return apiRequest<DriftReport>('/api/monitoring/drift', { signal })
}

export function fetchModels(signal?: AbortSignal) {
  return apiRequest<ModelSummary[]>('/api/models?skip=0&limit=50', { signal })
}

export function runMonitoringCheck() {
  return apiRequest('/api/monitoring/check', { method: 'POST' })
}

export function rollbackModel(modelVersion: string, reason: string | null) {
  return apiRequest<RollbackResult>(`/api/models/${modelVersion}/rollback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reason: reason || null }),
  })
}

