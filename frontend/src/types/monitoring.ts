export type MonitorStatus =
  | 'stable'
  | 'warning'
  | 'critical'
  | 'insufficient_data'

export interface FeatureDrift {
  feature_name: string
  psi: number | null
  status: MonitorStatus
  reference_samples: number
  current_samples: number
}

export interface DriftReport {
  model_version: string | null
  status: MonitorStatus
  sample_count: number
  minimum_samples: number
  window_size: number
  features: FeatureDrift[]
}

export interface MonitoringHealth {
  status: 'healthy' | 'warning' | 'unhealthy'
  model_status: 'healthy' | 'warning' | 'unhealthy'
  model_version: string | null
  artifact_available: boolean
  data_drift_status: MonitorStatus
  performance_status: MonitorStatus
  feedback_coverage: number
  training_in_progress: boolean
  last_training_status: string | null
  new_verified_samples: number
  retrain_threshold: number
  rollback_recommended: boolean
  message: string
}

export interface ModelSummary {
  model_version: string
  algorithm: string
  status: string
  is_active: boolean
  trigger_type: string | null
  training_sample_count: number
  accuracy: number | null
  precision: number | null
  recall: number | null
  f1_score: number | null
  roc_auc: number | null
  created_at: string
  completed_at: string | null
  promoted_at: string | null
}

export interface RollbackResult {
  active_model_version: string
  previous_model_version: string | null
  event_id: number
  message: string
}
