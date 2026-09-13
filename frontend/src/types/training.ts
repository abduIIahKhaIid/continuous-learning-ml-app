export type TrainingRunStatus =
  | 'queued'
  | 'running'
  | 'training'
  | 'completed'
  | 'failed'
  | 'rejected'
  | 'promoted'

export interface TrainingStatus {
  auto_retrain_enabled: boolean
  new_verified_samples: number
  retrain_threshold: number
  minimum_training_samples: number
  training_in_progress: boolean
  active_model_version: string | null
  last_training_run: {
    model_version: string
    status: TrainingRunStatus
    created_at: string
  } | null
}
