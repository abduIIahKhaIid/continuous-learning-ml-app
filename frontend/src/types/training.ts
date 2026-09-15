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
  total_samples: number
  labelled_samples: number
  unlabelled_samples: number
  verified_feedback_samples: number
  new_verified_samples: number
  retrain_threshold: number
  new_verified_samples_needed: number
  minimum_training_samples: number
  verified_samples_needed_for_minimum: number
  retraining_data_ready: boolean
  training_in_progress: boolean
  queued_jobs: number
  running_jobs: number
  redis_available: boolean
  celery_worker_available: boolean
  active_model_version: string | null
  current_training_run: {
    id: number
    task_id: string | null
    status: TrainingRunStatus
    progress_stage: string
    model_version: string
    retry_count: number
    dispatch_error: string | null
    created_at: string
    dispatched_at: string | null
    started_at: string | null
    heartbeat_at: string | null
    completed_at: string | null
    is_stale: boolean
  } | null
  last_training_run: {
    model_version: string
    status: TrainingRunStatus
    created_at: string
  } | null
  latest_completed_run: {
    model_version: string
    status: TrainingRunStatus
    created_at: string
  } | null
}
