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
