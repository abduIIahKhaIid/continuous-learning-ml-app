import type { TrainingStatus } from '../../types/training'
import { apiRequest } from './client'

export function fetchTrainingStatus(
  signal?: AbortSignal,
): Promise<TrainingStatus> {
  return apiRequest<TrainingStatus>('/api/training/status', { signal })
}
