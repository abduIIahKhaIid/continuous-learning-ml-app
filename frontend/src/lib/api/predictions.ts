import type {
  ModelStatus,
  PredictionRequest,
  PredictionResponse,
} from '../../types/prediction'
import { apiRequest } from './client'

export function submitPrediction(
  payload: PredictionRequest,
): Promise<PredictionResponse> {
  return apiRequest<PredictionResponse>('/api/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })
}

export function fetchModelStatus(signal?: AbortSignal): Promise<ModelStatus> {
  return apiRequest<ModelStatus>('/api/model/status', { signal })
}
