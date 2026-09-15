import type {
  ModelStatus,
  PredictionFeedbackRequest,
  PredictionFeedbackResponse,
  PredictionFeedbackStatus,
  PredictionFeedbackSummary,
  PredictionHistoryItem,
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

export function fetchPredictions(
  signal?: AbortSignal,
  skip = 0,
  limit = 20,
): Promise<PredictionHistoryItem[]> {
  return apiRequest<PredictionHistoryItem[]>(
    `/api/predictions?skip=${skip}&limit=${limit}`,
    { signal },
  )
}

export function submitPredictionFeedback(
  predictionId: number,
  payload: PredictionFeedbackRequest,
): Promise<PredictionFeedbackResponse> {
  return apiRequest<PredictionFeedbackResponse>(
    `/api/predictions/${predictionId}/feedback`,
    {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
  )
}

export function fetchPredictionFeedback(
  predictionId: number,
  signal?: AbortSignal,
): Promise<PredictionFeedbackStatus> {
  return apiRequest<PredictionFeedbackStatus>(
    `/api/predictions/${predictionId}/feedback`,
    { signal },
  )
}

export function fetchFeedbackSummary(
  signal?: AbortSignal,
): Promise<PredictionFeedbackSummary> {
  return apiRequest<PredictionFeedbackSummary>(
    '/api/predictions/feedback/summary',
    { signal },
  )
}
