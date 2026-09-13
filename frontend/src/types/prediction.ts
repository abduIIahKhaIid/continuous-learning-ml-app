export interface PredictionRequest {
  feature_1: number
  feature_2: number
  feature_3: number
}

export interface PredictionResponse {
  prediction_id: number
  prediction: number
  predicted_class: number
  probability: number | null
  model_version: string
  created_at: string
}

export interface ModelStatus {
  model_available: boolean
  model_version: string | null
  algorithm: string | null
  created_at: string | null
  metrics: {
    accuracy: number | null
    f1_score: number | null
    roc_auc: number | null
  } | null
  status: 'ready' | 'unavailable'
  detail: string | null
}
