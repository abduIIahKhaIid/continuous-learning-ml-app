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

export interface PredictionHistoryItem {
  id: number
  feature_1: number
  feature_2: number
  feature_3: number
  predicted_class: number
  prediction_probability: number | null
  model_version: string
  actual_label: number | null
  feedback_received: boolean
  created_at: string
  updated_at: string
}

export interface PredictionFeedbackRequest {
  actual_label: 0 | 1
}

export interface PredictionFeedbackStatus {
  prediction_id: number
  predicted_class: number
  actual_label: number | null
  feedback_received: boolean
  was_correct: boolean | null
  model_version: string
}

export interface PredictionFeedbackResponse
  extends PredictionFeedbackStatus {
  updated_at: string
}

export interface PredictionFeedbackSummary {
  total_predictions: number
  feedback_received: number
  feedback_pending: number
  correct_predictions: number
  incorrect_predictions: number
  verified_accuracy: number | null
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
