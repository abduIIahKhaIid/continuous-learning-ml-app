export interface DataCreate {
  feature_1: number
  feature_2: number
  feature_3: number
  label?: number
}

export interface DataRead {
  id: number
  feature_1: number
  feature_2: number
  feature_3: number
  label: number | null
  created_at: string
  updated_at: string
  used_for_training: boolean
  training_batch_id: string | null
  model_version: string | null
}
