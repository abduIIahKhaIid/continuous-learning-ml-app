export interface DataRequest {
  feature_1: number
  feature_2: number
  feature_3: number
  label?: number
}

export interface DataResponse {
  feature_1: number
  feature_2: number
  feature_3: number
  label: number | null
}
