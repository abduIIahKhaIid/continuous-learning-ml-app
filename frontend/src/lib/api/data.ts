import type { DataRequest, DataResponse } from '../../types/data'

function getApiBaseUrl(): string {
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL

  if (!apiBaseUrl) {
    throw new Error('VITE_API_BASE_URL is not configured.')
  }

  return apiBaseUrl.replace(/\/$/, '')
}

export async function submitData(payload: DataRequest): Promise<DataResponse> {
  const response = await fetch(`${getApiBaseUrl()}/api/data`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    throw new Error(`API request failed with status ${response.status}.`)
  }

  return (await response.json()) as DataResponse
}
