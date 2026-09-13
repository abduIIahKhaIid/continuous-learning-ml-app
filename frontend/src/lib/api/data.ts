import type { DataCreate, DataRead } from '../../types/data'
import { apiRequest } from './client'

export async function submitData(payload: DataCreate): Promise<DataRead> {
  return apiRequest<DataRead>('/api/data', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })
}
