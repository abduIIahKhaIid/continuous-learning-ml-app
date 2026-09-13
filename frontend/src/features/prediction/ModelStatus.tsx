import { useEffect, useState } from 'react'

import { fetchModelStatus } from '../../lib/api/predictions'
import type { ModelStatus as ModelStatusData } from '../../types/prediction'

export function ModelStatus() {
  const [model, setModel] = useState<ModelStatusData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const controller = new AbortController()
    fetchModelStatus(controller.signal)
      .then(setModel)
      .catch((requestError: unknown) => {
        if (requestError instanceof DOMException && requestError.name === 'AbortError') {
          return
        }
        setError(
          requestError instanceof Error
            ? requestError.message
            : 'Unable to load model status.',
        )
      })
      .finally(() => setIsLoading(false))

    return () => controller.abort()
  }, [])

  if (isLoading) {
    return <p className="model-status">Checking model status…</p>
  }
  if (error) {
    return <p className="model-status error">{error}</p>
  }
  if (!model?.model_available) {
    return (
      <p className="model-status unavailable">
        {model?.detail ?? 'No trained model available'}
        {model?.model_version ? ` (${model.model_version})` : ''}
      </p>
    )
  }

  return (
    <p className="model-status ready">
      Active model: <strong>{model.model_version}</strong> · Status: Ready
    </p>
  )
}
