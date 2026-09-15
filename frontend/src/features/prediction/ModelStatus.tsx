import { useEffect, useState } from 'react'

import { fetchModelStatus } from '../../lib/api/predictions'
import { isAbortError } from '../../lib/api/client'
import { statusBaseClass, statusTone } from '../../lib/ui'
import type { ModelStatus as ModelStatusData } from '../../types/prediction'

interface ModelStatusProps {
  refreshToken?: number
}

export function ModelStatus({ refreshToken = 0 }: ModelStatusProps) {
  const [model, setModel] = useState<ModelStatusData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const controller = new AbortController()
    let isCurrent = true
    setIsLoading(true)
    setError(null)
    fetchModelStatus(controller.signal)
      .then((result) => {
        if (isCurrent) setModel(result)
      })
      .catch((requestError: unknown) => {
        if (!isCurrent) return
        if (isAbortError(requestError)) return
        setError(
          requestError instanceof Error
            ? requestError.message
            : 'Unable to load model status.',
        )
      })
      .finally(() => {
        if (isCurrent) setIsLoading(false)
      })

    return () => {
      isCurrent = false
      controller.abort()
    }
  }, [refreshToken])

  if (isLoading) {
    return <p className={`${statusBaseClass} ${statusTone('neutral')}`}><span className="mr-2 size-2 animate-pulse rounded-full bg-slate-400 motion-reduce:animate-none" />Checking status…</p>
  }
  if (error) {
    return <p className={`${statusBaseClass} ${statusTone('failed')}`}>{error}</p>
  }
  if (!model?.model_available) {
    return (
      <p className={`${statusBaseClass} ${statusTone('warning')}`}>
        {model?.detail ?? 'No trained model available'}
        {model?.model_version ? ` (${model.model_version})` : ''}
      </p>
    )
  }

  return (
    <p className={`${statusBaseClass} ${statusTone('ready')}`}>
      <span className="mr-2 size-1.5 rounded-full bg-emerald-300" />
      Active: <strong className="ml-1 font-mono">{model.model_version}</strong>
    </p>
  )
}
