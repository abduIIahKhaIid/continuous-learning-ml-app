interface ErrorBody {
  detail?: string | Array<{ msg?: string }>
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export function isAbortError(error: unknown): boolean {
  return (
    (error instanceof DOMException && error.name === 'AbortError') ||
    (error instanceof Error && error.name === 'AbortError')
  )
}

function getApiBaseUrl(): string {
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL

  if (!apiBaseUrl) {
    throw new Error('VITE_API_BASE_URL is not configured.')
  }

  return apiBaseUrl.replace(/\/$/, '')
}

function errorMessage(body: ErrorBody | null, status: number): string {
  if (typeof body?.detail === 'string') {
    return body.detail
  }
  if (Array.isArray(body?.detail)) {
    const messages = body.detail
      .map((item) => item.msg)
      .filter((message): message is string => Boolean(message))
    if (messages.length > 0) {
      return messages.join(' ')
    }
  }
  return `API request failed with status ${status}.`
}

export async function apiRequest<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${getApiBaseUrl()}${path}`, options)
  } catch (error) {
    // Request cancellation is expected during component cleanup and polling.
    // Preserve AbortError so callers can ignore it instead of showing a false
    // network failure.
    if (options?.signal?.aborted) {
      throw new DOMException('The request was cancelled.', 'AbortError')
    }
    if (isAbortError(error)) {
      throw error
    }
    throw new Error(
      error instanceof Error
        ? `Network error: ${error.message}`
        : 'Unable to reach the API.',
    )
  }

  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as ErrorBody | null
    throw new ApiError(errorMessage(body, response.status), response.status)
  }

  return (await response.json()) as T
}
