import type {
  AskResult,
  Collection,
  ContradictionRequest,
  Contradictions,
  DeletedCollection,
  GraphPayload,
  Health,
  Job,
  MapLabels,
  MapPayload,
  Mode,
  PaperDetail,
  PaperSummary,
  PapersPage,
  RelatedWork,
  RelatedWorkRequest,
  Sort,
} from './types'

const API = '/api'

export class ApiError extends Error {
  readonly code: string
  readonly status: number
  readonly detail: unknown

  constructor(code: string, message: string, status: number, detail: unknown = null) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
    this.detail = detail
  }
}

interface Envelope {
  error?: { code?: string; message?: string; detail?: unknown }
}

function encodeQuery(params: Record<string, string | number | boolean | undefined | null>): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null) search.set(key, String(value))
  }
  const encoded = search.toString()
  return encoded ? `?${encoded}` : ''
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API}${path}`, {
      ...init,
      headers: {
        Accept: 'application/json',

        ...(typeof init?.body === 'string' ? { 'Content-Type': 'application/json' } : {}),
        ...init?.headers,
      },
    })
  } catch {
    throw new ApiError(
      'network_unavailable',
      'Cannot reach the server.',
      0,
    )
  }

  const text = await response.text()
  let payload: unknown = null
  if (text) {
    try {
      payload = JSON.parse(text)
    } catch {
      payload = null
    }
  }

  if (!response.ok) {
    const envelope = (payload ?? {}) as Envelope
    const error = envelope.error
    throw new ApiError(
      error?.code ?? 'http_error',
      error?.message ?? `Request failed with status ${response.status}`,
      response.status,
      error?.detail ?? null,
    )
  }
  return payload as T
}

export function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  if (error instanceof Error) return error.message
  return 'Something went wrong.'
}

export function getHealth(): Promise<Health> {
  return request<Health>('/health')
}

export function listCollections(): Promise<{ collections: Collection[] }> {
  return request<{ collections: Collection[] }>('/collections')
}

export function getCollection(collectionId: string): Promise<Collection> {
  return request<Collection>(`/collections/${encodeURIComponent(collectionId)}`)
}

export function deleteCollection(collectionId: string): Promise<DeletedCollection> {
  return request<DeletedCollection>(`/collections/${encodeURIComponent(collectionId)}`, {
    method: 'DELETE',
  })
}

export function cancelCollectionBuild(
  collectionId: string,
): Promise<{ collection_id: string; status: string }> {
  return request<{ collection_id: string; status: string }>(
    `/collections/${encodeURIComponent(collectionId)}/cancel`,
    { method: 'POST' },
  )
}

export function createCollection(query: string, maxPapers: number): Promise<{
  collection_id: string
  job_id: string
  status: string
}> {
  return request('/collections', {
    method: 'POST',
    body: JSON.stringify({ query, max_papers: maxPapers }),
  })
}

export function getJob(jobId: string): Promise<Job> {
  return request<Job>(`/jobs/${encodeURIComponent(jobId)}`)
}

export function listPapers(
  collectionId: string,
  params: {
    q?: string
    sort?: Sort
    mode?: Mode
    limit?: number
    offset?: number
  } = {},
): Promise<PapersPage> {
  return request<PapersPage>(
    `/collections/${encodeURIComponent(collectionId)}/papers${encodeQuery(params)}`,
  )
}

export function getPaper(paperId: string): Promise<PaperDetail> {
  return request<PaperDetail>(`/papers/${encodeURIComponent(paperId)}`)
}

export function getMap(collectionId: string): Promise<MapPayload> {
  return request<MapPayload>(`/collections/${encodeURIComponent(collectionId)}/map`)
}

export function uploadCollection(
  files: File[],
  name: string,
): Promise<{ collection_id: string; job_id: string; status: string }> {
  const form = new FormData()
  for (const file of files) form.append('files', file, file.name)
  if (name.trim()) form.append('name', name.trim())
  return request('/collections/upload', { method: 'POST', body: form })
}

export function getMapLabels(collectionId: string): Promise<MapLabels> {
  return request<MapLabels>(`/collections/${encodeURIComponent(collectionId)}/map/labels`)
}

export function ask(
  collectionId: string,
  question: string,
  mode: Mode = 'hybrid',
): Promise<AskResult> {
  return request<AskResult>(`/collections/${encodeURIComponent(collectionId)}/ask`, {
    method: 'POST',
    body: JSON.stringify({ question, mode }),
  })
}

export function summarisePaper(paperId: string): Promise<PaperSummary> {
  return request<PaperSummary>(`/papers/${encodeURIComponent(paperId)}/summary`, {
    method: 'POST',
  })
}

export function draftRelatedWork(
  collectionId: string,
  body: RelatedWorkRequest,
): Promise<RelatedWork> {
  return request<RelatedWork>(`/collections/${encodeURIComponent(collectionId)}/related-work`, {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function compareContradictions(
  collectionId: string,
  body: ContradictionRequest,
  maxPairs: number,
): Promise<Contradictions> {
  return request<Contradictions>(`/collections/${encodeURIComponent(collectionId)}/contradictions`, {
    method: 'POST',
    body: JSON.stringify({ ...body, max_pairs: maxPairs }),
  })
}

export function reasonCode(error: unknown): string | null {
  if (!(error instanceof ApiError)) return null
  const reason = (error.detail as { reason?: unknown } | null)?.reason
  return typeof reason === 'string' ? reason : error.code
}

export function safeHref(url: string | null | undefined): string | null {
  if (!url) return null
  if (/^https?:\/\//i.test(url) || url.startsWith('/')) return url
  return null
}

export function getGraph(
  collectionId: string,
  includeExternal = false,
): Promise<GraphPayload> {
  return request<GraphPayload>(
    `/collections/${encodeURIComponent(collectionId)}/graph${encodeQuery({
      include_external: includeExternal || undefined,
    })}`,
  )
}
