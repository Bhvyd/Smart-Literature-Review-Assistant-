export function formatAuthors(authors: string[], max = 3): string {
  const named = authors.filter((author) => author.trim() !== '')
  if (named.length === 0) return 'Unknown authors'
  if (named.length <= max) return named.join(', ')
  return `${named.slice(0, max).join(', ')} et al.`
}

export function formatCount(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'unknown'
  return value.toLocaleString('en-US')
}

export function formatScore(score: number | null): string {
  if (score === null) return ''
  return score.toFixed(3)
}

export type ParseTone = 'ok' | 'warn' | 'bad'

export function parseStatus(status: string): { label: string; tone: ParseTone } {
  if (status === 'ok') return { label: 'full text', tone: 'ok' }
  if (status === 'low_quality') return { label: 'low-quality text', tone: 'warn' }
  if (status === 'failed') return { label: 'abstract only', tone: 'bad' }
  if (status === 'not_attempted') return { label: 'abstract only', tone: 'ok' }
  return { label: status, tone: 'ok' }
}

export function degradationMessage(code: string): string {
  if (code === 'query_vector_missing') {
    return 'Dense search unavailable. Showing keyword results.'
  }
  if (code === 'no_dense_vectors') {
    return 'No stored vectors. Showing keyword results.'
  }
  if (code === 'openalex_rate_limited' || code === 'openalex_fetch_failed') {
    return 'Some citation data is missing.'
  }
  return `Degraded: ${code}`
}

const GENERATION_CODES: Record<string, string> = {
  insufficient_quota: 'quota exhausted',
  no_api_key: 'no API key',
  not_implemented: 'not available in this build',
  invalid_api_key: 'API key rejected',
  rate_limited: 'rate limited, try again shortly',
  provider_unreachable: 'provider unreachable',
  provider_error: 'provider error',
  model_not_found: 'model not found',
  blocked: 'blocked by the provider',
  truncated: 'reply too long, discarded',
  empty_response: 'empty reply',
}

export function generationReason(code: string): string | null {
  return GENERATION_CODES[code] ?? null
}

export function describeGenerationCode(code: string): string {
  return generationReason(code) ?? code
}
