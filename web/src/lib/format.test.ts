import { describe, expect, it } from 'vitest'

import {
  degradationMessage,
  describeGenerationCode,
  formatAuthors,
  formatCount,
  formatScore,
  parseStatus,
} from './format'

describe('format helpers', () => {
  it('truncates a long author list to three names', () => {
    expect(formatAuthors(['Ada', 'Grace', 'Alan', 'Edsger'])).toBe('Ada, Grace, Alan et al.')
    expect(formatAuthors(['Ada', 'Grace', 'Alan'])).toBe('Ada, Grace, Alan')
    expect(formatAuthors([])).toBe('Unknown authors')
  })

  it('never invents a citation count it does not have', () => {
    expect(formatCount(2801)).toBe('2,801')
    expect(formatCount(null)).toBe('unknown')
    expect(formatCount(0)).toBe('0')
  })

  it('renders an absent score as nothing, not zero', () => {
    expect(formatScore(0.12345)).toBe('0.123')
    expect(formatScore(null)).toBe('')
  })

  it('labels parse status in user words and tones', () => {
    expect(parseStatus('ok')).toEqual({ label: 'full text', tone: 'ok' })
    expect(parseStatus('low_quality')).toEqual({ label: 'low-quality text', tone: 'warn' })
    expect(parseStatus('failed')).toEqual({ label: 'abstract only', tone: 'bad' })
    expect(parseStatus('not_attempted')).toEqual({ label: 'abstract only', tone: 'ok' })
  })

  it('maps each degradation code to a short message', () => {
    expect(degradationMessage('query_vector_missing')).toBe(
      'Dense search unavailable. Showing keyword results.',
    )
    expect(degradationMessage('no_dense_vectors')).toBe('No stored vectors. Showing keyword results.')
    expect(degradationMessage('openalex_fetch_failed')).toBe('Some citation data is missing.')
    expect(degradationMessage('something_else')).toContain('something_else')
  })

  it('describes every generation failure code without naming a provider', () => {
    const expected: Record<string, string> = {
      no_api_key: 'no API key',
      insufficient_quota: 'quota exhausted',
      rate_limited: 'rate limited, try again shortly',
      invalid_api_key: 'API key rejected',
      provider_unreachable: 'provider unreachable',
      provider_error: 'provider error',
      model_not_found: 'model not found',
      blocked: 'blocked by the provider',
      truncated: 'reply too long, discarded',
      empty_response: 'empty reply',
    }
    for (const [code, words] of Object.entries(expected)) {
      expect(describeGenerationCode(code)).toBe(words)
      expect(describeGenerationCode(code)).not.toMatch(/OpenAI|Gemini/i)
    }
    expect(describeGenerationCode('something_new')).toBe('something_new')
  })
})
