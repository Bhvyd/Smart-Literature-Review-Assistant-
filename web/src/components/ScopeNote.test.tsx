import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { Health } from '../api/types'
import { MAX_UPLOAD_FILES, MAX_UPLOAD_MB, MAX_UPLOAD_TOTAL_MB } from '../lib/uploadLimits'
import { ScopeNote } from './ScopeNote'

function health(overrides: Partial<Health> = {}): Health {
  return {
    status: 'ok',
    version: '0.1.0',
    embedding_model: 'specter2',
    generation_available: false,
    generation_reason: 'no_api_key',
    generation_configured: false,
    generation_last_error: null,
    refusal_tau: null,
    ...overrides,
  }
}

describe('scope note', () => {
  it('states the out-of-scope items and the upload bounds', () => {
    render(<ScopeNote health={health()} />)

    expect(screen.getByRole('heading', { name: 'Scope' })).toBeVisible()
    expect(screen.getByText('Not covered:').closest('li')).toHaveTextContent(/peer review/)

    const sources = screen.getByText('Sources:').closest('li')
    expect(sources).toHaveTextContent(
      `up to ${MAX_UPLOAD_FILES} PDFs (${MAX_UPLOAD_MB} MB each, ${MAX_UPLOAD_TOTAL_MB} MB total)`,
    )
  })

  it('describes comparison as a lead to check, not a finding', () => {
    render(<ScopeNote health={health()} />)
    const line = screen.getByText('Compare papers:').closest('li')
    expect(line).toHaveTextContent('verbatim quotes')
    expect(line).toHaveTextContent('A lead to check, not a finding.')
  })

  it('takes generation status from /health, not from a fixed sentence', () => {
    const { rerender } = render(<ScopeNote health={health()} />)
    const line = () => screen.getByText('Generation:').closest('li')
    expect(line()).toHaveTextContent('unavailable (no API key)')

    rerender(
      <ScopeNote
        health={health({
          generation_available: true,
          generation_reason: null,
          generation_configured: true,
          generation_last_error: { code: 'insufficient_quota', message: 'exhausted' },
        })}
      />,
    )
    expect(line()).toHaveTextContent('last call failed (quota exhausted)')
    expect(line()).not.toHaveTextContent(/unavailable/)

    rerender(<ScopeNote health={null} />)
    expect(line()).toHaveTextContent('status unknown')
  })

  it('names the model when generation is available', () => {
    render(
      <ScopeNote
        health={health({
          generation_available: true,
          generation_reason: null,
          generation_configured: true,
          generation_provider: 'gemini',
          generation_model: 'gemini-3.1-flash-lite',
        })}
      />,
    )
    const line = screen.getByText('Generation:').closest('li')
    expect(line).toHaveTextContent('gemini-3.1-flash-lite')
    expect(line).not.toHaveTextContent(/OpenAI/)
  })
})
