import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { Health } from '../api/types'
import { Shell } from './Shell'

function health(overrides: Partial<Health> = {}): Health {
  return {
    status: 'ok',
    version: '0.1.0',
    embedding_model: 'allenai/specter2',
    generation_available: true,
    generation_reason: null,
    generation_configured: true,
    generation_last_error: null,
    generation_provider: 'gemini',
    generation_model: 'gemini-3.1-flash-lite',
    generation_config_note: null,
    ...overrides,
  }
}

function renderShell(value: Health | null) {
  return render(
    <Shell
      collections={[]}
      selectedId={null}
      onSelect={vi.fn()}
      onBuild={vi.fn()}
      onDelete={vi.fn()}
      health={value}
      view="papers"
      onViewChange={vi.fn()}
    >
      <p>content</p>
    </Shell>,
  )
}

describe('rail footer', () => {
  it('names the provider and model from /health', () => {
    renderShell(health())
    expect(screen.getByTestId('generation-provider')).toHaveTextContent('gemini · gemini-3.1-flash-lite')
    expect(screen.queryByText(/unavailable/)).toBeNull()
    expect(screen.queryByRole('note')).toBeNull()
    expect(screen.queryByText(/OpenAI/)).toBeNull()
  })

  it('shows a configuration slip once, plainly', () => {
    const note = "LLM_MODEL='gpt-5-nano' is not a gemini model; using 'gemini-3.1-flash-lite'"
    renderShell(health({ generation_config_note: note }))
    expect(screen.getAllByText(note)).toHaveLength(1)
  })

  it('reports an unavailable or failing provider in plain words, never as working', () => {
    const { rerender } = renderShell(
      health({ generation_available: false, generation_reason: 'no_api_key', generation_provider: 'openai', generation_model: 'gpt-5-mini' }),
    )
    expect(screen.getByTestId('generation-provider')).toHaveTextContent('openai · gpt-5-mini')
    expect(screen.getByText('Generation unavailable (no API key)')).toBeVisible()

    rerender(
      <Shell
        collections={[]}
        selectedId={null}
        onSelect={vi.fn()}
        onBuild={vi.fn()}
        onDelete={vi.fn()}
        health={health({ generation_last_error: { code: 'rate_limited', message: '429' } })}
        view="papers"
        onViewChange={vi.fn()}
      >
        <p>content</p>
      </Shell>,
    )
    expect(screen.queryByText('Generation unavailable')).toBeNull()
    expect(screen.getByText('rate limited, try again shortly')).toBeVisible()
  })

  it('does not guess a provider before /health answers', () => {
    renderShell(null)
    expect(screen.getByTestId('generation-provider')).toHaveTextContent('no generation provider')
    expect(screen.queryByText('Generation unavailable')).toBeNull()
  })
})
