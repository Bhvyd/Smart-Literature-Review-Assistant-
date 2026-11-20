import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { AskResult, Health, Paper, PapersPage } from '../api/types'
import { AskView } from './AskView'

function fakeResponse(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => JSON.stringify(body),
  } as unknown as Response
}

function health(overrides: Partial<Health> = {}): Health {
  return {
    status: 'ok',
    version: '0.1.0',
    embedding_model: 'specter2',
    generation_available: false,
    generation_reason: 'not_implemented',
    generation_configured: false,
    ...overrides,
  }
}

function paper(id: string, title: string): Paper {
  return {
    paper_id: id,
    title,
    year: 2024,
    citation_count: 12,
    score: 0.71,
    best_chunk_id: `chunk-${id}`,
    abstract: null,
    authors: ['Ada Lovelace'],
    venue: null,
    pdf_url: null,
    arxiv_id: null,
    parse_status: 'ok',
    parse_note: null,
    has_fulltext: 1,
  }
}

function page(): PapersPage {
  return {
    papers: [paper('p1', 'Retrieval Augmented Generation'), paper('p2', 'Dense Passage Retrieval')],
    total: 2,
    offset: 0,
    limit: 8,
    mode: 'hybrid',
    degraded: [],
  }
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('ask view', () => {
  it('says answers are unavailable, with the reason the API reported', () => {
    render(<AskView collectionId="c1" health={health()} />)

    expect(screen.getByText('Answers unavailable.')).toBeVisible()
    expect(screen.getByText(/Not available in this build\./)).toBeVisible()
  })

  it('never shows an answer box or a fabricated answer', () => {
    const { container } = render(<AskView collectionId="c1" health={health()} />)

    expect(container.querySelector('textarea')).toBeNull()
    expect(screen.queryByRole('textbox', { name: /answer/i })).toBeNull()

    expect(screen.queryByText(/\[1\]/)).toBeNull()
    expect(screen.queryByRole('heading', { name: 'Retrieved papers' })).toBeNull()
  })

  it('names the missing credential when that is the reason', () => {
    render(
      <AskView
        collectionId="c1"
        health={health({ generation_reason: 'no_api_key', generation_configured: false })}
      />,
    )

    expect(screen.getByText(/No API key is configured/)).toBeVisible()
  })

  it('previews retrieved papers under an unavailable-answers banner, not as an answer', async () => {
    const mock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (!url.startsWith('/api/collections/c1/papers')) {
        throw new Error(`unexpected request: ${url}`)
      }
      return fakeResponse(page())
    })
    vi.stubGlobal('fetch', mock)
    const user = userEvent.setup()

    render(<AskView collectionId="c1" health={health()} />)
    await user.type(screen.getByLabelText('Question'), 'how does retrieval augmentation work?')
    await user.click(screen.getByRole('button', { name: 'Preview retrieved evidence' }))

    expect(await screen.findByRole('heading', { name: 'Retrieved papers' })).toBeVisible()
    expect(screen.getByText('Answers unavailable.')).toBeVisible()
    expect(screen.queryByRole('heading', { name: 'Answer' })).toBeNull()
    const url = String(mock.mock.calls[0][0])
    expect(url).toContain('q=how+does+retrieval+augmentation+work%3F')
    expect(url).toContain('mode=hybrid')
    expect(url).toContain('limit=8')
    expect(screen.getByText('Retrieval Augmented Generation')).toBeVisible()
    expect(screen.getByText('Dense Passage Retrieval')).toBeVisible()
  })

  it('reports a failed preview without inventing results', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        fakeResponse(
          {
            error: {
              code: 'not_found',
              message: 'collection c1 does not exist',
              detail: null,
            },
          },
          404,
        ),
      ),
    )
    const user = userEvent.setup()

    render(<AskView collectionId="c1" health={health()} />)
    await user.type(screen.getByLabelText('Question'), 'anything')
    await user.click(screen.getByRole('button', { name: 'Preview retrieved evidence' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('collection c1 does not exist')
    expect(screen.queryByRole('heading', { name: 'Retrieved papers' })).toBeNull()
  })

  it('drops the previous collection’s preview when the collection changes', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => fakeResponse(page())),
    )
    const user = userEvent.setup()

    const { rerender } = render(<AskView collectionId="c1" health={health()} />)
    await user.type(screen.getByLabelText('Question'), 'retrieval')
    await user.click(screen.getByRole('button', { name: 'Preview retrieved evidence' }))
    expect(await screen.findByRole('heading', { name: 'Retrieved papers' })).toBeVisible()

    rerender(<AskView collectionId="c2" health={health()} />)

    expect(screen.queryByRole('heading', { name: 'Retrieved papers' })).toBeNull()
    expect(screen.queryByText('Retrieval Augmented Generation')).toBeNull()
    expect(screen.getByLabelText('Question')).toHaveValue('retrieval')
  })

  it('shows the question box, not the preview, when generation is available', () => {
    render(
      <AskView
        collectionId="c1"
        health={health({ generation_available: true, generation_reason: null })}
      />,
    )

    expect(screen.queryByText('Answers unavailable.')).toBeNull()
    expect(screen.getByRole('textbox', { name: 'Question' })).toBeVisible()
    expect(screen.getByRole('button', { name: 'Ask' })).toBeDisabled()
    expect(screen.queryByRole('button', { name: 'Preview retrieved evidence' })).toBeNull()
  })

  it('names the last provider failure from /health instead of a fixed excuse', () => {
    render(
      <AskView
        collectionId="c1"
        health={health({
          generation_available: true,
          generation_reason: null,
          generation_last_error: { code: 'insufficient_quota', message: 'credit balance exhausted' },
        })}
      />,
    )

    expect(screen.getByText('Last generation call failed:')).toBeVisible()
    expect(screen.getByText(/quota exhausted/)).toBeVisible()
  })
})

const available = () => health({ generation_available: true, generation_reason: null })

function result(overrides: Partial<AskResult> = {}): AskResult {
  return {
    question: 'what is rag?',
    answer: 'RAG retrieves passages first [1]. It then conditions generation on them [2].',
    refused: false,
    refusal_reason: null,
    answer_unavailable: null,
    citations: [
      {
        n: 1,
        paper_id: 'p1',
        chunk_id: 'k1',
        title: 'Retrieval Augmented Generation',
        year: 2020,
        arxiv_id: '2005.11401',
        url: 'https://arxiv.org/abs/2005.11401',
        quote: 'We combine a retriever with a seq2seq generator.',
      },
      {
        n: 2,
        paper_id: 'p2',
        chunk_id: 'k2',
        title: 'Uploaded Notes',
        year: null,
        arxiv_id: null,
        url: 'upload://abc',
        quote: 'Generation is conditioned on the retrieved documents.',
      },
    ],
    confidence: 1,
    confidence_basis: 'citation_coverage',
    citation_anomaly: [],
    retrieval: {
      mode: 'hybrid',
      degraded: [],
      top_score: 0.83,
      n_passages: 8,
      tau: null,
      tau_calibrated: false,
    },
    model: 'gpt-5-mini',
    cached: false,
    latency_ms: 1200,
    ...overrides,
  }
}

async function askWith(body: AskResult) {
  const mock = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
    const url = String(input)
    if (url !== '/api/collections/c1/ask') throw new Error(`unexpected request: ${url}`)
    return fakeResponse(body)
  })
  vi.stubGlobal('fetch', mock)
  const user = userEvent.setup()
  render(<AskView collectionId="c1" health={available()} />)
  await user.type(screen.getByLabelText('Question'), 'what is rag?')
  await user.click(screen.getByRole('button', { name: 'Ask' }))
  return { mock, user }
}

describe('ask view — grounded answers', () => {
  it('turns citation markers into controls that focus the matching source', async () => {
    const { mock, user } = await askWith(result())

    expect(await screen.findByRole('heading', { name: 'Answer' })).toBeVisible()
    expect(JSON.parse(String(mock.mock.calls[0][1]?.body))).toEqual({
      question: 'what is rag?',
      mode: 'hybrid',
    })
    await user.click(screen.getByRole('button', { name: 'Source 2: Uploaded Notes' }))
    expect(screen.getByRole('listitem', { name: 'Source 2' })).toHaveFocus()
    expect(screen.getByText('Generation is conditioned on the retrieved documents.')).toBeVisible()

    expect(screen.getByRole('link', { name: 'Retrieval Augmented Generation' })).toHaveAttribute(
      'href',
      'https://arxiv.org/abs/2005.11401',
    )

    expect(screen.queryByRole('link', { name: 'Uploaded Notes' })).toBeNull()
    expect(screen.getByText('100%').closest('p')).toHaveTextContent('of sentences cite a source')
    expect(screen.getByText('AI-generated')).toBeVisible()
  })

  it('shows a refusal with its reason in plain words and no invented answer', async () => {
    await askWith(
      result({
        answer: 'The collection does not contain enough information to answer this question.',
        refused: true,
        refusal_reason: 'model_declined',
        citations: [],
        confidence: null,
      }),
    )

    expect(await screen.findByRole('heading', { name: 'No answer' })).toBeVisible()
    expect(screen.getByText('The collection does not cover this question.')).toBeVisible()
    expect(screen.queryByRole('heading', { name: 'Answer' })).toBeNull()
    expect(screen.queryByText('AI-generated')).toBeNull()
    expect(screen.queryByText(/of sentences cite a source/)).toBeNull()
  })

  it('shows the unavailable code and the retrieved evidence when no answer could be generated', async () => {
    await askWith(
      result({
        answer: null,
        answer_unavailable: { code: 'insufficient_quota', message: 'credit balance exhausted' },
        confidence: null,
        retrieval: { ...result().retrieval, degraded: ['query_vector_missing'] },
      }),
    )

    expect(await screen.findByText('No answer generated:')).toBeVisible()
    expect(screen.getByText(/quota exhausted/)).toBeVisible()
    expect(screen.getByRole('heading', { name: 'Retrieved evidence' })).toBeVisible()
    expect(screen.getByText('We combine a retriever with a seq2seq generator.')).toBeVisible()
    expect(screen.queryByRole('heading', { name: 'Answer' })).toBeNull()
    expect(screen.queryByText(/of sentences cite a source/)).toBeNull()
    expect(screen.queryByText('AI-generated')).toBeNull()
    expect(screen.getByText('Dense search unavailable. Showing keyword results.')).toBeVisible()

    const meta = document.querySelector('.generated-meta') as HTMLElement
    expect(meta).toHaveTextContent('no model output')
    expect(meta).not.toHaveTextContent('gpt-5-mini')
  })

  it('names the model only when a call completed', async () => {
    await askWith(result())
    expect(await screen.findByRole('heading', { name: 'Answer' })).toBeVisible()
    expect(document.querySelector('.generated-meta')).toHaveTextContent('model gpt-5-mini')
  })

  it.each(['no_retrieval_hits', 'below_threshold'])(
    'says no model was called when the question was refused before generation (%s)',
    async (refusal_reason) => {
      await askWith(result({ refused: true, refusal_reason, citations: [], confidence: null }))
      await screen.findByRole('heading', { name: 'No answer' })
      const meta = document.querySelector('.generated-meta') as HTMLElement
      expect(meta).toHaveTextContent('no model call')
      expect(meta).not.toHaveTextContent('gpt-5-mini')
    },
  )

  it('reports citations the model invented and the API removed', async () => {
    await askWith(result({ citation_anomaly: [7] }))

    expect(await screen.findByText('Invalid citations removed:')).toBeVisible()
    expect(screen.getByText('7')).toBeVisible()
  })
})
