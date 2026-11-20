import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { PaperDrawer } from './PaperDrawer'

function fakeResponse(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => JSON.stringify(body),
  } as unknown as Response
}

const detail = {
  paper_id: 'p1',
  collection_id: 'c1',
  arxiv_id: '2402.00001',
  openalex_id: 'W123',
  doi: null,
  title: 'Attention Is All You Need',
  abstract: 'The dominant sequence transduction models are based on complex recurrent networks.',
  authors: ['Ashish Vaswani', 'Noam Shazeer'],
  year: 2017,
  venue: 'NeurIPS',
  pdf_url: 'https://example.org/paper.pdf',
  citation_count: 120000,
  source: 'arxiv',
  parse_status: 'ok',
  parse_note: null,
  has_fulltext: 1,
  chunk_count: 3,
}

function stub(detailBody: unknown) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => fakeResponse(detailBody)),
  )
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('paper drawer', () => {
  it('shows the stored row and closes on Escape', async () => {
    stub(detail)
    const onClose = vi.fn()
    const user = userEvent.setup()

    render(<PaperDrawer paperId="p1" onClose={onClose} />)

    expect(await screen.findByText(/dominant sequence transduction/)).toBeVisible()
    expect(screen.getByText('Ashish Vaswani, Noam Shazeer')).toBeVisible()
    expect(screen.getByText('120,000')).toBeVisible()
    expect(screen.getByText(/3 chunks indexed/)).toBeVisible()
    expect(screen.getByRole('link', { name: 'Open the PDF' })).toHaveAttribute(
      'href',
      'https://example.org/paper.pdf',
    )

    await user.keyboard('{Escape}')
    expect(onClose).toHaveBeenCalled()
  })

  it('moves focus to Close so Shift+Tab cannot fall out of the dialog', async () => {
    stub(detail)
    const user = userEvent.setup()

    render(<PaperDrawer paperId="p1" onClose={vi.fn()} />)

    const close = await screen.findByRole('button', { name: 'Close' })
    expect(close).toHaveFocus()

    await user.tab({ shift: true })

    const dialog = screen.getByRole('dialog', { name: 'Attention Is All You Need' })
    expect(dialog.contains(document.activeElement)).toBe(true)
  })

  it('says a paper was ranked from its abstract when there is no full text', async () => {
    stub({
      ...detail,
      parse_status: 'failed',
      parse_note: 'no PDF was available',
      has_fulltext: 0,
      chunk_count: 0,
    })

    render(<PaperDrawer paperId="p1" onClose={vi.fn()} />)

    expect(await screen.findByText('Ranked from the abstract only.')).toBeVisible()
    expect(screen.getByText('no PDF was available')).toBeVisible()
    expect(screen.getByText('abstract only')).toBeVisible()
  })

  it('reports a detail fetch failure without inventing a paper', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        fakeResponse(
          { error: { code: 'not_found', message: 'paper p1 does not exist', detail: null } },
          404,
        ),
      ),
    )

    render(<PaperDrawer paperId="p1" onClose={vi.fn()} />)

    expect(await screen.findByRole('alert')).toHaveTextContent('paper p1 does not exist')
    expect(screen.queryByText('Attention Is All You Need')).not.toBeInTheDocument()
  })

  it('links an upload to its stored PDF and never to the upload:// scheme', async () => {
    stub({
      ...detail,
      arxiv_id: null,
      source: 'upload',
      pdf_url: 'upload://0123abcd',
      source_url: '/api/papers/p1/pdf',
      parse_status: 'failed',
      parse_note: 'PDF could not be parsed',
      has_fulltext: 0,
      chunk_count: 0,
    })

    render(<PaperDrawer paperId="p1" onClose={vi.fn()} />)

    expect(await screen.findByText('This PDF could not be read.')).toBeVisible()
    expect(screen.getByText('PDF could not be parsed')).toBeVisible()
    expect(screen.getByRole('link', { name: 'Open the uploaded PDF' })).toHaveAttribute(
      'href',
      '/api/papers/p1/pdf',
    )
    expect(screen.queryByRole('link', { name: 'Open the PDF' })).toBeNull()
  })
})

describe('paper drawer — summary', () => {
  function stubSummary(summary: { body: unknown; status?: number }) {
    const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url === '/api/papers/p1/summary' && init?.method === 'POST') {
        return fakeResponse(summary.body, summary.status ?? 200)
      }
      if (url === '/api/papers/p1') return fakeResponse(detail)
      throw new Error(`unexpected request: ${url}`)
    })
    vi.stubGlobal('fetch', mock)
    return mock
  }

  it('offers no Summarise action when generation is unavailable', async () => {
    stub(detail)
    render(<PaperDrawer paperId="p1" onClose={vi.fn()} />)

    await screen.findByText(/dominant sequence transduction/)
    expect(screen.queryByRole('button', { name: 'Summarise' })).toBeNull()
  })

  it('never summarises on open, labels the result and warns about unsupported numbers', async () => {
    const mock = stubSummary({
      body: {
        paper_id: 'p1',
        title: 'Attention Is All You Need',
        summary: 'Problem: sequence transduction.\nMethod: attention only.\nResult: 28.4 BLEU and 99% faster.',
        model: 'gpt-5-mini',
        cached: false,
        sources: ['title', 'abstract', 'introduction (chunk 1)'],
        unsupported_numbers: ['28.4', '99'],
        ai_generated: true,
      },
    })
    const user = userEvent.setup()
    render(<PaperDrawer paperId="p1" onClose={vi.fn()} generationAvailable />)

    await screen.findByText(/dominant sequence transduction/)
    expect(mock.mock.calls.some((call) => String(call[0]).endsWith('/summary'))).toBe(false)

    await user.click(screen.getByRole('button', { name: 'Summarise' }))

    expect(await screen.findByText(/attention only/)).toBeVisible()
    expect(screen.getByText('AI-generated')).toBeVisible()
    expect(screen.getByText('Numbers not found in the source text:')).toBeVisible()
    expect(screen.getByText('28.4, 99')).toBeVisible()
    expect(screen.getByText(/From title, abstract, introduction \(chunk 1\)/)).toBeVisible()
  })

  it('shows the reason in plain words when generation is unavailable for the summary', async () => {
    stubSummary({
      status: 503,
      body: {
        error: {
          code: 'generation_unavailable',
          message: 'credit balance exhausted',
          detail: { reason: 'insufficient_quota' },
        },
      },
    })
    const user = userEvent.setup()
    render(<PaperDrawer paperId="p1" onClose={vi.fn()} generationAvailable />)

    await user.click(await screen.findByRole('button', { name: 'Summarise' }))

    expect(await screen.findByText('No summary was written.')).toBeVisible()
    expect(screen.getByText(/quota exhausted/)).toBeVisible()
    expect(screen.getByRole('button', { name: 'Try summarising again' })).toBeVisible()
  })
})
