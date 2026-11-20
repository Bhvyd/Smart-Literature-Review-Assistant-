import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { Collection } from '../api/types'
import { PapersView } from './PapersView'

function fakeResponse(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => JSON.stringify(body),
  } as unknown as Response
}

function collection(overrides: Partial<Collection> = {}): Collection {
  return {
    id: 'c1',
    mode: 'topic',
    query: 'retrieval augmented generation',
    status: 'ready',
    error: null,
    embedding_model: 'allenai/specter2',
    n_papers: 2,
    n_chunks: 10,
    created_at: '2026-11-13T10:00:00+00:00',
    updated_at: '2026-11-13T10:20:00+00:00',
    ...overrides,
  }
}

function paper(index: number, overrides: Record<string, unknown> = {}) {
  return {
    paper_id: `p${index}`,
    title: `Paper ${index}`,
    year: 2024,
    citation_count: 10,
    score: 1 - index / 10,
    best_chunk_id: 'chunk',
    abstract: 'An abstract.',
    authors: ['Ada Lovelace'],
    venue: null,
    pdf_url: null,
    arxiv_id: null,
    parse_status: 'ok',
    parse_note: null,
    has_fulltext: 1,
    ...overrides,
  }
}

function page(papers: unknown[], overrides: Record<string, unknown> = {}) {
  return {
    papers,
    mode: 'lexical',
    degraded: [],
    total: papers.length,
    offset: 0,
    limit: 20,
    ...overrides,
  }
}

function stub(fetchMock: (url: string) => Response) {
  const mock = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => fetchMock(String(input)))
  vi.stubGlobal('fetch', mock)
  return mock
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('papers view', () => {
  it('renders ranks and the result count from the API', async () => {
    stub((url) => {
      if (url.startsWith('/api/collections/c1/papers')) {
        return fakeResponse(page([paper(1), paper(2)], { total: 2 }))
      }
      throw new Error(`unexpected request: ${url}`)
    })

    const { container } = render(<PapersView collection={collection()} onOpenPaper={vi.fn()} />)

    expect(await screen.findByText('Paper 1')).toBeVisible()
    expect(screen.getByText('2 of 2')).toBeVisible()
    const ranks = Array.from(container.querySelectorAll('.paper-rank')).map((node) => node.textContent)
    expect(ranks).toEqual(['1', '2'])
  })

  it('does not draw a score bar when the score is absent', async () => {
    stub(() =>
      fakeResponse(page([paper(1, { score: null }), paper(2)], { total: 2, mode: 'influence' })),
    )

    const { container } = render(<PapersView collection={collection()} onOpenPaper={vi.fn()} />)
    await screen.findByText('Paper 1')

    const rows = container.querySelectorAll('.paper-row')
    expect(rows[0].querySelector('.score-fill')).toBeNull()
    expect(rows[1].querySelector('.score-fill')).not.toBeNull()
  })

  it('truncates a long author list and never invents a citation count', async () => {
    stub(() =>
      fakeResponse(
        page([
          paper(1, {
            authors: ['Ada', 'Grace', 'Alan', 'Edsger'],
            citation_count: null,
          }),
        ]),
      ),
    )

    render(<PapersView collection={collection()} onOpenPaper={vi.fn()} />)

    expect(await screen.findByText(/Ada, Grace, Alan et al\./)).toBeVisible()
    expect(screen.getByText('unknown citations')).toBeVisible()
  })

  it('explains a degraded response instead of showing an empty list', async () => {
    stub(() =>
      fakeResponse(
        page([paper(1)], {
          mode: 'lexical',
          degraded: ['query_vector_missing'],
          total: 1,
        }),
      ),
    )

    render(<PapersView collection={collection()} onOpenPaper={vi.fn()} />)

    expect(await screen.findByText(/Showing keyword results/)).toBeVisible()
    expect(screen.getByText('Paper 1')).toBeVisible()
  })

  it('asks again when the collection finishes building', async () => {
    let calls = 0
    stub((url) => {
      if (!url.startsWith('/api/collections/c1/papers')) {
        throw new Error(`unexpected request: ${url}`)
      }
      calls += 1
      if (calls === 1) {
        return fakeResponse(
          {
            error: {
              code: 'not_ready',
              message: 'collection c1 is still building',
              detail: { status: 'building' },
            },
          },
          409,
        )
      }
      return fakeResponse(page([paper(1)]))
    })

    const { rerender } = render(
      <PapersView collection={collection({ status: 'building', n_papers: 0 })} onOpenPaper={vi.fn()} />,
    )
    expect(await screen.findByText(/still building/)).toBeVisible()

    rerender(<PapersView collection={collection({ status: 'ready' })} onOpenPaper={vi.fn()} />)

    expect(await screen.findByRole('button', { name: /Paper 1/ })).toBeVisible()
    expect(screen.queryByText(/still building/)).toBeNull()
  })

  it('reports the chosen paper to the parent instead of owning the drawer', async () => {
    stub((url) => {
      if (url.startsWith('/api/collections/c1/papers')) return fakeResponse(page([paper(1)]))
      throw new Error(`unexpected request: ${url}`)
    })
    const onOpenPaper = vi.fn()
    const user = userEvent.setup()

    render(<PapersView collection={collection()} onOpenPaper={onOpenPaper} />)
    await user.click(await screen.findByRole('button', { name: /Paper 1/ }))

    expect(onOpenPaper).toHaveBeenCalledWith('p1')
  })

  it('clears a stuck load-more when a new search starts', async () => {
    const first = Array.from({ length: 20 }, (_, index) => paper(index + 1))
    const second = Array.from({ length: 5 }, (_, index) => paper(index + 21))
    let releaseMore: ((response: Response) => void) | undefined
    const more = new Promise<Response>((resolve) => {
      releaseMore = resolve
    })

    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input)
        if (url.includes('offset=20')) return more
        return fakeResponse(page(first, { total: 25 }))
      }),
    )
    const user = userEvent.setup()

    render(<PapersView collection={collection()} onOpenPaper={vi.fn()} />)
    expect(await screen.findByText('20 of 25')).toBeVisible()

    await user.click(screen.getByRole('button', { name: /Load more/ }))
    expect(screen.getByRole('button', { name: /Loading/ })).toBeDisabled()

    await user.selectOptions(screen.getByLabelText('Sort'), 'year')
    releaseMore?.(fakeResponse(page(second, { total: 25, offset: 20 })))

    await waitFor(() => expect(screen.getByRole('button', { name: /Load more/ })).toBeEnabled())
  })

  it('requests the next page at the current offset', async () => {
    const first = Array.from({ length: 20 }, (_, index) => paper(index + 1))
    const second = Array.from({ length: 5 }, (_, index) => paper(index + 21))

    const mock = stub((url) => {
      if (!url.startsWith('/api/collections/c1/papers')) {
        throw new Error(`unexpected request: ${url}`)
      }
      if (url.includes('offset=20')) return fakeResponse(page(second, { total: 25, offset: 20 }))
      return fakeResponse(page(first, { total: 25 }))
    })
    const user = userEvent.setup()

    render(<PapersView collection={collection()} onOpenPaper={vi.fn()} />)
    expect(await screen.findByText('20 of 25')).toBeVisible()

    await user.click(screen.getByRole('button', { name: /Load more/ }))

    expect(await screen.findByText('Paper 25')).toBeVisible()
    expect(screen.getByText('25 of 25')).toBeVisible()
    expect(mock.mock.calls.some((call) => String(call[0]).includes('offset=20'))).toBe(true)
  })

  it('reports a collection that is not ready instead of an empty page', async () => {
    stub(() =>
      fakeResponse(
        {
          error: {
            code: 'not_ready',
            message: 'collection is building, not ready',
            detail: { status: 'building', error: null },
          },
        },
        409,
      ),
    )

    render(
      <PapersView
        collection={collection({ status: 'building', n_papers: 0 })}
        onOpenPaper={vi.fn()}
      />,
    )

    expect(await screen.findByText(/still building/)).toBeVisible()
  })

  it('lists an upload that failed to parse with its note instead of hiding it', async () => {
    stub(() =>
      fakeResponse(
        page([
          paper(1, {
            source: 'upload',
            parse_status: 'failed',
            parse_note: 'PDF could not be parsed',
            has_fulltext: 0,
          }),
        ]),
      ),
    )

    render(<PapersView collection={collection({ mode: 'upload' })} onOpenPaper={vi.fn()} />)

    expect(await screen.findByText('Paper 1')).toBeVisible()
    expect(screen.getByText('abstract only')).toBeVisible()
    expect(screen.getByText('PDF could not be parsed')).toBeVisible()
  })

  it('offers no selection when generation is unavailable', async () => {
    stub(() => fakeResponse(page([paper(1), paper(2)])))

    render(<PapersView collection={collection()} onOpenPaper={vi.fn()} />)

    await screen.findByText('Paper 1')
    expect(screen.queryByRole('checkbox')).toBeNull()
    expect(screen.queryByRole('button', { name: 'Draft related work' })).toBeNull()
  })

  it('enforces 2–12 selected papers with the reason, then drafts from the selection', async () => {
    const papers = Array.from({ length: 13 }, (_, index) => paper(index + 1))
    const mock = stub((url) => {
      if (url === '/api/collections/c1/related-work') {
        return fakeResponse({
          paragraph: 'Paper 1 introduced the idea [Lovelace, 2024a].',
          citations: [
            { key: 'Lovelace, 2024a', paper_id: 'p1', title: 'Paper 1', year: 2024, url: null },
            { key: 'Lovelace, 2024b', paper_id: 'p2', title: 'Paper 2', year: 2024, url: null },
          ],
          unknown_citations: [],
          uncited_papers: ['Lovelace, 2024b'],
          word_count: 7,
          ordered_by: 'year, then citation count',
          ai_assisted_draft: true,
          label: 'AI-assisted draft - verify every sentence against its source before use',
          model: 'gpt-5-mini',
          cached: true,
        })
      }
      return fakeResponse(page(papers, { total: 13 }))
    })
    const user = userEvent.setup()

    render(<PapersView collection={collection()} onOpenPaper={vi.fn()} generationAvailable />)
    await screen.findByText('Paper 1')

    const draft = screen.getByRole('button', { name: 'Draft related work' })
    expect(draft).toBeDisabled()
    expect(screen.getByText('Select 2 or more papers.')).toBeVisible()

    const boxes = screen.getAllByRole('checkbox')
    for (const box of boxes) await user.click(box)
    expect(screen.getByText('Drafts take up to 12 papers.')).toBeVisible()
    expect(draft).toBeDisabled()

    await user.click(screen.getByRole('button', { name: 'Clear selection' }))
    await user.click(screen.getByRole('checkbox', { name: /Select “Paper 1”/ }))
    await user.click(screen.getByRole('checkbox', { name: /Select “Paper 2”/ }))
    expect(draft).toBeEnabled()
    await user.click(draft)

    expect(await screen.findByText(/Verify before use/)).toBeVisible()
    const call = mock.mock.calls.find((entry) => String(entry[0]).endsWith('/related-work'))
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({ paper_ids: ['p1', 'p2'] })
    expect(screen.getByText('Not cited:')).toBeVisible()
    expect(screen.getByRole('button', { name: 'Copy paragraph' })).toBeVisible()
  })

  it('enforces 2–8 papers for a comparison with the reason, and never runs it unasked', async () => {
    const papers = Array.from({ length: 9 }, (_, index) => paper(index + 1))
    const mock = stub((url) => {
      if (url === '/api/collections/c1/contradictions') throw new Error('must not be called')
      return fakeResponse(page(papers, { total: 9 }))
    })
    const user = userEvent.setup()

    render(<PapersView collection={collection()} onOpenPaper={vi.fn()} generationAvailable />)
    await screen.findByText('Paper 1')

    const compare = screen.getByRole('button', { name: 'Compare papers' })
    expect(compare).toBeDisabled()

    await user.click(screen.getByRole('checkbox', { name: /Select “Paper 1”/ }))
    expect(compare).toBeDisabled()
    expect(screen.getByText('Select 2 or more papers.')).toBeVisible()

    for (const box of screen.getAllByRole('checkbox').slice(1)) await user.click(box)

    expect(screen.getByText('Comparisons take up to 8 papers.')).toBeVisible()
    expect(compare).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Draft related work' })).toBeEnabled()

    await user.click(screen.getByRole('checkbox', { name: /Select “Paper 9”/ }))
    expect(compare).toBeEnabled()
    await user.click(compare)

    expect(screen.getByRole('heading', { name: /Compare papers · 8 selected papers/ })).toBeVisible()
    expect(screen.getByText(/28 pairs possible/)).toBeVisible()
    expect(mock.mock.calls.some((call) => String(call[0]).endsWith('/contradictions'))).toBe(false)
  })

  describe('while a request is in flight', () => {
    function deferredServer() {
      const pending: ((response: Response) => void)[] = []
      const mock = vi.fn(
        () => new Promise<Response>((resolve) => pending.push(resolve)),
      )
      vi.stubGlobal('fetch', mock)
      return { mock, answer: (body: unknown, status = 200) => pending.shift()?.(fakeResponse(body, status)) }
    }

    it('shows a loading state until the first page arrives, then the papers', async () => {
      const server = deferredServer()
      render(<PapersView collection={collection()} onOpenPaper={vi.fn()} />)

      expect(screen.getByRole('status')).toHaveTextContent('Ranking papers')
      expect(screen.queryByText('No papers matched')).toBeNull()

      server.answer(page([paper(1)]))
      expect(await screen.findByRole('button', { name: /Paper 1/ })).toBeVisible()
      expect(screen.queryByRole('status')).toBeNull()
    })

    it('keeps the previous list on screen while the next query loads', async () => {
      const server = deferredServer()
      const user = userEvent.setup()
      render(<PapersView collection={collection()} onOpenPaper={vi.fn()} />)
      server.answer(page([paper(1)]))
      expect(await screen.findByRole('button', { name: /Paper 1/ })).toBeVisible()

      await user.type(screen.getByLabelText('Query'), 'graphs{Enter}')

      expect(screen.getByRole('button', { name: /Paper 1/ })).toBeVisible()
      server.answer(page([paper(2)]))
      expect(await screen.findByRole('button', { name: /Paper 2/ })).toBeVisible()
      expect(screen.queryByRole('button', { name: /Paper 1/ })).toBeNull()
    })

    it('clears an error when the next request starts', async () => {
      const server = deferredServer()
      const user = userEvent.setup()
      render(<PapersView collection={collection()} onOpenPaper={vi.fn()} />)
      server.answer({ error: { code: 'server_error', message: 'boom' } }, 500)
      expect(await screen.findByRole('alert')).toHaveTextContent('boom')

      await user.type(screen.getByLabelText('Query'), 'again{Enter}')

      expect(screen.queryByRole('alert')).toBeNull()
    })
  })
})
