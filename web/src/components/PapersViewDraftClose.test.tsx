import { render, screen } from '@testing-library/react'
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

function collection(): Collection {
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
  }
}

function paper(index: number) {
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
  }
}

function page(papers: unknown[]) {
  return { papers, mode: 'lexical', degraded: [], total: papers.length, offset: 0, limit: 20 }
}

function stub() {
  const mock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)
    if (url.endsWith('/related-work')) {
      return fakeResponse({
        paragraph: 'A draft [Lovelace, 2024a].',
        citations: [
          { key: 'Lovelace, 2024a', paper_id: 'p1', title: 'Paper 1', year: 2024, url: null },
        ],
        unknown_citations: [],
        uncited_papers: [],
        word_count: 4,
        ordered_by: 'year, then citation count',
        ai_assisted_draft: true,
        label: 'AI-assisted draft',
        model: 'm',
        cached: true,
      })
    }
    return fakeResponse(page([paper(1), paper(2)]))
  })
  vi.stubGlobal('fetch', mock)
  return mock
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('papers view draft close', () => {
  it('closes the related-work draft when Close draft is clicked', async () => {
    stub()
    const user = userEvent.setup()

    render(<PapersView collection={collection()} onOpenPaper={vi.fn()} generationAvailable />)
    await screen.findByText('Paper 1')

    await user.click(screen.getByRole('checkbox', { name: /Select “Paper 1”/ }))
    await user.click(screen.getByRole('checkbox', { name: /Select “Paper 2”/ }))
    await user.click(screen.getByRole('button', { name: 'Draft related work' }))

    expect(
      await screen.findByRole('heading', { name: /Related-work draft · 2 selected papers/ }),
    ).toBeVisible()

    await user.click(screen.getByRole('button', { name: 'Close draft' }))

    expect(screen.queryByRole('heading', { name: /Related-work draft/ })).toBeNull()
  })
})
