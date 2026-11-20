import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { AskView } from './AskView'
import { GraphView } from './GraphView'
import { MapView } from './MapView'
import { PaperDrawer } from './PaperDrawer'
import { RelatedWorkDraft } from './RelatedWorkDraft'

function resp(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => JSON.stringify(body),
  } as unknown as Response
}

type Pending = { resolve: (r: Response) => void; url: string }

function deferredFetch() {
  const pending: Pending[] = []
  const calls: string[] = []
  const fn = vi.fn((input: RequestInfo | URL) => {
    const url = String(input)
    calls.push(url)
    return new Promise<Response>((resolve) => pending.push({ resolve, url }))
  })
  vi.stubGlobal('fetch', fn)
  return { pending, calls, fn }
}

const graphOf = (title: string) => ({
  nodes: [{ id: 'a', title, year: 2020, citation_count: 1, resolved: true, cluster: 0 }],
  edges: [],
  rankings: [],
  unidentified: 0,
  clusters: { available: true, reason: null, ids: [0] },
})

afterEach(() => vi.unstubAllGlobals())

describe('graph', () => {
  it('collection switch: old payload gone at once, one fetch per key, late old response ignored', async () => {
    const d = deferredFetch()
    const { rerender } = render(
      <GraphView collectionId="c1" includeExternal={false} onToggleExternal={() => {}} onSelect={() => {}} />,
    )
    expect(d.calls).toEqual(['/api/collections/c1/graph'])
    await act(async () => d.pending[0].resolve(resp(graphOf('FROM C1'))))
    expect(await screen.findByLabelText(/FROM C1/)).toBeTruthy()

    rerender(<GraphView collectionId="c2" includeExternal={false} onToggleExternal={() => {}} onSelect={() => {}} />)
    expect(screen.queryByLabelText(/FROM C1/)).toBeNull()
    expect(d.calls).toEqual(['/api/collections/c1/graph', '/api/collections/c2/graph'])
    await act(async () => d.pending[1].resolve(resp(graphOf('FROM C2'))))
    expect(await screen.findByLabelText(/FROM C2/)).toBeTruthy()

    rerender(<GraphView collectionId="c2" includeExternal={true} onToggleExternal={() => {}} onSelect={() => {}} />)
    expect(screen.queryByLabelText(/FROM C2/)).toBeNull()
    expect(d.calls.length).toBe(3)
  })

  it('late response for the previous key does not land', async () => {
    const d = deferredFetch()
    const { rerender } = render(
      <GraphView collectionId="c1" includeExternal={false} onToggleExternal={() => {}} onSelect={() => {}} />,
    )
    rerender(<GraphView collectionId="c2" includeExternal={false} onToggleExternal={() => {}} onSelect={() => {}} />)
    await act(async () => d.pending[1].resolve(resp(graphOf('FROM C2'))))
    await act(async () => d.pending[0].resolve(resp(graphOf('FROM C1 LATE'))))
    expect(await screen.findByLabelText(/FROM C2/)).toBeTruthy()
    expect(screen.queryByLabelText(/FROM C1 LATE/)).toBeNull()
  })
})

const mapOf = () => ({
  points: Array.from({ length: 6 }, (_, i) => ({ paper_id: `p${i}`, x: i, y: i % 2, cluster: i < 3 ? 0 : 1 })),
  method: 'umap-hdbscan',
  params_hash: 'h',
  excluded: 0,
  cluster_note: null,
})

describe('map', () => {
  it('collection switch clears payload and labels; labels of c1 never decorate c2', async () => {
    const d = deferredFetch()
    const { rerender } = render(
      <MapView collectionId="c1" papersById={{}} onSelect={() => {}} generationAvailable />,
    )
    await act(async () => d.pending[0].resolve(resp(mapOf())))

    await vi.waitFor(() => expect(d.calls).toContain('/api/collections/c1/map/labels'))
    rerender(<MapView collectionId="c2" papersById={{}} onSelect={() => {}} generationAvailable />)

    expect(screen.getByText('Projecting the collection')).toBeTruthy()
    expect(screen.queryByRole('group', { name: /Similarity map/ })).toBeNull()

    const lab = d.pending.find((p) => p.url.endsWith('c1/map/labels'))!
    await act(async () => lab.resolve(resp({ labels: { '0': 'C1 NAME' }, unavailable: null, terms: { '0': ['c1term'] }, distinct: true })))
    const m2 = d.pending.find((p) => p.url.endsWith('c2/map'))!
    await act(async () => m2.resolve(resp(mapOf())))
    await vi.waitFor(() => expect(d.calls).toContain('/api/collections/c2/map/labels'))
    expect(screen.queryByText(/C1 NAME/)).toBeNull()
    expect(screen.queryByText(/c1term/)).toBeNull()
  })
})

describe('drawer', () => {
  it('paper switch resets paper and summary', async () => {
    const d = deferredFetch()
    const base = {
      collection_id: 'c', arxiv_id: null, openalex_id: null, doi: null, abstract: 'x', authors: [], year: 2020,
      venue: null, pdf_url: null, citation_count: 1, source: 'arxiv', parse_status: 'ok', parse_note: null,
      has_fulltext: 1, chunk_count: 1,
    }
    const { rerender } = render(<PaperDrawer paperId="p1" onClose={() => {}} generationAvailable />)
    await act(async () => d.pending[0].resolve(resp({ ...base, paper_id: 'p1', title: 'TITLE ONE' })))
    expect(await screen.findByText('TITLE ONE')).toBeTruthy()
    const user = userEvent.setup()
    const btn = screen.queryByRole('button', { name: /summar/i })
    if (btn) await user.click(btn)
    rerender(<PaperDrawer paperId="p2" onClose={() => {}} generationAvailable />)
    expect(screen.queryByText('TITLE ONE')).toBeNull()
    await act(async () => d.pending[d.pending.length - 1].resolve(resp({ ...base, paper_id: 'p2', title: 'TITLE TWO' })))
    expect(await screen.findByText('TITLE TWO')).toBeTruthy()

    for (const p of d.pending.filter((p) => p.url.includes('summary'))) {
      await act(async () => p.resolve(resp({ summary: 'LATE SUMMARY ONE', model: 'm', cached: false, unsupported_numbers: [] })))
    }
    expect(screen.queryByText(/LATE SUMMARY ONE/)).toBeNull()
  })
})

describe('related work', () => {
  it('request change resets draft; late old draft ignored', async () => {
    const d = deferredFetch()
    const draft = (t: string) => ({ paragraph: t, unknown_citations: [], uncited_papers: [], word_count: 1, ordered_by: 'x', ai_assisted_draft: true, label: 'l', citations: [], warnings: [], model: 'm', cached: false, papers: [] })
    const { rerender } = render(
      <RelatedWorkDraft collectionId="c1" request={{ cluster_id: 1 }} subject="cluster 1" onClose={() => {}} />,
    )
    rerender(<RelatedWorkDraft collectionId="c1" request={{ cluster_id: 2 }} subject="cluster 2" onClose={() => {}} />)
    expect(d.calls.length).toBe(2)
    await act(async () => d.pending[1].resolve(resp(draft('DRAFT TWO'))))
    await act(async () => d.pending[0].resolve(resp(draft('DRAFT ONE LATE'))))
    expect(await screen.findByText(/DRAFT TWO/)).toBeTruthy()
    expect(screen.queryByText(/DRAFT ONE LATE/)).toBeNull()
  })
})

describe('ask', () => {
  it('a slow answer for c1 does not land after switching to c2', async () => {
    const d = deferredFetch()
    const user = userEvent.setup()
    const health = { status: 'ok', version: '0', embedding_model: 'x', generation_available: true, generation_reason: null, generation_configured: true } as never
    const { rerender } = render(<AskView collectionId="c1" health={health} />)
    await user.type(screen.getByLabelText('Question'), 'what is speculative decoding')
    await user.click(screen.getByRole('button', { name: /ask/i }))
    expect(d.calls.length).toBe(1)
    rerender(<AskView collectionId="c2" health={health} />)
    await act(async () =>
      d.pending[0].resolve(
        resp({ answer: 'STALE ANSWER FROM C1', refused: false, citations: [], papers: [], retrieval: { mode: 'hybrid', degraded: [], top_score: 0.5, n_passages: 1, tau: 0.1, tau_calibrated: true }, confidence_basis: 'x', question: 'q', model: 'm', cached: false, latency_ms: 1 }),
      ),
    )
    expect(screen.queryByText(/STALE ANSWER FROM C1/)).toBeNull()
  })
})

describe('visible results are dropped the moment the key changes', () => {
  it('related work: a finished draft disappears and the loading state returns for the next request', async () => {
    const d = deferredFetch()
    const draft = (t: string) => ({ paragraph: t, unknown_citations: [], uncited_papers: [], word_count: 1, ordered_by: 'x', ai_assisted_draft: true, label: 'l', citations: [], warnings: [], model: 'm', cached: false, papers: [] })
    const { rerender } = render(
      <RelatedWorkDraft collectionId="c1" request={{ cluster_id: 1 }} subject="cluster 1" onClose={() => {}} />,
    )
    await act(async () => d.pending[0].resolve(resp(draft('DRAFT ONE'))))
    expect(await screen.findByText(/DRAFT ONE/)).toBeTruthy()
    rerender(<RelatedWorkDraft collectionId="c1" request={{ cluster_id: 2 }} subject="cluster 2" onClose={() => {}} />)
    expect(screen.queryByText(/DRAFT ONE/)).toBeNull()
    expect(screen.getByText('Drafting')).toBeTruthy()
  })

  it('ask: an answer for the previous collection is cleared on switch', async () => {
    const d = deferredFetch()
    const user = userEvent.setup()
    const health = { status: 'ok', version: '0', embedding_model: 'x', generation_available: true, generation_reason: null, generation_configured: true } as never
    const { rerender } = render(<AskView collectionId="c1" health={health} />)
    await user.type(screen.getByLabelText('Question'), 'what is speculative decoding')
    await user.click(screen.getByRole('button', { name: /ask/i }))
    await act(async () =>
      d.pending[0].resolve(
        resp({ answer: 'ANSWER FOR C1', refused: false, citation_anomaly: [], confidence: 1, answer_unavailable: null, refusal_reason: null, citations: [], papers: [], retrieval: { mode: 'hybrid', degraded: [], top_score: 0.5, n_passages: 1, tau: 0.1, tau_calibrated: true }, confidence_basis: 'x', question: 'q', model: 'm', cached: false, latency_ms: 1 }),
      ),
    )
    expect(await screen.findByText(/ANSWER FOR C1/)).toBeTruthy()
    rerender(<AskView collectionId="c2" health={health} />)
    expect(screen.queryByText(/ANSWER FOR C1/)).toBeNull()
  })
})
