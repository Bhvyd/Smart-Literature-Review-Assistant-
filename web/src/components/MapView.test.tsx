import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { MapPayload, Paper } from '../api/types'
import { MapView } from './MapView'

function fakeResponse(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => JSON.stringify(body),
  } as unknown as Response
}

function paper(id: string, title: string, year = 2024): Paper {
  return {
    paper_id: id,
    title,
    year,
    citation_count: 10,
    score: null,
    best_chunk_id: null,
    abstract: null,
    authors: [],
    venue: null,
    pdf_url: null,
    arxiv_id: null,
    parse_status: 'ok',
    parse_note: null,
    has_fulltext: 1,
  }
}

function mapPayload(overrides: Partial<MapPayload> = {}): MapPayload {
  return {
    points: [
      { paper_id: 'p1', x: -3.2, y: 1.4, cluster: 0 },
      { paper_id: 'p2', x: 0.5, y: -2.1, cluster: 0 },
      { paper_id: 'p3', x: 4.4, y: 3.3, cluster: 3 },
      { paper_id: 'p4', x: 6.1, y: -0.4, cluster: -1 },
    ],
    method: 'umap-hdbscan',
    params_hash: 'abc123',
    excluded: 0,
    cluster_note: null,
    ...overrides,
  }
}

function stubMap(
  body: unknown,
  status = 200,
  labels: unknown = { labels: {}, unavailable: null },
  extra?: (url: string, init?: RequestInit) => Response | undefined,
) {
  const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    const handled = extra?.(url, init)
    if (handled) return handled
    if (url === '/api/collections/c1/map/labels') return fakeResponse(labels)
    if (url === '/api/collections/c1/map') return fakeResponse(body, status)
    throw new Error(`unexpected request: ${url}`)
  })
  vi.stubGlobal('fetch', mock)
  return mock
}

const papersById: Record<string, Paper> = {
  p1: paper('p1', 'Paper One'),
  p2: paper('p2', 'Paper Two'),
  p3: paper('p3', 'Paper Three'),
  p4: paper('p4', 'Paper Four'),
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('map view', () => {
  it('draws one point per paper, marks noise, and reports a click', async () => {
    stubMap(mapPayload())
    const onSelect = vi.fn()
    const user = userEvent.setup()

    const { container } = render(
      <MapView collectionId="c1" papersById={papersById} onSelect={onSelect} />,
    )

    const chart = await screen.findByRole('group', { name: /Similarity map of 4 papers/ })
    expect(within(chart).getAllByRole('button')).toHaveLength(4)
    expect(container.querySelectorAll('circle.map-point')).toHaveLength(4)
    expect(container.querySelectorAll('circle.map-point.noise')).toHaveLength(1)

    await user.click(screen.getByRole('button', { name: /Paper Three/ }))
    expect(onSelect).toHaveBeenCalledWith('p3')
  })

  it('builds the legend from the payload, not from a fixed vocabulary', async () => {
    stubMap(mapPayload())

    render(<MapView collectionId="c1" papersById={papersById} onSelect={vi.fn()} />)

    expect(await screen.findByText('Cluster 0 · 2 papers')).toBeVisible()
    expect(screen.getByText('Cluster 3 · 1 papers')).toBeVisible()
    expect(screen.getByText('Unclustered · 1 papers')).toBeVisible()
  })

  it('shows each cluster’s distinctive terms beside its name, and says when names still overlap', async () => {
    stubMap(mapPayload(), 200, {
      labels: { '0': 'Tree Verification', '3': 'Tree Verification Methods' },
      unavailable: null,
      terms: { '0': ['draft', 'tree attention'], '3': ['quantization', 'kernels'] },
      distinct: false,
      collisions: [[0, 3]],
      unnamed: [],
      uninformative: [],
    })

    render(<MapView collectionId="c1" papersById={papersById} onSelect={vi.fn()} />)

    expect(await screen.findByText('draft, tree attention')).toBeVisible()
    expect(screen.getByText('quantization, kernels')).toBeVisible()
    expect(screen.getByText(/Some names overlap/)).toBeVisible()
  })

  it('still shows the distinctive terms when naming is unavailable', async () => {
    stubMap(mapPayload(), 200, {
      labels: {},
      unavailable: { code: 'no_api_key', message: 'no key' },
      terms: { '0': ['draft'], '3': ['kernels'] },
      distinct: false,
      unnamed: [0, 3],
      collisions: [],
      uninformative: [],
    })

    render(<MapView collectionId="c1" papersById={papersById} onSelect={vi.fn()} />)

    expect(await screen.findByText('draft')).toBeVisible()
    expect(screen.getByText(/Cluster names unavailable/)).toBeVisible()
    expect(screen.queryByText(/Some names overlap/)).toBeNull()
    expect(screen.queryByText(/unnamed/)).toBeNull()
  })

  it('says which clusters the model left unnamed', async () => {
    stubMap(mapPayload(), 200, {
      labels: { '0': 'Tree Verification' },
      unavailable: null,
      terms: {},
      distinct: false,
      unnamed: [3],
      collisions: [],
      uninformative: [],
    })
    render(<MapView collectionId="c1" papersById={papersById} onSelect={vi.fn()} />)
    expect(await screen.findByText(/One cluster is unnamed/)).toBeVisible()
    expect(screen.getByText('Cluster 3 · 1 papers')).toBeVisible()
    expect(screen.queryByText(/Some names overlap/)).toBeNull()
  })

  it('renders a map refusal with the numbers the API reported', async () => {
    stubMap(
      {
        error: {
          code: 'map_needs_more_papers',
          message: 'a similarity map needs at least 15 papers with vectors',
          detail: { reason: 'map_needs_more_papers', needed: 15, have: 4 },
        },
      },
      409,
    )

    render(<MapView collectionId="c1" papersById={{}} onSelect={vi.fn()} />)

    expect(await screen.findByText(/Not enough papers to map/)).toBeVisible()
    expect(screen.getByText(/Needs 15, this collection has 4/)).toBeVisible()
  })

  it('shows how many papers have no vectors and warns about unreliable clusters', async () => {
    stubMap(mapPayload({ excluded: 2, cluster_note: 'clusters_unreliable' }))

    render(<MapView collectionId="c1" papersById={papersById} onSelect={vi.fn()} />)

    expect(await screen.findByText(/papers are not plotted \(no vectors\)/)).toBeVisible()
    expect(screen.getByText('2')).toBeVisible()
    expect(screen.getByText('Clustering was unreliable.')).toBeVisible()
  })

  it('renders a hundred points without extra layout work', async () => {
    const many = Array.from({ length: 100 }, (_, index) => ({
      paper_id: `q${index}`,
      x: Math.sin(index) * 10,
      y: Math.cos(index) * 6,
      cluster: index % 6,
    }))
    stubMap(mapPayload({ points: many }))

    const { container } = render(
      <MapView collectionId="c1" papersById={{}} onSelect={vi.fn()} />,
    )

    await screen.findByRole('group', { name: /Similarity map of 100 papers/ })
    expect(container.querySelectorAll('circle.map-point')).toHaveLength(100)
  })

  it('says which vectors were projected', async () => {
    stubMap(mapPayload({ vector_source: 'chunk_mean' }))

    render(<MapView collectionId="c1" papersById={papersById} onSelect={vi.fn()} />)

    expect(await screen.findByText(/placed from chunk vectors/)).toBeVisible()
  })

  it('names clusters from the labels endpoint, once, after the map renders', async () => {
    const mock = stubMap(mapPayload(), 200, {
      labels: { '0': 'Dense retrieval', '3': 'Graph learning' },
      unavailable: null,
    })

    const { rerender } = render(
      <MapView collectionId="c1" papersById={papersById} onSelect={vi.fn()} />,
    )

    expect(await screen.findByText('Dense retrieval · 2 papers')).toBeVisible()
    expect(screen.getByText('Graph learning · 1 papers')).toBeVisible()
    expect(screen.queryByText(/Cluster names unavailable/)).toBeNull()

    rerender(<MapView collectionId="c1" papersById={{ ...papersById }} onSelect={vi.fn()} />)
    const labelCalls = mock.mock.calls.filter((call) => String(call[0]).endsWith('/map/labels'))
    expect(labelCalls).toHaveLength(1)
  })

  it('keeps numbered clusters and states the code when names are unavailable', async () => {
    const mock = stubMap(mapPayload(), 200, {
      labels: {},
      unavailable: { code: 'insufficient_quota', message: 'credit balance exhausted' },
    })

    render(<MapView collectionId="c1" papersById={papersById} onSelect={vi.fn()} />)

    expect(await screen.findByText(/Cluster names unavailable/)).toBeVisible()
    expect(screen.getByText(/quota exhausted/)).toBeVisible()
    expect(screen.getByText('Cluster 0 · 2 papers')).toBeVisible()

    await new Promise((resolve) => setTimeout(resolve, 50))
    const labelCalls = mock.mock.calls.filter((call) => String(call[0]).endsWith('/map/labels'))
    expect(labelCalls).toHaveLength(1)
  })

  it('drafts related work for a cluster and surfaces the draft’s warnings', async () => {
    const mock = stubMap(mapPayload(), 200, undefined, (url) => {
      if (url !== '/api/collections/c1/related-work') return undefined
      return fakeResponse({
        paragraph: 'Early work framed retrieval as ranking [Lee, 2019]. Later work [Kim, 2030].',
        citations: [
          { key: 'Lee, 2019', paper_id: 'p1', title: 'Paper One', year: 2019, url: 'https://arxiv.org/abs/1' },
          { key: 'Park, 2021', paper_id: 'p2', title: 'Paper Two', year: 2021, url: null },
        ],
        unknown_citations: ['Kim, 2030'],
        uncited_papers: ['Park, 2021'],
        word_count: 12,
        ordered_by: 'year, then citation count',
        ai_assisted_draft: true,
        label: 'AI-assisted draft - verify every sentence against its source before use',
        model: 'gpt-5-mini',
        cached: false,
      })
    })
    const user = userEvent.setup()

    render(
      <MapView collectionId="c1" papersById={papersById} onSelect={vi.fn()} generationAvailable />,
    )

    await user.click(
      await screen.findByRole('button', { name: 'Draft related work for Cluster 0' }),
    )

    expect(await screen.findByText(/Verify before use/)).toBeVisible()
    const call = mock.mock.calls.find((entry) => String(entry[0]).endsWith('/related-work'))
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({ cluster_id: 0 })
    expect(screen.getByText('Unknown citations:')).toBeVisible()
    expect(screen.getByText('[Kim, 2030]')).toBeVisible()
    expect(screen.getByText('Not cited:')).toBeVisible()
    expect(screen.getByRole('link', { name: 'Paper One' })).toHaveAttribute('href', 'https://arxiv.org/abs/1')

    expect(screen.queryByRole('link', { name: 'Paper Two' })).toBeNull()
  })

  it('offers no draft action when generation is unavailable', async () => {
    stubMap(mapPayload())

    render(<MapView collectionId="c1" papersById={papersById} onSelect={vi.fn()} />)

    await screen.findByText('Cluster 0 · 2 papers')
    expect(screen.queryByRole('button', { name: /Draft related work/ })).toBeNull()
    expect(screen.queryByRole('button', { name: /Compare papers/ })).toBeNull()
  })

  it('opens a cluster comparison that waits for an explicit run', async () => {
    const posted: unknown[] = []
    const mock = stubMap(mapPayload(), 200, { labels: {}, unavailable: null }, (url, init) => {
      if (url !== '/api/collections/c1/contradictions') return undefined
      posted.push(JSON.parse(String(init?.body)))
      return fakeResponse({
        comparisons: [],
        n_pairs_requested: 1,
        unavailable: null,
        method: 'pairs chosen by map similarity',
        note: 'A verdict is a lead to check, not a finding.',
        ai_generated: true,
      })
    })
    const user = userEvent.setup()

    render(
      <MapView collectionId="c1" papersById={papersById} onSelect={vi.fn()} generationAvailable />,
    )
    await user.click(await screen.findByRole('button', { name: 'Compare papers in Cluster 0' }))

    expect(
      screen.getByRole('heading', { name: /Compare papers · Cluster 0 \(papers nearest its centre, at most 8\)/ }),
    ).toBeVisible()

    expect(screen.getByText(/model calls/)).toHaveTextContent('Up to 2 model calls.')
    expect(mock.mock.calls.some((call) => String(call[0]).endsWith('/contradictions'))).toBe(false)

    await user.click(screen.getByRole('button', { name: 'Run comparison' }))
    expect(await screen.findByText('No pairs were compared.')).toBeVisible()
    expect(posted).toEqual([{ cluster_id: 0, max_pairs: 1 }])
  })
})
