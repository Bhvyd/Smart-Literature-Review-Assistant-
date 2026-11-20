import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { GraphPayload } from '../api/types'
import { GraphView } from './GraphView'

function fakeResponse(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => JSON.stringify(body),
  } as unknown as Response
}

function payload(overrides: Partial<GraphPayload> = {}): GraphPayload {
  return {
    nodes: [
      { id: 'a', title: 'Paper A', year: 2020, citation_count: 10, resolved: true },
      { id: 'b', title: 'Paper B', year: 2021, citation_count: 40, resolved: true },
      { id: 'c', title: 'Paper C', year: 2022, citation_count: 5, resolved: false },
    ],
    edges: [
      { source: 'a', target: 'b', resolved: true },
      { source: 'b', target: 'c', resolved: false },
    ],
    rankings: [
      { paper_id: 'b', score: 40, complete: true },
      { paper_id: 'a', score: 10, complete: false },
    ],
    unidentified: 0,
    ...overrides,
  }
}

function stubGraph(body: unknown, status = 200) {
  const mock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)
    if (!url.startsWith('/api/collections/c1/graph')) {
      throw new Error(`unexpected request: ${url}`)
    }
    return fakeResponse(body, status)
  })
  vi.stubGlobal('fetch', mock)
  return mock
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('citation graph view', () => {
  it('draws an arrowed edge from the citing paper to the cited one', async () => {
    stubGraph(payload())
    const onSelect = vi.fn()
    const user = userEvent.setup()

    const { container } = render(
      <GraphView
        collectionId="c1"
        includeExternal={false}
        onToggleExternal={vi.fn()}
        onSelect={onSelect}
      />,
    )

    await screen.findByRole('group', { name: /Citation graph with 3 nodes and 2 edges/ })
    const edges = container.querySelectorAll('line.graph-edge')
    expect(edges).toHaveLength(2)
    for (const edge of edges) {
      expect(edge.getAttribute('marker-end')).toBe('url(#graph-arrow)')
    }
    expect(edges[0].getAttribute('data-source')).toBe('a')
    expect(edges[0].getAttribute('data-target')).toBe('b')

    await user.click(screen.getByRole('button', { name: /Paper A, 10 citations/ }))
    expect(onSelect).toHaveBeenCalledWith('a')
  })

  it('lists influence in the order the API returned it', async () => {
    stubGraph(payload())

    render(
      <GraphView
        collectionId="c1"
        includeExternal={false}
        onToggleExternal={vi.fn()}
        onSelect={vi.fn()}
      />,
    )

    const rows = await screen.findAllByRole('button', { name: /Paper [AB]$/ })
    const titles = rows.map((row) => row.textContent)
    expect(titles).toEqual(['Paper B', 'Paper A'])
    expect(screen.getByText(/lower bound/)).toBeVisible()
  })

  it('explains an edge threshold instead of drawing an empty canvas', async () => {
    stubGraph(
      payload({
        nodes: [],
        edges: [],
        rankings: [],
        reason: 'not_enough_edges',
        needed: 5,
        have: 2,
      }),
    )

    render(
      <GraphView
        collectionId="c1"
        includeExternal={false}
        onToggleExternal={vi.fn()}
        onSelect={vi.fn()}
      />,
    )

    expect(await screen.findByText('Not enough citations to draw a graph.')).toBeVisible()
    expect(screen.getByText(/Needs 5 citation links, this collection has 2/)).toBeVisible()
  })

  it('asks the parent to include external references', async () => {
    stubGraph(payload())
    const onToggleExternal = vi.fn()
    const user = userEvent.setup()

    render(
      <GraphView
        collectionId="c1"
        includeExternal={false}
        onToggleExternal={onToggleExternal}
        onSelect={vi.fn()}
      />,
    )

    await user.click(await screen.findByRole('button', { name: 'Show external references' }))
    expect(onToggleExternal).toHaveBeenCalledWith(true)
  })

  it('requests external nodes only when the toggle is on', async () => {
    const mock = stubGraph(payload())

    render(
      <GraphView
        collectionId="c1"
        includeExternal
        onToggleExternal={vi.fn()}
        onSelect={vi.fn()}
      />,
    )

    await screen.findByRole('group', { name: /Citation graph/ })
    expect(String(mock.mock.calls[0][0])).toContain('include_external=true')
    expect(await screen.findByRole('button', { name: 'Hide external references' })).toBeVisible()
  })
})

function withViews(overrides: Partial<GraphPayload> = {}): GraphPayload {
  return payload({
    nodes: [
      { id: 'a', title: 'Paper A', year: 2020, citation_count: 10, resolved: true, pagerank: 0.02 },
      { id: 'b', title: 'Paper B', year: 2021, citation_count: null, resolved: true, pagerank: 0.08 },
      { id: 'd', title: 'Paper D', year: null, citation_count: 3, resolved: true, pagerank: null },
    ],
    edges: [
      { source: 'a', target: 'b', resolved: true },
      { source: 'd', target: 'b', resolved: true },
    ],
    views: {
      available: true,
      pagerank: [
        { id: 'ext:W1', title: 'Attention Is All You Need', in_collection: false, score: 0.12 },
        { id: 'b', title: 'Paper B', in_collection: true, score: 0.08 },
      ],
      authority: [
        { id: 'ext:W2', title: 'BERT', in_collection: false, score: 0.4 },
        { id: 'a', title: 'Paper A', in_collection: true, score: 0.1 },
      ],
      hub: [{ id: 'd', title: 'Paper D', in_collection: true, score: 0.7 }],
      notes: [],
      method: {
        pagerank: 'networkx.pagerank alpha=0.85',
        hits: 'networkx.hits (Kleinberg 1999); can drift toward a densely cited tangent',
      },
    },
    components: { full: 3, internal: 2 },
    ...overrides,
  })
}

function renderGraph(onSelect = vi.fn()) {
  return render(
    <GraphView
      collectionId="c1"
      includeExternal={false}
      onToggleExternal={vi.fn()}
      onSelect={onSelect}
    />,
  )
}

describe('citation graph ranked views', () => {
  it('switches between the three ranked views with the keyboard', async () => {
    stubGraph(withViews())
    const user = userEvent.setup()
    renderGraph()

    const influential = await screen.findByRole('tab', { name: 'Most influential' })
    expect(influential).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tabpanel')).toHaveTextContent('Attention Is All You Need')

    influential.focus()
    await user.keyboard('{ArrowRight}')
    const foundational = screen.getByRole('tab', { name: 'Foundational' })
    expect(foundational).toHaveFocus()
    expect(foundational).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tabpanel')).toHaveTextContent('BERT')

    await user.click(screen.getByRole('tab', { name: 'Best surveys to start with' }))
    expect(screen.getByRole('tabpanel')).toHaveTextContent('Paper D')

    expect(screen.getAllByText(/can drift toward a densely cited tangent/)).toHaveLength(1)
  })

  it('labels references outside the collection and does not open them', async () => {
    stubGraph(withViews())
    const onSelect = vi.fn()
    const user = userEvent.setup()
    renderGraph(onSelect)

    const panel = await screen.findByRole('tabpanel')
    expect(within(panel).getByText('outside collection')).toBeVisible()
    expect(within(panel).queryByRole('button', { name: /Attention Is All You Need/ })).toBeNull()

    await user.click(within(panel).getByRole('button', { name: 'Paper B' }))
    expect(onSelect).toHaveBeenCalledWith('b')
  })

  it('reads an unknown citation count as unknown, never as zero', async () => {
    stubGraph(withViews())
    renderGraph()

    const node = await screen.findByRole('button', { name: /^Paper B,/ })
    expect(node).toHaveAccessibleName('Paper B, unknown citations, 2021')
    expect(screen.queryByRole('button', { name: /Paper B, 0 citations/ })).toBeNull()
  })

  it('sizes collection nodes by PageRank and keeps unknown PageRank at the default size', async () => {
    stubGraph(withViews())
    const { container } = renderGraph()
    await screen.findByRole('group', { name: /Citation graph/ })

    const radius = (id: string) =>
      Number(container.querySelector(`circle[data-id="${id}"]`)?.getAttribute('r'))
    expect(radius('b')).toBeGreaterThan(radius('a'))
    expect(radius('a')).toBeGreaterThanOrEqual(4)
    expect(radius('d')).toBe(5)
  })

  it('filters by year, keeps undated nodes, and says so', async () => {
    stubGraph(withViews())
    const user = userEvent.setup()
    const { container } = renderGraph()
    await screen.findByRole('group', { name: /Citation graph/ })

    expect(screen.getByText(/undated, always shown/)).toBeVisible()
    await user.type(screen.getByLabelText('From'), '2021')

    expect(container.querySelector('circle[data-id="a"]')).toBeNull()
    expect(container.querySelector('circle[data-id="b"]')).not.toBeNull()
    expect(container.querySelector('circle[data-id="d"]')).not.toBeNull()
    expect(container.querySelector('.year-note')).toHaveTextContent(/^\d+ hidden · \d+ undated, always shown$/)
  })

  it('reports connected components instead of hiding disconnected papers', async () => {
    stubGraph(withViews())
    renderGraph()

    expect(await screen.findByText(/connected components/)).toBeVisible()
  })

  it('says why the ranked views are missing when the API reports them unavailable', async () => {
    stubGraph(withViews({ views: { available: false, reason: 'networkx_missing' } }))
    renderGraph()

    expect(await screen.findByText(/Rankings unavailable/)).toBeVisible()
    expect(screen.getByText('networkx_missing')).toBeVisible()
  })
})

describe('citation graph cluster colour', () => {
  const clustered = (): GraphPayload =>
    withViews({
      nodes: [
        { id: 'a', title: 'Paper A', year: 2020, citation_count: 10, resolved: true, cluster: 0 },
        { id: 'b', title: 'Paper B', year: 2021, citation_count: 40, resolved: true, cluster: 2 },
        { id: 'd', title: 'Paper D', year: 2022, citation_count: 3, resolved: true, cluster: -1 },
        { id: 'ext:W1', title: 'Outside', year: null, citation_count: null, resolved: false },
      ],
      edges: [{ source: 'a', target: 'b', resolved: true }],
      clusters: { available: true, reason: null, ids: [0, 2] },
    } as Partial<GraphPayload>)

  it('colours collection nodes with the Map’s cluster palette and leaves outside references neutral', async () => {
    stubGraph(clustered())
    const { container } = renderGraph()
    await screen.findByRole('group', { name: /Citation graph/ })

    const fill = (id: string) =>
      (container.querySelector(`circle[data-id="${id}"]`) as SVGElement).style.fill
    expect(fill('a')).toBe('var(--cluster-1)')
    expect(fill('b')).toBe('var(--cluster-2)')
    expect(fill('d')).toBe('var(--noise)')
    expect(fill('ext:W1')).toBe('var(--surface)')
  })

  it('states the cluster in the accessible name, so colour is not the only signal', async () => {
    stubGraph(clustered())
    renderGraph()
    expect(await screen.findByRole('button', { name: /^Paper A,.*, cluster 0$/ })).toBeVisible()
    expect(screen.getByRole('button', { name: /^Paper D,.*, unclustered$/ })).toBeVisible()
  })

  it('shows a legend with counts, and says names live on the Map', async () => {
    stubGraph(clustered())
    renderGraph()
    expect(await screen.findByText('Cluster 0 · 1 papers')).toBeVisible()
    expect(screen.getByText('Cluster 2 · 1 papers')).toBeVisible()
    expect(screen.getByText('Unclustered · 1 papers')).toBeVisible()
    expect(screen.getByText(/Cluster names are on the Map/)).toBeVisible()
  })

  it.each([
    ['map_not_computed', /Open the Map once to colour nodes by cluster/],
    ['clusters_unreliable', /the clustering was unreliable/],
  ])('keeps the default colour and says why when clusters are unavailable (%s)', async (reason, copy) => {
    stubGraph(
      withViews({ clusters: { available: false, reason, ids: [], unplaced: 0 } } as Partial<GraphPayload>),
    )
    const { container } = renderGraph()

    expect(await screen.findByText(copy)).toBeVisible()
    expect(screen.queryByRole('list', { name: /Cluster colours/ })).toBeNull()
    const node = container.querySelector('circle[data-id="a"]') as SVGElement
    expect(node.style.fill).toBe('var(--accent)')
  })

  it('never colours by cluster when an older server sends no cluster fields', async () => {
    stubGraph(withViews())
    const { container } = renderGraph()
    await screen.findByRole('group', { name: /Citation graph/ })
    expect((container.querySelector('circle[data-id="a"]') as SVGElement).style.fill).toBe(
      'var(--accent)',
    )
    expect(screen.queryByText(/Open the Map once/)).toBeNull()
  })
})

describe('untitled external references', () => {
  it('says "Untitled reference" and links the OpenAlex id instead of showing the id as a title', async () => {
    stubGraph(
      withViews({
        views: {
          available: true,
          pagerank: [
            { id: 'ext:W6948136082', title: 'W6948136082', in_collection: false, score: 0.12 },
            { id: 'ext:W2', title: 'BERT', in_collection: false, score: 0.1 },
          ],
          authority: [],
          hub: [],
          notes: [],
          method: { pagerank: 'p', hits: 'h' },
        },
      }),
    )
    renderGraph()

    const panel = await screen.findByRole('tabpanel')
    const link = within(panel).getByRole('link', { name: /Open W6948136082 on OpenAlex/ })
    expect(link).toHaveAttribute('href', 'https://openalex.org/W6948136082')
    expect(link).toHaveTextContent('W6948136082')
    expect(panel).toHaveTextContent('Untitled reference W6948136082')

    expect(within(panel).getByText(/BERT/)).toBeVisible()
    expect(within(panel).getAllByRole('link')).toHaveLength(1)
  })
})

describe('citation graph: unplaced papers, filters and references', () => {
  const withUnplaced = (): GraphPayload =>
    withViews({
      nodes: [
        { id: 'a', title: 'Paper A', year: 2020, citation_count: 10, resolved: true, cluster: 0 },
        { id: 'b', title: 'Paper B', year: 2021, citation_count: 40, resolved: true, cluster: 0 },
        { id: 'd', title: 'Paper D', year: 2022, citation_count: 3, resolved: true, cluster: null },
        { id: 'ext:W1', title: 'W1', year: null, citation_count: null, resolved: false },
      ],
      edges: [{ source: 'a', target: 'b', resolved: true }],
      clusters: { available: true, reason: null, ids: [0], unplaced: 1 },
    } as Partial<GraphPayload>)

  it('draws a paper missing from the cached map hollow, not in any cluster colour or the default one', async () => {
    stubGraph(withUnplaced())
    const { container } = renderGraph()
    await screen.findByRole('group', { name: /Citation graph/ })

    const node = container.querySelector('circle[data-id="d"]') as SVGElement
    expect(node.style.fill).toBe('var(--surface)')
    expect(node.style.stroke).toBe('var(--noise)')
    expect(node.getAttribute('stroke-dasharray')).toBeNull()
    expect(screen.getByRole('button', { name: /^Paper D,.*, not on the map$/ })).toBeVisible()
    expect(screen.getByText('Not on the map · 1 papers')).toBeVisible()
    expect(screen.getByText('Cluster 0 · 2 papers')).toBeVisible()
  })

  it('counts the legend over the nodes the year filter leaves visible', async () => {
    stubGraph(withUnplaced())
    const user = userEvent.setup()
    renderGraph()
    expect(await screen.findByText('Cluster 0 · 2 papers')).toBeVisible()

    await user.type(screen.getByLabelText('From'), '2021')
    expect(screen.getByText('Cluster 0 · 1 papers')).toBeVisible()
    expect(screen.getByText('Not on the map · 1 papers')).toBeVisible()
    await user.type(screen.getByLabelText('From'), '{Backspace}{Backspace}{Backspace}{Backspace}2023')
    expect(screen.getByText('Cluster 0 · 0 papers')).toBeVisible()
    expect(screen.queryByText(/^Not on the map/)).toBeNull()
  })

  it('opens an untitled reference on OpenAlex in a new tab without leaking the opener', async () => {
    stubGraph(
      withViews({
        views: {
          available: true,
          pagerank: [{ id: 'ext:W6948136082', title: 'W6948136082', in_collection: false, score: 0.1 }],
          authority: [],
          hub: [],
          notes: [],
          method: { pagerank: 'p', hits: 'h' },
        },
      }),
    )
    renderGraph()
    const link = await screen.findByRole('link', { name: /Open W6948136082 on OpenAlex/ })
    expect(link).toHaveAttribute('href', 'https://openalex.org/W6948136082')
    expect(link).toHaveAttribute('target', '_blank')
    const rel = (link.getAttribute('rel') ?? '').split(' ')
    expect(rel).toEqual(expect.arrayContaining(['noopener', 'noreferrer']))
  })

  it('reads "Untitled reference" in the readout and the node name, not a bare id', async () => {
    stubGraph(
      withViews({
        nodes: [
          { id: 'a', title: 'Paper A', year: 2020, citation_count: 10, resolved: true },
          { id: 'ext:W6948136082', title: 'W6948136082', year: null, citation_count: null, resolved: false },
        ],
        edges: [{ source: 'a', target: 'ext:W6948136082', resolved: false }],
      }),
    )
    const { container } = render(
      <GraphView collectionId="c1" includeExternal onToggleExternal={vi.fn()} onSelect={vi.fn()} />,
    )
    await screen.findByRole('group', { name: /Citation graph/ })
    const node = container.querySelector('circle[data-id="ext:W6948136082"]') as SVGElement
    expect(node.getAttribute('aria-label')).toMatch(/^Untitled reference W6948136082, reference outside/)

    fireEvent.mouseEnter(node)
    const readout = container.querySelector('.map-readout') as HTMLElement
    expect(readout.textContent).toMatch(/^Untitled reference W6948136082/)
    expect(readout.textContent).not.toMatch(/^W6948136082/)
  })

  it('says why an empty stored map cannot colour the graph', async () => {
    stubGraph(
      withViews({
        clusters: { available: false, reason: 'map_empty', ids: [], unplaced: 0 },
      } as Partial<GraphPayload>),
    )
    renderGraph()
    expect(await screen.findByText(/the map is empty/)).toBeVisible()
  })
})
