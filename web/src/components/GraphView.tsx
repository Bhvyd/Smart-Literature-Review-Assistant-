import { useEffect, useMemo, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'

import { getGraph, messageFor } from '../api/client'
import type { GraphNode, GraphPayload, GraphViewEntry } from '../api/types'
import { formatCount } from '../lib/format'
import {
  describeClusterReason,
  nodeCluster,
  openAlexUrl,
  readClusters,
  untitledOpenAlexId,
} from '../lib/graphData'
import { layoutGraph, trimToRim } from '../lib/graphLayout'
import { clusterColour } from '../lib/mapScale'
import { Banner, ErrorState, LoadingState } from './States'

const WIDTH = 900
const HEIGHT = 600
const NODE_RADIUS = 5
const MIN_RADIUS = 4
const MAX_RADIUS = 13
const EXTERNAL_RADIUS = 3

type Tab = 'pagerank' | 'authority' | 'hub'

const TABS: { id: Tab; label: string; explain: string }[] = [
  {
    id: 'pagerank',
    label: 'Most influential',
    explain: 'Cited often, by papers that are themselves cited (PageRank).',
  },
  {
    id: 'authority',
    label: 'Foundational',
    explain: 'Works this collection builds on (HITS authority).',
  },
  {
    id: 'hub',
    label: 'Best surveys to start with',
    explain: 'Papers whose references cover the foundations (HITS hub).',
  },
]

const NOTE_COPY: Record<string, string> = {
  hits_did_not_converge: 'Hub and authority scores unavailable (HITS did not converge).',
  no_edges: 'No citation links, so no hub or authority scores.',
}

interface GraphViewProps {
  collectionId: string
  includeExternal: boolean
  onToggleExternal: (next: boolean) => void
  onSelect: (paperId: string) => void
}

function nodeRadius(node: GraphNode, range: { min: number; max: number }): number {
  if (!node.resolved) return EXTERNAL_RADIUS
  if (typeof node.pagerank !== 'number') return NODE_RADIUS
  const span = range.max - range.min
  if (span <= 0) return NODE_RADIUS
  return MIN_RADIUS + (MAX_RADIUS - MIN_RADIUS) * Math.sqrt((node.pagerank - range.min) / span)
}

function displayTitle(node: GraphNode): string {
  const openalexId = untitledOpenAlexId(node.id, node.title)
  return openalexId ? `Untitled reference ${openalexId}` : node.title
}

function describeNode(node: GraphNode, coloured: boolean): string {
  if (!node.resolved) return `${displayTitle(node)}, reference outside this collection`
  const cluster = nodeCluster(node)
  const group =
    cluster === null
      ? coloured
        ? ', not on the map'
        : ''
      : cluster < 0
        ? ', unclustered'
        : `, cluster ${cluster}`
  return `${node.title}, ${formatCount(node.citation_count)} citations, ${node.year ?? 'year unknown'}${group}`
}

function nodeFill(node: GraphNode, clusterIds: number[] | null): string {
  if (!node.resolved) return 'var(--surface)'
  const cluster = nodeCluster(node)
  if (clusterIds === null) return 'var(--accent)'

  if (cluster === null) return 'var(--surface)'
  return clusterColour(cluster, clusterIds)
}

export function GraphView({
  collectionId,
  includeExternal,
  onToggleExternal,
  onSelect,
}: GraphViewProps) {
  const [payload, setPayload] = useState<GraphPayload | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [loading, setLoading] = useState(true)
  const [zoom, setZoom] = useState(1)
  const [tab, setTab] = useState<Tab>('pagerank')
  const [fromYear, setFromYear] = useState('')
  const [toYear, setToYear] = useState('')
  const [focusId, setFocusId] = useState<string | null>(null)
  const tabRefs = useRef<Record<Tab, HTMLButtonElement | null>>({
    pagerank: null,
    authority: null,
    hub: null,
  })

  const loadKey = `${collectionId}|${includeExternal}`
  const [loadedFor, setLoadedFor] = useState(loadKey)
  if (loadedFor !== loadKey) {
    setLoadedFor(loadKey)
    setLoading(true)
    setError(null)
    setPayload(null)
  }

  useEffect(() => {
    let alive = true
    getGraph(collectionId, includeExternal)
      .then((next) => {
        if (alive) {
          setPayload(next)
          setLoading(false)
        }
      })
      .catch((caught: unknown) => {
        if (alive) {
          setError(caught)
          setLoading(false)
        }
      })
    return () => {
      alive = false
    }
  }, [collectionId, includeExternal])

  const positions = useMemo(
    () => layoutGraph(payload?.nodes ?? [], payload?.edges ?? [], { width: WIDTH, height: HEIGHT }),
    [payload],
  )

  if (loading) return <LoadingState label="Reading citation edges" />
  if (error) return <ErrorState message={messageFor(error)} />
  if (!payload) return null

  const byId = new Map(payload.nodes.map((node) => [node.id, node]))
  const externalCount = payload.nodes.filter((node) => !node.resolved).length
  const halfWidth = WIDTH / (2 * zoom)
  const halfHeight = HEIGHT / (2 * zoom)
  const viewBox = `${WIDTH / 2 - halfWidth} ${HEIGHT / 2 - halfHeight} ${WIDTH / zoom} ${HEIGHT / zoom}`

  const ranks = payload.nodes.flatMap((node) =>
    node.resolved && typeof node.pagerank === 'number' ? [node.pagerank] : [],
  )
  const rankRange = ranks.length
    ? { min: Math.min(...ranks), max: Math.max(...ranks) }
    : { min: 0, max: 0 }

  const years = payload.nodes.flatMap((node) => (node.year === null ? [] : [node.year]))
  const minYear = years.length ? Math.min(...years) : null
  const maxYear = years.length ? Math.max(...years) : null
  const undated = payload.nodes.length - years.length
  const from = fromYear === '' ? null : Number(fromYear)
  const to = toYear === '' ? null : Number(toYear)
  const inRange = (node: GraphNode) =>
    node.year === null ||
    ((from === null || node.year >= from) && (to === null || node.year <= to))
  const visible = new Set(payload.nodes.filter(inRange).map((node) => node.id))
  const hiddenByYear = payload.nodes.length - visible.size

  const views = payload.views && 'available' in payload.views ? payload.views : null
  const focused = focusId ? byId.get(focusId) ?? null : null
  const components = payload.components ?? {}

  const clusters = readClusters(payload)
  const clusterIds = clusters?.available ? clusters.ids : null
  const clusterCounts = new Map<number, number>()
  let unplacedShown = 0
  if (clusterIds) {
    for (const node of payload.nodes) {
      if (!node.resolved || !visible.has(node.id)) continue
      const cluster = nodeCluster(node)
      if (cluster === null) unplacedShown += 1
      else clusterCounts.set(cluster, (clusterCounts.get(cluster) ?? 0) + 1)
    }
  }

  function onTabKey(event: KeyboardEvent<HTMLButtonElement>) {
    const order = TABS.map((item) => item.id)
    const index = order.indexOf(tab)
    let next: Tab | null = null
    if (event.key === 'ArrowRight') next = order[(index + 1) % order.length]
    if (event.key === 'ArrowLeft') next = order[(index - 1 + order.length) % order.length]
    if (event.key === 'Home') next = order[0]
    if (event.key === 'End') next = order[order.length - 1]
    if (next) {
      event.preventDefault()
      setTab(next)
      tabRefs.current[next]?.focus()
    }
  }

  function entryRow(entry: GraphViewEntry, index: number) {
    const untitled = entry.in_collection ? null : untitledOpenAlexId(entry.id, entry.title)
    return (
      <li className="ranking-row" key={entry.id}>
        <span className="mono muted">{index + 1}</span>
        {entry.in_collection ? (
          <button
            type="button"
            className="ranking-title ranking-link"
            onClick={() => onSelect(entry.id)}
          >
            {entry.title}
          </button>
        ) : (
          <span className="ranking-title ranking-external">
            {untitled ? (
              <>
                Untitled reference{' '}
                <a
                  className="ranking-id"
                  href={openAlexUrl(untitled)}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={`Open ${untitled} on OpenAlex (opens in a new tab)`}
                >
                  {untitled}
                </a>
              </>
            ) : (
              (entry.title ?? entry.id)
            )}{' '}
            <span className="badge">outside collection</span>
          </span>
        )}
        <span className="ranking-score">{entry.score.toPrecision(3)}</span>
      </li>
    )
  }

  return (
    <section>
      <div className="toolbar">
        <p className="muted map-method">
          <span className="mono">{payload.nodes.length}</span> nodes ·{' '}
          <span className="mono">{payload.edges.length}</span> links
        </p>
        <div className="chart-controls">
          <button
            type="button"
            className="button button-secondary"
            aria-pressed={includeExternal}
            onClick={() => onToggleExternal(!includeExternal)}
          >
            {includeExternal ? 'Hide external references' : 'Show external references'}
          </button>
          <button
            type="button"
            className="button button-secondary"
            onClick={() => setZoom((value) => Math.min(8, value + 0.5))}
          >
            Zoom in
          </button>
          <button
            type="button"
            className="button button-secondary"
            onClick={() => setZoom((value) => Math.max(1, value - 0.5))}
          >
            Zoom out
          </button>
          <button type="button" className="button button-secondary" onClick={() => setZoom(1)}>
            Reset
          </button>
        </div>
      </div>

      {payload.reason === 'not_enough_edges' ? (
        <Banner tone="warning" title="Not enough citations to draw a graph.">
          Needs {payload.needed ?? 'more'} citation links, this collection has{' '}
          {payload.have ?? 'fewer'}.
        </Banner>
      ) : null}

      {includeExternal && externalCount > 0 ? (
        <p className="muted map-note">
          <span className="mono">{externalCount}</span> of these are references outside the
          collection (dashed).
        </p>
      ) : null}

      {minYear !== null && maxYear !== null ? (
        <fieldset className="year-filter">
          <legend className="field-label">Year range</legend>
          <label>
            From
            <input
              className="input year-input"
              type="number"
              min={minYear}
              max={maxYear}
              placeholder={String(minYear)}
              value={fromYear}
              onChange={(event) => setFromYear(event.target.value)}
            />
          </label>
          <label>
            To
            <input
              className="input year-input"
              type="number"
              min={minYear}
              max={maxYear}
              placeholder={String(maxYear)}
              value={toYear}
              onChange={(event) => setToYear(event.target.value)}
            />
          </label>
          <p className="muted year-note" aria-live="polite">
            {hiddenByYear > 0 ? (
              <>
                <span className="mono">{hiddenByYear}</span> hidden
              </>
            ) : null}
            {hiddenByYear > 0 && undated > 0 ? ' · ' : null}
            {undated > 0 ? (
              <>
                <span className="mono">{undated}</span> undated, always shown
              </>
            ) : null}
          </p>
        </fieldset>
      ) : null}

      {typeof components.full === 'number' ? (
        <p className="muted map-note">
          <span className="mono">{components.full}</span> connected{' '}
          {components.full === 1 ? 'component' : 'components'}
        </p>
      ) : null}

      {clusterIds ? (
        <ul className="legend" aria-label="Cluster colours, shared with the Map">
          {clusterIds.map((cluster) => (
            <li key={cluster} className="legend-item">
              <span
                className="legend-swatch"
                style={{ background: clusterColour(cluster, clusterIds) }}
                aria-hidden="true"
              />
              Cluster {cluster} · {clusterCounts.get(cluster) ?? 0} papers
            </li>
          ))}
          {unplacedShown > 0 ? (
            <li className="legend-item">
              <span className="legend-swatch unplaced" aria-hidden="true" />
              Not on the map · {unplacedShown} papers
            </li>
          ) : null}
          {(clusterCounts.get(-1) ?? 0) > 0 ? (
            <li className="legend-item">
              <span className="legend-swatch noise" aria-hidden="true" />
              Unclustered · {clusterCounts.get(-1)} papers
            </li>
          ) : null}
          <li className="legend-item legend-hint">Cluster names are on the Map.</li>
        </ul>
      ) : clusters ? (
        <p className="muted map-note">{describeClusterReason(clusters.reason)}</p>
      ) : null}

      <div className="graph-layout">
        <div>
          <div className="chart-frame">
            <svg
              viewBox={viewBox}
              role="group"
              aria-label={`Citation graph with ${payload.nodes.length} nodes and ${payload.edges.length} edges`}
            >
              <defs>
                <marker
                  id="graph-arrow"
                  viewBox="0 0 8 8"
                  refX="7"
                  refY="4"
                  markerWidth="7"
                  markerHeight="7"
                  orient="auto-start-reverse"
                >
                  <path d="M0,0 L8,4 L0,8 z" fill="var(--muted)" />
                </marker>
              </defs>

              {payload.edges.map((edge) => {
                if (!visible.has(edge.source) || !visible.has(edge.target)) return null
                const from = positions.get(edge.source)
                const to = positions.get(edge.target)
                const target = byId.get(edge.target)
                if (!from || !to) return null
                const radius = target ? nodeRadius(target, rankRange) : NODE_RADIUS
                const end = trimToRim(from, to, radius + 3)
                return (
                  <line
                    key={`${edge.source}->${edge.target}`}
                    className="graph-edge"
                    data-source={edge.source}
                    data-target={edge.target}
                    x1={from.x}
                    y1={from.y}
                    x2={end.x}
                    y2={end.y}
                    stroke={edge.resolved ? 'var(--hairline-strong)' : 'var(--noise)'}
                    strokeWidth={edge.resolved ? 1.2 : 0.8}
                    markerEnd="url(#graph-arrow)"
                  />
                )
              })}

              {payload.nodes.map((node) => {
                if (!visible.has(node.id)) return null
                const position = positions.get(node.id)
                if (!position) return null
                const label = describeNode(node, clusterIds !== null)
                const unplaced = clusterIds !== null && node.resolved && nodeCluster(node) === null
                return (
                  <circle
                    key={node.id}
                    className={`graph-node${node.resolved ? '' : ' external'}`}
                    data-id={node.id}
                    cx={position.x}
                    cy={position.y}
                    r={nodeRadius(node, rankRange)}
                    style={{
                      fill: nodeFill(node, clusterIds),
                      stroke: node.resolved && !unplaced ? 'none' : 'var(--noise)',
                    }}
                    strokeWidth={node.resolved ? (unplaced ? 1.5 : 0) : 1}
                    strokeDasharray={node.resolved ? undefined : '2 2'}
                    tabIndex={node.resolved ? 0 : undefined}
                    role={node.resolved ? 'button' : undefined}
                    aria-label={label}
                    onMouseEnter={() => setFocusId(node.id)}
                    onMouseLeave={() => setFocusId(null)}
                    onFocus={() => setFocusId(node.id)}
                    onBlur={() => setFocusId(null)}
                    onClick={() => {
                      if (node.resolved) onSelect(node.id)
                    }}
                    onKeyDown={(event) => {
                      if (node.resolved && (event.key === 'Enter' || event.key === ' ')) {
                        event.preventDefault()
                        onSelect(node.id)
                      }
                    }}
                  >
                    <title>{label}</title>
                  </circle>
                )
              })}
            </svg>
          </div>
          <p className="map-readout" aria-live="polite">
            {focused ? (
              <>
                <span className="paper-title-inline">{displayTitle(focused)}</span>
                <span className="muted">
                  {' '}
                  · {focused.year ?? 'year unknown'} ·{' '}
                  {focused.resolved
                    ? `${formatCount(focused.citation_count)} citations`
                    : 'reference outside this collection'}
                </span>
              </>
            ) : (
              <span className="muted">Hover a node for details. Click a paper to open it.</span>
            )}
          </p>
        </div>

        <div className="rankings">
          {views && views.available ? (
            <>
              <div className="tabs" role="tablist" aria-label="Ranked views">
                {TABS.map((item) => (
                  <button
                    key={item.id}
                    ref={(element) => {
                      tabRefs.current[item.id] = element
                    }}
                    type="button"
                    role="tab"
                    id={`graph-tab-${item.id}`}
                    aria-selected={tab === item.id}
                    aria-controls="graph-tabpanel"
                    tabIndex={tab === item.id ? 0 : -1}
                    className="tab"
                    onClick={() => setTab(item.id)}
                    onKeyDown={onTabKey}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
              <div role="tabpanel" id="graph-tabpanel" aria-labelledby={`graph-tab-${tab}`}>
                <p className="muted ranking-note">
                  {TABS.find((item) => item.id === tab)?.explain}
                </p>
                {views[tab].length === 0 ? (
                  <p className="muted ranking-note">No papers in this view.</p>
                ) : (
                  <ol className="ranking-list">{views[tab].map(entryRow)}</ol>
                )}
              </div>
              {views.notes.map((note) => (
                <p key={note} className="ranking-note warn-text">
                  {NOTE_COPY[note] ?? note}
                </p>
              ))}
              <p className="muted ranking-note ranking-method">
                Method: {views.method.pagerank}; {views.method.hits}.
              </p>
            </>
          ) : (
            <>
              <h2>Influence</h2>
              {views && !views.available ? (
                <p className="ranking-note warn-text">
                  Rankings unavailable (<span className="mono">{views.reason}</span>).
                </p>
              ) : null}
              <p className="muted ranking-note">Weighted by how cited the citing papers are.</p>
              {payload.rankings.slice(0, 20).map((row, index) => (
                <div className="ranking-row" key={row.paper_id}>
                  <span className="mono muted">{index + 1}</span>
                  <button
                    type="button"
                    className="ranking-title ranking-link"
                    onClick={() => onSelect(row.paper_id)}
                  >
                    {byId.get(row.paper_id)?.title ?? row.paper_id}
                  </button>
                  <span className="ranking-score">
                    {row.score.toFixed(2)}
                    {row.complete ? '' : ' · lower bound'}
                  </span>
                </div>
              ))}
              {payload.rankings.length === 0 ? (
                <p className="muted ranking-note">No influence ranking for this collection.</p>
              ) : null}
            </>
          )}
        </div>
      </div>
    </section>
  )
}
