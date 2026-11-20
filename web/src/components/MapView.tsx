import { useEffect, useMemo, useState } from 'react'

import { ApiError, getMap, getMapLabels, messageFor } from '../api/client'
import type { MapLabels, MapPayload, Paper } from '../api/types'
import { describeGenerationCode } from '../lib/format'
import { clusterColour, fitToViewport } from '../lib/mapScale'
import { COMPARE_MAX, ContradictionTable } from './ContradictionTable'
import { RelatedWorkDraft } from './RelatedWorkDraft'
import { Banner, ErrorState, LoadingState } from './States'

const WIDTH = 900
const HEIGHT = 560
const PADDING = 30

const VECTOR_SOURCE_COPY: Record<string, string> = {
  chunk_mean: 'placed from chunk vectors',
}

interface MapViewProps {
  collectionId: string
  papersById: Record<string, Paper>
  onSelect: (paperId: string) => void

  generationAvailable?: boolean
}

export function MapView({
  collectionId,
  papersById,
  onSelect,
  generationAvailable = false,
}: MapViewProps) {
  const [payload, setPayload] = useState<MapPayload | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [loading, setLoading] = useState(true)
  const [zoom, setZoom] = useState(1)
  const [focusId, setFocusId] = useState<string | null>(null)

  const [labels, setLabels] = useState<{ collectionId: string; value: MapLabels } | null>(null)
  const [labelsError, setLabelsError] = useState<unknown>(null)
  const [draftCluster, setDraftCluster] = useState<number | null>(null)
  const [compareCluster, setCompareCluster] = useState<number | null>(null)

  const [loadedFor, setLoadedFor] = useState(collectionId)
  if (loadedFor !== collectionId) {
    setLoadedFor(collectionId)
    setLoading(true)
    setError(null)
    setPayload(null)
    setZoom(1)
    setLabels(null)
    setLabelsError(null)
    setDraftCluster(null)
    setCompareCluster(null)
  }

  useEffect(() => {
    let alive = true
    getMap(collectionId)
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
  }, [collectionId])

  const fitted = useMemo(
    () => fitToViewport(payload?.points ?? [], WIDTH, HEIGHT, PADDING),
    [payload],
  )

  const clusters = useMemo(() => {
    const ids = new Set<number>()
    for (const point of payload?.points ?? []) ids.add(point.cluster)
    return [...ids].filter((id) => id >= 0).sort((a, b) => a - b)
  }, [payload])

  const reliable = payload !== null && payload.cluster_note !== 'clusters_unreliable'
  const wantLabels = reliable && clusters.length > 0
  useEffect(() => {
    if (!wantLabels) return
    let alive = true
    getMapLabels(collectionId)
      .then((value) => {
        if (alive) setLabels({ collectionId, value })
      })
      .catch((caught: unknown) => {
        if (alive) setLabelsError(caught)
      })
    return () => {
      alive = false
    }
  }, [collectionId, wantLabels])

  const counts = useMemo(() => {
    const result = new Map<number, number>()
    for (const point of payload?.points ?? []) {
      result.set(point.cluster, (result.get(point.cluster) ?? 0) + 1)
    }
    return result
  }, [payload])

  if (loading) return <LoadingState label="Projecting the collection" />

  if (error instanceof ApiError && error.code === 'map_needs_more_papers') {
    const detail = (error.detail ?? {}) as { needed?: number; have?: number }
    return (
      <Banner tone="warning" title="Not enough papers to map.">
        {detail.needed !== undefined && detail.have !== undefined
          ? `Needs ${detail.needed}, this collection has ${detail.have}.`
          : error.message}
      </Banner>
    )
  }

  if (error) return <ErrorState message={messageFor(error)} />
  if (!payload) return null

  const halfWidth = WIDTH / (2 * zoom)
  const halfHeight = HEIGHT / (2 * zoom)
  const viewBox = `${WIDTH / 2 - halfWidth} ${HEIGHT / 2 - halfHeight} ${WIDTH / zoom} ${HEIGHT / zoom}`
  const focused = focusId ? papersById[focusId] : null
  const noise = counts.get(-1) ?? 0
  const names: Record<string, string> =
    (labels?.collectionId === collectionId ? labels.value.labels : null) ?? {}
  const labelsUnavailable =
    (labels?.collectionId === collectionId ? labels.value.unavailable : null) ?? null
  const labelsFailureCode =
    labelsUnavailable?.code ??
    (labelsError instanceof ApiError ? labelsError.code : labelsError ? 'request_failed' : null)
  const clusterName = (cluster: number) => names[String(cluster)] ?? `Cluster ${cluster}`
  const extra: Partial<MapLabels> = labels?.collectionId === collectionId ? labels.value : {}
  const terms = extra.terms ?? {}

  const namesOverlap =
    labelsUnavailable === null &&
    ((extra.collisions?.length ?? 0) > 0 || (extra.uninformative?.length ?? 0) > 0)
  const unnamedClusters = labelsUnavailable === null ? (extra.unnamed ?? []) : []
  const vectorNote = payload.vector_source ? VECTOR_SOURCE_COPY[payload.vector_source] : null

  return (
    <section>
      <div className="toolbar">
        <p className="muted map-method">
          <span className="mono">{payload.points.length}</span> papers
          {vectorNote ? <> · {vectorNote}</> : null}
        </p>
        <div className="chart-controls">
          <button
            type="button"
            className="button button-secondary"
            onClick={() => setZoom((value) => Math.min(8, Math.round((value + 0.5) * 2) / 2))}
          >
            Zoom in
          </button>
          <button
            type="button"
            className="button button-secondary"
            onClick={() => setZoom((value) => Math.max(1, Math.round((value - 0.5) * 2) / 2))}
          >
            Zoom out
          </button>
          <button type="button" className="button button-secondary" onClick={() => setZoom(1)}>
            Reset
          </button>
        </div>
      </div>

      {payload.cluster_note === 'clusters_unreliable' ? (
        <Banner tone="warning" title="Clustering was unreliable.">
          Colours are not a meaningful grouping here.
        </Banner>
      ) : null}

      {payload.excluded > 0 ? (
        <Banner tone="warning">
          <span className="mono">{payload.excluded}</span>{' '}
          {payload.excluded === 1 ? 'paper is' : 'papers are'} not plotted (no vectors).
        </Banner>
      ) : null}

      <div className="map-layout">
        <div className="map-canvas">
          <div className="chart-frame">
            <svg
              viewBox={viewBox}
              role="group"
              aria-label={`Similarity map of ${payload.points.length} papers`}
            >
              {payload.points.map((point) => {
                const placed = fitted.project(point.x, point.y)
                const title = papersById[point.paper_id]?.title ?? point.paper_id
                return (
                  <circle
                    key={point.paper_id}
                    className={`map-point${point.cluster < 0 ? ' noise' : ''}`}
                    cx={placed.x}
                    cy={placed.y}
                    r={3}
                    style={{ fill: clusterColour(point.cluster, clusters) }}
                    stroke={point.cluster < 0 ? 'var(--noise)' : 'none'}
                    strokeWidth={point.cluster < 0 ? 1 : 0}
                    tabIndex={0}
                    role="button"
                    aria-label={`${title}, ${point.cluster < 0 ? 'unclustered' : clusterName(point.cluster)}`}
                    onMouseEnter={() => setFocusId(point.paper_id)}
                    onMouseLeave={() => setFocusId(null)}
                    onFocus={() => setFocusId(point.paper_id)}
                    onBlur={() => setFocusId(null)}
                    onClick={() => onSelect(point.paper_id)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault()
                        onSelect(point.paper_id)
                      }
                    }}
                  />
                )
              })}
            </svg>
          </div>

          <p className="map-readout" aria-live="polite">
            {focused ? (
              <>
                <span className="paper-title-inline">{focused.title}</span>
                {focused.year ? <span className="muted"> · {focused.year}</span> : null}
              </>
            ) : (
              <span className="muted">Hover a point for its title. Click to open it.</span>
            )}
          </p>
        </div>
        <aside className="map-side" aria-label="Clusters">
          {labelsFailureCode ? (
            <p className="muted map-note">
              Cluster names unavailable ({describeGenerationCode(labelsFailureCode)}).
            </p>
          ) : null}

          {namesOverlap ? (
            <p className="muted map-note">
              Some names overlap. The terms under each name show the difference.
            </p>
          ) : null}

          {unnamedClusters.length > 0 ? (
            <p className="muted map-note">
              {unnamedClusters.length === 1 ? 'One cluster is' : `${unnamedClusters.length} clusters are`}{' '}
              unnamed.
            </p>
          ) : null}

          <ul className="legend">
            {clusters.map((cluster) => (
              <li key={cluster} className="legend-item">
                <span
                  className="legend-swatch"
                  style={{ background: clusterColour(cluster, clusters) }}
                  aria-hidden="true"
                />
                <span className="legend-text">
                  {clusterName(cluster)} · {counts.get(cluster) ?? 0} papers
                  {terms[String(cluster)]?.length ? (
                    <span className="legend-terms">{terms[String(cluster)].join(', ')}</span>
                  ) : null}
                </span>
                {generationAvailable && reliable && (counts.get(cluster) ?? 0) >= 2 ? (
                  <>
                    <button
                      type="button"
                      className="legend-action"
                      aria-label={`Draft related work for ${clusterName(cluster)}`}
                      onClick={() => setDraftCluster(cluster)}
                    >
                      Draft related work
                    </button>
                    <button
                      type="button"
                      className="legend-action"
                      aria-label={`Compare papers in ${clusterName(cluster)}`}
                      onClick={() => setCompareCluster(cluster)}
                    >
                      Compare papers
                    </button>
                  </>
                ) : null}
              </li>
            ))}
            {noise > 0 ? (
              <li className="legend-item">
                <span className="legend-swatch noise" aria-hidden="true" />
                Unclustered · {noise} papers
              </li>
            ) : null}
          </ul>
        </aside>
      </div>

      {draftCluster !== null ? (
        <RelatedWorkDraft
          key={draftCluster}
          collectionId={collectionId}
          request={{ cluster_id: draftCluster }}
          subject={`${clusterName(draftCluster)} (papers nearest its centre, at most 12)`}
          onClose={() => setDraftCluster(null)}
        />
      ) : null}

      {compareCluster !== null ? (
        <ContradictionTable
          key={compareCluster}
          collectionId={collectionId}
          request={{ cluster_id: compareCluster }}
          subject={`${clusterName(compareCluster)} (papers nearest its centre, at most ${COMPARE_MAX})`}
          paperCount={Math.min(counts.get(compareCluster) ?? 0, COMPARE_MAX) || null}
          onClose={() => setCompareCluster(null)}
        />
      ) : null}

    </section>
  )
}
