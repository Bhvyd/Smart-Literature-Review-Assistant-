import type { GraphClusters, GraphNode, GraphPayload } from '../api/types'

export type { GraphClusters }

export function readClusters(payload: GraphPayload): GraphClusters | null {
  const raw: Partial<GraphClusters> | undefined = payload.clusters
  if (!raw || typeof raw !== 'object') return null
  return {
    available: raw.available === true,
    reason: typeof raw.reason === 'string' ? raw.reason : null,
    ids: Array.isArray(raw.ids) ? raw.ids : [],
    unplaced: typeof raw.unplaced === 'number' ? raw.unplaced : 0,
  }
}

export function nodeCluster(node: GraphNode): number | null {
  const value: number | null | undefined = node.cluster
  return typeof value === 'number' ? value : null
}

export const CLUSTER_REASON_COPY: Record<string, string> = {
  map_not_computed: 'Open the Map once to colour nodes by cluster.',
  clusters_unreliable: 'No cluster colours: the clustering was unreliable.',
  map_stale: 'Open the Map to refresh cluster colours.',
  map_unreadable: 'No cluster colours: the stored map could not be read.',
  map_empty: 'No cluster colours: the map is empty.',
}

export function describeClusterReason(reason: string | null): string {
  return (reason && CLUSTER_REASON_COPY[reason]) ?? 'No cluster colours available.'
}

const OPENALEX_ID = /^W\d+$/

export function untitledOpenAlexId(id: string, title: string | null | undefined): string | null {
  if (!id.startsWith('ext:')) return null
  const openalexId = id.slice(4)
  if (!OPENALEX_ID.test(openalexId)) return null
  return !title || title === openalexId ? openalexId : null
}

export function openAlexUrl(openalexId: string): string {
  return `https://openalex.org/${openalexId}`
}
