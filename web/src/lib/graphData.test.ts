import { describe, expect, it } from 'vitest'

import type { GraphPayload } from '../api/types'
import {
  describeClusterReason,
  nodeCluster,
  openAlexUrl,
  readClusters,
  untitledOpenAlexId,
} from './graphData'

const base: GraphPayload = { nodes: [], edges: [], rankings: [], unidentified: 0 }

describe('graph cluster fields', () => {
  it('reads the cluster block and falls back to null for an older server', () => {
    expect(readClusters(base)).toBeNull()
    const withClusters = { ...base, clusters: { available: true, reason: null, ids: [0, 2], unplaced: 3 } }
    expect(readClusters(withClusters as GraphPayload)).toEqual({
      available: true,
      reason: null,
      ids: [0, 2],
      unplaced: 3,
    })
  })

  it('reads a node cluster only when it is a number', () => {
    const node = { id: 'a', title: 'A', year: null, citation_count: null, resolved: true }
    expect(nodeCluster(node)).toBeNull()
    expect(nodeCluster({ ...node, cluster: 3 } as typeof node)).toBe(3)
    expect(nodeCluster({ ...node, cluster: -1 } as typeof node)).toBe(-1)
    expect(nodeCluster({ ...node, cluster: null } as typeof node)).toBeNull()
  })

  it('explains each unavailable reason and does not guess at unknown ones', () => {
    expect(describeClusterReason('map_not_computed')).toMatch(/Open the Map once/)
    expect(describeClusterReason('clusters_unreliable')).toMatch(/unreliable/)
    expect(describeClusterReason('map_empty')).toMatch(/map is empty/)
    expect(describeClusterReason('something_new')).toBe('No cluster colours available.')
  })
})

describe('untitled references', () => {
  it('recognises an OpenAlex id used as a title, and nothing else', () => {
    expect(untitledOpenAlexId('ext:W6948136082', 'W6948136082')).toBe('W6948136082')
    expect(untitledOpenAlexId('ext:W6948136082', null)).toBe('W6948136082')
    expect(untitledOpenAlexId('ext:W6948136082', 'Attention Is All You Need')).toBeNull()
    expect(untitledOpenAlexId('ext:title:abc123', 'Some Real Title')).toBeNull()
    expect(untitledOpenAlexId('paper-1', 'paper-1')).toBeNull()
  })

  it('links to the OpenAlex record', () => {
    expect(openAlexUrl('W6948136082')).toBe('https://openalex.org/W6948136082')
  })
})
