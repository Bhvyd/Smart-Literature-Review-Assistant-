import { describe, expect, it, vi } from 'vitest'

import { layoutGraph } from './graphLayout'

const WIDTH = 900
const HEIGHT = 600

function chain() {
  return {
    nodes: [
      { id: 'a', resolved: true },
      { id: 'b', resolved: true },
      { id: 'c', resolved: true },
    ],
    edges: [
      { source: 'a', target: 'b' },
      { source: 'b', target: 'c' },
    ],
  }
}

describe('graph layout', () => {
  it('positions every node deterministically and finitely', () => {
    const { nodes, edges } = chain()
    const first = layoutGraph(nodes, edges, { width: WIDTH, height: HEIGHT })
    const second = layoutGraph(nodes, edges, { width: WIDTH, height: HEIGHT })

    expect([...first.keys()].sort()).toEqual(['a', 'b', 'c'])
    for (const position of first.values()) {
      expect(Number.isFinite(position.x)).toBe(true)
      expect(Number.isFinite(position.y)).toBe(true)
    }
    expect([...second.entries()]).toEqual([...first.entries()])
  })

  it('never calls Math.random', () => {
    const random = vi.spyOn(Math, 'random')
    const { nodes, edges } = chain()

    layoutGraph(nodes, edges, { width: WIDTH, height: HEIGHT })

    expect(random).not.toHaveBeenCalled()
  })

  it('places external references near the internal node they cite', () => {
    const nodes = [
      { id: 'a', resolved: true },
      { id: 'b', resolved: true },
      ...Array.from({ length: 40 }, (_, index) => ({ id: `ext-${index}`, resolved: false })),
    ]
    const edges = [
      { source: 'a', target: 'b' },
      ...nodes
        .filter((node) => !node.resolved)
        .map((node) => ({ source: 'a', target: node.id })),
    ]

    const positions = layoutGraph(nodes, edges, { width: WIDTH, height: HEIGHT })
    const anchor = positions.get('a')!

    expect(positions.size).toBe(42)
    for (const node of nodes.filter((row) => !row.resolved)) {
      const placed = positions.get(node.id)!
      expect(Math.hypot(placed.x - anchor.x, placed.y - anchor.y)).toBeLessThan(140)
    }
  })

  it('lays out 2,400 nodes without dropping any or recursing', () => {
    const internal = Array.from({ length: 60 }, (_, index) => ({
      id: `i${index}`,
      resolved: true,
    }))
    const external = Array.from({ length: 2340 }, (_, index) => ({
      id: `e${index}`,
      resolved: false,
    }))
    const edges = external.map((node, index) => ({
      source: `i${index % 60}`,
      target: node.id,
    }))

    const positions = layoutGraph([...internal, ...external], edges, {
      width: WIDTH,
      height: HEIGHT,
    })

    expect(positions.size).toBe(2400)
    for (const position of positions.values()) {
      expect(Number.isFinite(position.x) && Number.isFinite(position.y)).toBe(true)
    }
  })

  it('keeps every external node inside the viewport, however many there are', () => {
    const internal = Array.from({ length: 60 }, (_, index) => ({
      id: `i${index}`,
      resolved: true,
    }))
    const external = Array.from({ length: 2340 }, (_, index) => ({
      id: `e${index}`,
      resolved: false,
    }))
    const edges = external.map((node, index) => ({
      source: `i${index % 60}`,
      target: node.id,
    }))

    const positions = layoutGraph([...internal, ...external], edges, {
      width: WIDTH,
      height: HEIGHT,
      padding: 36,
    })

    for (const position of positions.values()) {
      expect(position.x).toBeGreaterThanOrEqual(35)
      expect(position.x).toBeLessThanOrEqual(WIDTH - 35)
      expect(position.y).toBeGreaterThanOrEqual(35)
      expect(position.y).toBeLessThanOrEqual(HEIGHT - 35)
    }
  })
})
