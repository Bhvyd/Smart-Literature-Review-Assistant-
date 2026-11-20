import { describe, expect, it } from 'vitest'

import { clusterColour, fitToViewport } from './mapScale'

const WIDTH = 900
const HEIGHT = 560
const PADDING = 30

function points(count: number) {
  return Array.from({ length: count }, (_, index) => ({
    x: Math.sin(index) * 40,
    y: Math.cos(index) * 25,
  }))
}

describe('map scale', () => {
  it('keeps every point inside the viewport', () => {
    const fitted = fitToViewport(points(20), WIDTH, HEIGHT, PADDING)

    for (const point of points(20)) {
      const placed = fitted.project(point.x, point.y)
      expect(placed.x).toBeGreaterThanOrEqual(PADDING - 1e-6)
      expect(placed.x).toBeLessThanOrEqual(WIDTH - PADDING + 1e-6)
      expect(placed.y).toBeGreaterThanOrEqual(PADDING - 1e-6)
      expect(placed.y).toBeLessThanOrEqual(HEIGHT - PADDING + 1e-6)
    }
  })

  it('puts a single point in the middle with a finite scale', () => {
    const fitted = fitToViewport([{ x: 7, y: -3 }], WIDTH, HEIGHT, PADDING)
    const placed = fitted.project(7, -3)

    expect(Number.isFinite(placed.x)).toBe(true)
    expect(Number.isFinite(placed.y)).toBe(true)
    expect(placed.x).toBeCloseTo(WIDTH / 2, 6)
    expect(placed.y).toBeCloseTo(HEIGHT / 2, 6)
  })

  it('expands a zero-span axis instead of dividing by zero', () => {
    const rows = [
      { x: 0, y: 5 },
      { x: 10, y: 5 },
    ]
    const fitted = fitToViewport(rows, WIDTH, HEIGHT, PADDING)
    const first = fitted.project(0, 5)
    const second = fitted.project(10, 5)

    expect(Number.isFinite(first.y)).toBe(true)
    expect(first.y).toBeCloseTo(HEIGHT / 2, 6)
    expect(second.y).toBeCloseTo(HEIGHT / 2, 6)
    expect(second.x).toBeGreaterThan(first.x)
  })

  it('is deterministic and inverts Y so larger values are higher', () => {
    const rows = [
      { x: 0, y: 0 },
      { x: 1, y: 10 },
    ]
    const a = fitToViewport(rows, WIDTH, HEIGHT, PADDING)
    const b = fitToViewport(rows, WIDTH, HEIGHT, PADDING)

    expect(a.project(1, 10)).toEqual(b.project(1, 10))
    expect(a.project(1, 10).y).toBeLessThan(a.project(0, 0).y)
  })

  it('never hands an empty payload a broken projection', () => {
    const fitted = fitToViewport([], WIDTH, HEIGHT, PADDING)
    const placed = fitted.project(1, 2)

    expect(placed).toEqual({ x: WIDTH / 2, y: HEIGHT / 2 })
  })

  it('maps clusters to a stable palette slot by sorted rank', () => {
    const clusters = [2, 5, 9]

    expect(clusterColour(2, clusters)).toBe('var(--cluster-1)')
    expect(clusterColour(5, clusters)).toBe('var(--cluster-2)')
    expect(clusterColour(9, clusters)).toBe('var(--cluster-3)')
    expect(clusterColour(-1, clusters)).toBe('var(--noise)')
  })
})
