export interface GraphNodeLike {
  id: string
  resolved: boolean
}

export interface GraphEdgeLike {
  source: string
  target: string
}

export interface LayoutOptions {
  width: number
  height: number
  padding?: number
}

export interface Position {
  x: number
  y: number
}

const MAX_INTERNAL = 400
const ITERATIONS = 200

function seedFrom(ids: string[]): number {
  let hash = 0x811c9dc5
  for (const id of ids) {
    for (let index = 0; index < id.length; index += 1) {
      hash ^= id.charCodeAt(index)
      hash = Math.imul(hash, 0x01000193)
    }
  }
  return hash >>> 0
}

function makeRandom(seed: number): () => number {
  let state = seed || 0x9e3779b9
  return () => {
    state ^= state << 13
    state ^= state >>> 17
    state ^= state << 5
    state >>>= 0
    return state / 0xffffffff
  }
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value))
}

export function layoutGraph(
  nodes: GraphNodeLike[],
  edges: GraphEdgeLike[],
  { width, height, padding = 36 }: LayoutOptions,
): Map<string, Position> {
  const positions = new Map<string, Position>()
  const internal = nodes.filter((node) => node.resolved).map((node) => node.id).sort()
  const internalSet = new Set(internal)

  const innerWidth = Math.max(1, width - padding * 2)
  const innerHeight = Math.max(1, height - padding * 2)
  const area = innerWidth * innerHeight
  const count = Math.min(internal.length, MAX_INTERNAL)

  const xs = new Float64Array(count)
  const ys = new Float64Array(count)
  const indexById = new Map<string, number>()

  if (count > 0) {
    const random = makeRandom(seedFrom(internal.slice(0, count)))
    const radius = Math.min(innerWidth, innerHeight) * 0.38
    for (let index = 0; index < count; index += 1) {
      const angle = (index / count) * Math.PI * 2
      const jitter = 0.85 + random() * 0.3
      xs[index] = innerWidth / 2 + Math.cos(angle) * radius * jitter
      ys[index] = innerHeight / 2 + Math.sin(angle) * radius * jitter
      indexById.set(internal[index], index)
    }

    const k = Math.sqrt(area / count)
    let temperature = Math.min(innerWidth, innerHeight) / 8

    const dx = new Float64Array(count)
    const dy = new Float64Array(count)

    for (let step = 0; step < ITERATIONS; step += 1) {
      dx.fill(0)
      dy.fill(0)

      for (let a = 0; a < count; a += 1) {
        for (let b = a + 1; b < count; b += 1) {
          let deltaX = xs[a] - xs[b]
          let deltaY = ys[a] - ys[b]
          let distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY)
          if (distance < 1e-6) {

            deltaX = 1e-3 * (a + 1)
            deltaY = 1e-3 * (b + 1)
            distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY)
          }
          const force = (k * k) / distance
          const unitX = deltaX / distance
          const unitY = deltaY / distance
          dx[a] += unitX * force
          dy[a] += unitY * force
          dx[b] -= unitX * force
          dy[b] -= unitY * force
        }
      }

      for (const edge of edges) {
        const a = indexById.get(edge.source)
        const b = indexById.get(edge.target)
        if (a === undefined || b === undefined || a === b) continue
        const deltaX = xs[a] - xs[b]
        const deltaY = ys[a] - ys[b]
        const distance = Math.max(1e-6, Math.sqrt(deltaX * deltaX + deltaY * deltaY))
        const force = (distance * distance) / k
        const unitX = deltaX / distance
        const unitY = deltaY / distance
        dx[a] -= unitX * force
        dy[a] -= unitY * force
        dx[b] += unitX * force
        dy[b] += unitY * force
      }

      for (let index = 0; index < count; index += 1) {
        const magnitude = Math.max(1e-9, Math.sqrt(dx[index] * dx[index] + dy[index] * dy[index]))
        const limited = Math.min(magnitude, temperature)
        xs[index] = clamp(xs[index] + (dx[index] / magnitude) * limited, 0, innerWidth)
        ys[index] = clamp(ys[index] + (dy[index] / magnitude) * limited, 0, innerHeight)
      }
      temperature = Math.max(temperature * 0.97, 0.5)
    }

    for (let index = 0; index < count; index += 1) {
      positions.set(internal[index], { x: xs[index] + padding, y: ys[index] + padding })
    }
  }

  for (let index = count; index < internal.length; index += 1) {
    positions.set(internal[index], {
      x: padding + ((index - count) % 10) * 12,
      y: height - padding,
    })
  }

  const neighbours = new Map<string, string[]>()
  for (const edge of edges) {

    if (!internalSet.has(edge.source)) continue
    const list = neighbours.get(edge.target) ?? []
    list.push(edge.source)
    neighbours.set(edge.target, list)
  }

  const external = nodes
    .filter((node) => !node.resolved)
    .map((node) => node.id)
    .sort()
  const golden = Math.PI * (3 - Math.sqrt(5))

  const perAnchor = new Map<string, number>()
  external.forEach((id) => {
    const anchorId = (neighbours.get(id) ?? []).slice().sort()[0] ?? ''
    const seen = perAnchor.get(anchorId) ?? 0
    perAnchor.set(anchorId, seen + 1)
    const anchor = anchorId ? positions.get(anchorId) : undefined
    const ring = 20 + 7 * Math.floor(seen / 12)
    const angle = seen * golden
    const point = anchor
      ? { x: anchor.x + Math.cos(angle) * ring, y: anchor.y + Math.sin(angle) * ring }
      : {

          x: padding + innerWidth / 2 + Math.cos(angle) * Math.min(innerWidth, innerHeight) * 0.46,
          y: padding + innerHeight / 2 + Math.sin(angle) * Math.min(innerWidth, innerHeight) * 0.46,
        }
    positions.set(id, {
      x: clamp(point.x, padding, width - padding),
      y: clamp(point.y, padding, height - padding),
    })
  })

  return positions
}

export function trimToRim(
  from: Position,
  to: Position,
  radius: number,
): { x: number; y: number } {
  const deltaX = to.x - from.x
  const deltaY = to.y - from.y
  const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY)
  if (distance <= radius + 1e-6) return { x: to.x, y: to.y }
  return {
    x: to.x - (deltaX / distance) * radius,
    y: to.y - (deltaY / distance) * radius,
  }
}
