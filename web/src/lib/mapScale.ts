export interface MapPointLike {
  x: number
  y: number
}

export interface FittedProjection {

  project: (x: number, y: number) => { x: number; y: number }
  domain: { minX: number; maxX: number; minY: number; maxY: number }
}

export function fitToViewport(
  points: MapPointLike[],
  width: number,
  height: number,
  padding = 24,
): FittedProjection {
  if (points.length === 0) {
    return {
      project: () => ({ x: width / 2, y: height / 2 }),
      domain: { minX: -1, maxX: 1, minY: -1, maxY: 1 },
    }
  }

  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity
  for (const point of points) {
    if (point.x < minX) minX = point.x
    if (point.x > maxX) maxX = point.x
    if (point.y < minY) minY = point.y
    if (point.y > maxY) maxY = point.y
  }

  const centreX = (minX + maxX) / 2
  const centreY = (minY + maxY) / 2
  if (maxX - minX < 1e-9) {
    minX = centreX - 1
    maxX = centreX + 1
  }
  if (maxY - minY < 1e-9) {
    minY = centreY - 1
    maxY = centreY + 1
  }

  const innerWidth = Math.max(1, width - padding * 2)
  const innerHeight = Math.max(1, height - padding * 2)
  const spanX = maxX - minX
  const spanY = maxY - minY
  const scale = Math.min(innerWidth / spanX, innerHeight / spanY)

  const offsetX = (width - spanX * scale) / 2
  const offsetY = (height - spanY * scale) / 2

  return {
    project: (x: number, y: number) => ({
      x: offsetX + (x - minX) * scale,
      y: height - offsetY - (y - minY) * scale,
    }),
    domain: { minX, maxX, minY, maxY },
  }
}

export function clusterColour(cluster: number, sortedClusters: number[]): string {
  if (cluster < 0) return 'var(--noise)'
  const rank = sortedClusters.indexOf(cluster)
  return `var(--cluster-${((rank < 0 ? 0 : rank) % 7) + 1})`
}
