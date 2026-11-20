import { afterEach, describe, expect, it, vi } from 'vitest'

import { ApiError, getMap, listPapers, messageFor, reasonCode, safeHref } from './client'

function fakeResponse(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => JSON.stringify(body),
  } as unknown as Response
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('api client', () => {
  it('turns an error envelope into an ApiError with its code and detail', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        fakeResponse(
          {
            error: {
              code: 'map_needs_more_papers',
              message: 'a similarity map needs at least 15 papers with vectors',
              detail: { reason: 'map_needs_more_papers', needed: 15, have: 4 },
            },
          },
          409,
        ),
      ),
    )

    const failure = await getMap('c1').catch((error: unknown) => error)

    expect(failure).toBeInstanceOf(ApiError)
    const error = failure as ApiError
    expect(error.code).toBe('map_needs_more_papers')
    expect(error.status).toBe(409)
    expect((error.detail as { needed: number }).needed).toBe(15)
    expect(messageFor(error)).toContain('at least 15 papers')
  })

  it('resolves a JSON payload and encodes the query string', async () => {
    const fetchMock = vi.fn(async () =>
      fakeResponse({ papers: [], mode: 'lexical', degraded: [], total: 0, offset: 0, limit: 20 }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const page = await listPapers('c1', { q: 'graph neural', limit: 20, offset: 40 })

    expect(page.total).toBe(0)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const url = String((fetchMock.mock.calls[0] as unknown[])[0])
    expect(url).toBe('/api/collections/c1/papers?q=graph+neural&limit=20&offset=40')
  })

  it('only turns real web addresses into links', () => {
    expect(safeHref('https://arxiv.org/abs/1')).toBe('https://arxiv.org/abs/1')
    expect(safeHref('/api/papers/p1/pdf')).toBe('/api/papers/p1/pdf')
    expect(safeHref('upload://abc')).toBeNull()
    expect(safeHref('javascript:alert(1)')).toBeNull()
    expect(safeHref(null)).toBeNull()
  })

  it('reads the generation reason out of an error envelope', () => {
    const error = new ApiError('generation_unavailable', 'no credits', 503, {
      reason: 'insufficient_quota',
    })
    expect(reasonCode(error)).toBe('insufficient_quota')
    expect(reasonCode(new ApiError('invalid_request', 'bad', 422))).toBe('invalid_request')
    expect(reasonCode(new Error('x'))).toBeNull()
  })

  it('reports an unreachable API instead of throwing a bare network error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch')
      }),
    )

    const failure = await getMap('c1').catch((error: unknown) => error)

    expect(failure).toBeInstanceOf(ApiError)
    expect((failure as ApiError).code).toBe('network_unavailable')
    expect((failure as ApiError).status).toBe(0)
  })
})
