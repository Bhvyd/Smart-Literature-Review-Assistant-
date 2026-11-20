import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import App from './App'

function fakeResponse(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => JSON.stringify(body),
  } as unknown as Response
}

function collection(overrides: Record<string, unknown> = {}) {
  return {
    id: 'c1',
    mode: 'topic',
    query: 'retrieval augmented generation',
    status: 'ready',
    error: null,
    embedding_model: 'allenai/specter2',
    n_papers: 100,
    n_chunks: 5030,
    created_at: '2026-11-13T10:00:00+00:00',
    updated_at: '2026-11-13T10:20:00+00:00',
    ...overrides,
  }
}

function stubApi(collections: unknown[], generationAvailable = false) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)
    if (url === '/api/health') {
      return fakeResponse({
        status: 'ok',
        version: '0.1.0',
        embedding_model: 'allenai/specter2',
        generation_available: generationAvailable,
      })
    }
    if (url === '/api/collections') return fakeResponse({ collections })
    throw new Error(`unexpected request: ${url}`)
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('app shell', () => {
  it('lists every collection and switches the selected one', async () => {
    stubApi([
      collection({ id: 'c1', query: 'retrieval augmented generation', n_papers: 100 }),
      collection({
        id: 'c2',
        query: 'graph neural networks',
        status: 'failed',
        error: 'arxiv unreachable',
        n_papers: 0,
      }),
    ])
    const user = userEvent.setup()
    render(<App />)

    const first = await screen.findByRole('button', { name: /retrieval augmented generation/ })
    expect(first).toHaveAttribute('aria-current', 'true')
    const second = screen.getByRole('button', { name: /graph neural networks/ })

    await user.click(second)

    const meta = second
    expect(meta).toHaveTextContent('failed')
    expect(meta).toHaveTextContent('arxiv unreachable')
    expect(second).toHaveAttribute('aria-current', 'true')
    expect(first).not.toHaveAttribute('aria-current')
    expect(screen.getByTestId('collection-heading')).toHaveTextContent('graph neural networks')

    expect(second).toBeVisible()
  })

  it('marks generation as unavailable in the navigation', async () => {
    stubApi([collection()], false)
    render(<App />)

    await screen.findByRole('button', { name: /retrieval augmented generation/ })
    const nav = screen.getByRole('navigation', { name: 'Views' })
    expect(within(nav).getByRole('button', { name: /Ask/ })).toHaveTextContent('unavailable')
  })

  it('shows an empty state and a build action when there are no collections', async () => {
    stubApi([])
    const user = userEvent.setup()
    render(<App />)

    expect(await screen.findByText('No collections yet')).toBeVisible()
    expect(screen.getByText('None yet')).toBeVisible()

    await user.click(screen.getByRole('button', { name: 'Build a collection' }))
    expect(screen.getByRole('dialog', { name: 'Build a collection' })).toBeVisible()
  })

  it('refreshes the rail while a collection is still building', async () => {
    let listCalls = 0
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input)
        if (url === '/api/health') {
          return fakeResponse({
            status: 'ok',
            version: '0.1.0',
            embedding_model: 'allenai/specter2',
            generation_available: false,
            generation_reason: 'not_implemented',
            generation_configured: false,
          })
        }
        if (url === '/api/collections') {
          listCalls += 1
          return fakeResponse({
            collections: [
              collection({
                status: listCalls === 1 ? 'building' : 'ready',
                n_papers: listCalls === 1 ? 0 : 100,
              }),
            ],
          })
        }
        if (url.includes('/papers')) {
          return fakeResponse({
            papers: [],
            mode: 'lexical',
            degraded: [],
            total: 0,
            offset: 0,
            limit: url.includes('limit=200') ? 200 : 20,
          })
        }
        throw new Error(`unexpected request: ${url}`)
      }),
    )

    render(<App />)

    expect(await screen.findByText('building')).toBeVisible()
    expect(await screen.findByText('ready', {}, { timeout: 4000 })).toBeVisible()
  })

  it('shows the build mode and a reduced-function badge that names its reasons', async () => {
    stubApi([
      collection({ degraded: ['openalex_rate_limited'] }),
      collection({ id: 'c2', mode: 'upload', query: 'My PDFs', degraded: [] }),
    ])
    const user = userEvent.setup()
    render(<App />)

    const first = await screen.findByRole('button', { name: /retrieval augmented generation/ })
    expect(first).toHaveTextContent('topic search')
    expect(screen.getByRole('button', { name: /My PDFs/ })).toHaveTextContent('PDF upload')

    const badges = screen.getAllByText('reduced-function')
    expect(badges).toHaveLength(1)
    await user.click(badges[0])
    expect(screen.getByText(/Some citation data is missing/)).toBeVisible()
    expect(screen.getByText('(openalex_rate_limited)')).toBeVisible()
  })

  it('never shows a paper count of zero while a build is running', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input)
        if (url === '/api/health') {
          return fakeResponse({ status: 'ok', version: '0.1.0', embedding_model: null, generation_available: false })
        }
        if (url === '/api/collections') {
          return fakeResponse({ collections: [collection({ status: 'building', n_papers: 0 })] })
        }
        if (url.includes('/papers')) {
          return fakeResponse(
            { error: { code: 'not_ready', message: 'collection is building, not ready', detail: { status: 'building' } } },
            409,
          )
        }
        throw new Error(`unexpected request: ${url}`)
      }),
    )
    render(<App />)

    const item = await screen.findByRole('button', { name: /retrieval augmented generation/ })
    expect(item).toHaveTextContent('building')
    expect(item).not.toHaveTextContent(/·\s*0\b/)
  })

  it('does not request data views for a collection that is still building', async () => {
    const fetchMock = stubApi([collection({ status: 'building', n_papers: 0 })], true)
    const user = userEvent.setup()
    render(<App />)

    expect(await screen.findByText('This collection is still building.')).toBeVisible()
    for (const name of [/Map/, /Citations/, /Ask/, /Papers/]) {
      await user.click(within(screen.getByRole('navigation', { name: 'Views' })).getByRole('button', { name }))
      expect(screen.getByText('This collection is still building.')).toBeVisible()
    }
    const urls = fetchMock.mock.calls.map((call) => String(call[0]))
    expect(urls.filter((url) => /\/(papers|map|graph|ask)/.test(url))).toEqual([])
  })

  it("shows a failed collection's reason instead of requesting its data", async () => {
    const fetchMock = stubApi([collection({ status: 'failed', error: 'arxiv unreachable' })])
    render(<App />)

    expect(await screen.findByText('This collection failed to build.')).toBeVisible()
    expect(screen.getAllByText(/arxiv unreachable/).length).toBeGreaterThan(0)
    expect(fetchMock.mock.calls.some((call) => String(call[0]).includes('/papers'))).toBe(false)
  })

  it('reports an unreachable API with a retry action', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch')
      }),
    )
    render(<App />)

    expect(await screen.findByRole('alert')).toHaveTextContent('Cannot reach the server')
    expect(screen.getByRole('button', { name: 'Try again' })).toBeVisible()
  })
})

describe('deleting a collection', () => {
  const emptyPage = { papers: [], mode: 'lexical', degraded: [], total: 0, offset: 0, limit: 20 }
  const health = { status: 'ok', version: '0.1.0', embedding_model: null, generation_available: false }

  function stubWithDelete(rows: unknown[], deleteResponse: () => Response) {
    let current = rows as { id: string }[]
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (init?.method === 'DELETE') {
        const response = deleteResponse()
        if (response.ok) current = current.filter((row) => !url.endsWith(`/${row.id}`))
        return response
      }
      if (url === '/api/health') return fakeResponse(health)
      if (url === '/api/collections') return fakeResponse({ collections: current })
      if (url.includes('/papers')) return fakeResponse(emptyPage)
      throw new Error(`unexpected request: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)
    return fetchMock
  }

  const deleteCalls = (mock: ReturnType<typeof stubWithDelete>) =>
    mock.mock.calls.filter((call) => (call[1] as RequestInit | undefined)?.method === 'DELETE')

  const two = () => [
    collection({ id: 'c1', query: 'retrieval augmented generation', n_papers: 100 }),
    collection({ id: 'c2', query: 'graph neural networks', n_papers: 42 }),
  ]

  it('asks first, naming the collection and its paper count, and Cancel deletes nothing', async () => {
    const fetchMock = stubWithDelete(two(), () => fakeResponse({}))
    const user = userEvent.setup()
    render(<App />)

    await user.click(
      await screen.findByRole('button', { name: 'Delete the selected collection' }),
    )
    const dialog = screen.getByRole('alertdialog', { name: /Delete .retrieval augmented generation.\?/ })
    expect(dialog).toHaveTextContent('100 papers')
    expect(dialog).toHaveTextContent('cannot be undone')

    expect(screen.getByRole('button', { name: 'Cancel' })).toHaveFocus()
    expect(screen.getByRole('button', { name: 'Delete collection' })).toHaveClass('button-secondary')

    await user.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(screen.queryByRole('alertdialog')).toBeNull()
    expect(deleteCalls(fetchMock)).toHaveLength(0)
  })

  it('deletes after confirmation, refreshes the rail and selects another collection', async () => {
    const fetchMock = stubWithDelete(two(), () =>
      fakeResponse({ collection_id: 'c1', deleted: { collections: 1 } }),
    )
    const user = userEvent.setup()
    render(<App />)

    await user.click(
      await screen.findByRole('button', { name: 'Delete the selected collection' }),
    )
    await user.click(screen.getByRole('button', { name: 'Delete collection' }))

    await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull())
    expect(deleteCalls(fetchMock).map((call) => String(call[0]))).toEqual(['/api/collections/c1'])
    expect(screen.queryByRole('button', { name: /^retrieval augmented generation/ })).toBeNull()
    expect(screen.getByRole('button', { name: /^graph neural networks/ })).toHaveAttribute(
      'aria-current',
      'true',
    )
  })

  it('shows the empty state after the last collection is deleted', async () => {
    stubWithDelete([collection()], () => fakeResponse({ collection_id: 'c1', deleted: {} }))
    const user = userEvent.setup()
    render(<App />)

    await user.click(await screen.findByRole('button', { name: 'Delete the selected collection' }))
    await user.click(screen.getByRole('button', { name: 'Delete collection' }))

    expect(await screen.findByText('No collections yet')).toBeVisible()
  })

  it('keeps the dialog open and says why when the server refuses', async () => {
    stubWithDelete(two(), () =>
      fakeResponse(
        { error: { code: 'build_in_progress', message: 'a collection that is still building cannot be deleted' } },
        409,
      ),
    )
    const user = userEvent.setup()
    render(<App />)

    await user.click(
      await screen.findByRole('button', { name: 'Delete the selected collection' }),
    )
    await user.click(screen.getByRole('button', { name: 'Delete collection' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('still building cannot be deleted')
    expect(screen.getByRole('alertdialog')).toBeVisible()
  })

  it('does not offer deletion while the collection is building (it offers Cancel build instead)', async () => {
    stubApi([collection({ status: 'building', n_papers: 0 })])
    render(<App />)

    expect(await screen.findByRole('button', { name: 'Cancel the build of the selected collection' })).toBeEnabled()
    expect(screen.queryByRole('button', { name: 'Delete the selected collection' })).toBeNull()
  })
})

describe('delete dialog behaviour', () => {
  const health = { status: 'ok', version: '0.1.0', embedding_model: null, generation_available: false }
  const aPaper = {
    paper_id: 'p1',
    title: 'Paper One',
    year: 2024,
    citation_count: 10,
    score: null,
    best_chunk_id: null,
    abstract: null,
    authors: ['Ada Lovelace'],
    venue: null,
    pdf_url: null,
    arxiv_id: null,
    parse_status: 'ok',
    parse_note: null,
    has_fulltext: 1,
  }
  const pageOf = (papers: unknown[]) => ({
    papers,
    mode: 'lexical',
    degraded: [],
    total: papers.length,
    offset: 0,
    limit: 20,
  })

  function stubServer(onDelete: () => Response, papers: unknown[] = []) {
    let rows = [
      collection({ id: 'c1', query: 'retrieval augmented generation', n_papers: 100 }),
      collection({ id: 'c2', query: 'graph neural networks', n_papers: 42 }),
    ]
    const listHolds: (() => void)[] = []
    let holdNextList = false
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (init?.method === 'DELETE') {
        const response = onDelete()
        if (response.ok) rows = rows.filter((row) => !url.endsWith(`/${row.id}`))
        return response
      }
      if (url === '/api/health') return fakeResponse(health)
      if (url === '/api/collections') {
        const snapshot = rows
        if (holdNextList) {
          holdNextList = false
          await new Promise<void>((resolve) => listHolds.push(resolve))
        }
        return fakeResponse({ collections: snapshot })
      }
      if (url.startsWith('/api/papers/')) return fakeResponse({ error: { code: 'not_found', message: 'gone' } }, 404)
      if (url.includes('/papers')) return fakeResponse(pageOf(papers))
      throw new Error(`unexpected request: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)
    return {
      fetchMock,
      holdNextList: () => {
        holdNextList = true
      },
      release: () => listHolds.splice(0).forEach((resolve) => resolve()),
    }
  }

  const deleteCount = (mock: ReturnType<typeof vi.fn>) =>
    mock.mock.calls.filter((call) => (call[1] as RequestInit | undefined)?.method === 'DELETE').length

  async function openDialog(user: ReturnType<typeof userEvent.setup>) {
    const opener = await screen.findByRole('button', { name: 'Delete the selected collection' })
    await user.click(opener)
    return opener
  }

  it('Escape cancels, even after focus has moved off the dialog, and returns focus to the opener', async () => {
    const { fetchMock } = stubServer(() => fakeResponse({}))
    const user = userEvent.setup()
    render(<App />)
    const opener = await openDialog(user)

    screen.getByRole('button', { name: /^Papers/ }).focus()
    await user.keyboard('{Escape}')

    expect(screen.queryByRole('alertdialog')).toBeNull()
    expect(opener).toHaveFocus()
    expect(deleteCount(fetchMock)).toBe(0)
  })

  it('a click on the backdrop cancels and returns focus to the opener', async () => {
    const { fetchMock } = stubServer(() => fakeResponse({}))
    const user = userEvent.setup()
    render(<App />)
    const opener = await openDialog(user)

    await user.click(screen.getByRole('alertdialog').parentElement as HTMLElement)

    expect(screen.queryByRole('alertdialog')).toBeNull()
    expect(opener).toHaveFocus()
    expect(deleteCount(fetchMock)).toBe(0)
  })

  it('keeps Tab and Shift+Tab inside the dialog', async () => {
    stubServer(() => fakeResponse({}))
    const user = userEvent.setup()
    render(<App />)
    await openDialog(user)

    const cancel = screen.getByRole('button', { name: 'Cancel' })
    const confirm = screen.getByRole('button', { name: 'Delete collection' })
    expect(cancel).toHaveFocus()
    await user.tab()
    expect(confirm).toHaveFocus()
    await user.tab()
    expect(cancel).toHaveFocus()
    await user.tab({ shift: true })
    expect(confirm).toHaveFocus()
  })

  it('draws the dialog outside the rail, on document.body', async () => {
    stubServer(() => fakeResponse({}))
    const user = userEvent.setup()
    render(<App />)
    await openDialog(user)

    expect(screen.getByRole('alertdialog').closest('.rail')).toBeNull()
    expect(screen.getByRole('alertdialog').closest('.dialog-backdrop')?.parentElement).toBe(document.body)
  })

  it('keeps focus in the dialog when the delete fails', async () => {
    stubServer(() => fakeResponse({ error: { code: 'build_in_progress', message: 'still building' } }, 409))
    const user = userEvent.setup()
    render(<App />)
    await openDialog(user)

    const confirm = screen.getByRole('button', { name: 'Delete collection' })
    await user.click(confirm)

    expect(await screen.findByRole('alert')).toHaveTextContent('still building')
    expect(screen.getByRole('alertdialog')).toContainElement(document.activeElement as HTMLElement)
    expect(confirm).toHaveFocus()
  })

  it('moves focus to the newly selected collection after a successful delete', async () => {
    stubServer(() => fakeResponse({ collection_id: 'c1', deleted: {} }))
    const user = userEvent.setup()
    render(<App />)
    await openDialog(user)
    await user.click(screen.getByRole('button', { name: 'Delete collection' }))

    const next = await screen.findByRole('button', { name: /^graph neural networks/ })
    await waitFor(() => expect(next).toHaveFocus())
  })

  it('treats a 404 on delete as already gone and refreshes the list', async () => {
    const { fetchMock } = stubServer(() => fakeResponse({ error: { code: 'not_found', message: 'gone' } }, 404))
    const user = userEvent.setup()
    render(<App />)
    await openDialog(user)
    await user.click(screen.getByRole('button', { name: 'Delete collection' }))

    await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull())
    expect(deleteCount(fetchMock)).toBe(1)

    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('closes an open paper drawer when its collection is deleted', async () => {
    stubServer(() => fakeResponse({ collection_id: 'c1', deleted: {} }), [aPaper])
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole('button', { name: /Paper One/ }))
    expect(await screen.findByRole('dialog')).toBeVisible()

    await user.click(screen.getByRole('button', { name: 'Delete the selected collection' }))
    await user.click(screen.getByRole('button', { name: 'Delete collection' }))

    await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull())
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('a refresh that was in flight before the delete cannot bring the deleted row back', async () => {
    const server = stubServer(() => fakeResponse({ collection_id: 'c1', deleted: {} }))
    const user = userEvent.setup()
    render(<App />)
    await screen.findByRole('button', { name: /^retrieval augmented generation/ })

    server.holdNextList()
    await user.click(screen.getByRole('button', { name: 'Build a collection' }))
    await user.click(screen.getByRole('button', { name: 'Cancel' }))

    await user.click(screen.getByRole('button', { name: 'Delete the selected collection' }))
    await user.click(screen.getByRole('button', { name: 'Delete collection' }))
    await waitFor(() => expect(screen.queryByRole('button', { name: /^retrieval augmented generation/ })).toBeNull())

    server.release()
    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(screen.queryByRole('button', { name: /^retrieval augmented generation/ })).toBeNull()
    expect(screen.getByRole('button', { name: /^graph neural networks/ })).toBeVisible()
  })
})

describe('build progress and cancellation in the rail', () => {
  const health = { status: 'ok', version: '0.1.0', embedding_model: null, generation_available: false }
  const running = { id: 'j1', status: 'running', progress: 0.45, message: 'ingested 45/100' }

  function stubBuilding(onCancel: () => Response, job: unknown = running) {
    let row = collection({ id: 'c1', status: 'building', n_papers: 0, job })
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (init?.method === 'POST' && url.endsWith('/cancel')) {
        const response = onCancel()
        if (response.ok) row = collection({ id: 'c1', status: 'failed', error: 'Cancelled.', job: null })
        return response
      }
      if (url === '/api/health') return fakeResponse(health)
      if (url === '/api/collections') return fakeResponse({ collections: [row] })
      throw new Error(`unexpected request: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)
    return fetchMock
  }
  const cancelCalls = (mock: ReturnType<typeof vi.fn>) =>
    mock.mock.calls.filter((call) => String(call[0]).endsWith('/cancel'))

  it('shows the live count while building, and "queued" while waiting for the slot', async () => {
    stubBuilding(() => fakeResponse({}))
    render(<App />)
    const item = await screen.findByRole('button', { name: /^retrieval augmented generation/ })
    expect(item).toHaveTextContent('building · 45/100')
  })

  it('says only "queued" for a build waiting for another build', async () => {
    stubBuilding(() => fakeResponse({}), {
      id: 'j1',
      status: 'queued',
      progress: 0,
      message: 'waiting for another build to finish',
    })
    render(<App />)
    const item = await screen.findByRole('button', { name: /^retrieval augmented generation/ })
    expect(item).toHaveTextContent('queued')
    expect(item).not.toHaveTextContent('building')
  })

  it('says only "building" when the job has no count yet or the backend sends no job', async () => {
    stubBuilding(() => fakeResponse({}), { id: 'j1', status: 'running', progress: 0, message: 'searching arXiv' })
    const { unmount } = render(<App />)
    expect(await screen.findByRole('button', { name: /^retrieval augmented generation/ })).toHaveTextContent(
      /building · topic search$/,
    )
    unmount()
    stubBuilding(() => fakeResponse({}), null)
    render(<App />)
    expect(await screen.findByRole('button', { name: /^retrieval augmented generation/ })).toHaveTextContent(
      /building · topic search$/,
    )
  })

  it('offers Cancel build, not Delete, while building', async () => {
    stubBuilding(() => fakeResponse({}))
    render(<App />)

    expect(await screen.findByRole('button', { name: 'Cancel the build of the selected collection' })).toBeEnabled()
    expect(screen.queryByRole('button', { name: 'Delete the selected collection' })).toBeNull()
  })

  it('asks first, with the stop copy, focusing the harmless choice; dismissing cancels nothing', async () => {
    const fetchMock = stubBuilding(() => fakeResponse({}))
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole('button', { name: 'Cancel the build of the selected collection' }))

    const dialog = screen.getByRole('alertdialog', { name: /Stop building .retrieval augmented generation.\?/ })
    expect(dialog).toHaveTextContent('Papers fetched so far are discarded.')
    expect(screen.getByRole('button', { name: 'Keep building' })).toHaveFocus()
    expect(screen.getByRole('button', { name: 'Cancel build' })).toHaveClass('button-secondary')

    await user.click(screen.getByRole('button', { name: 'Keep building' }))
    expect(screen.queryByRole('alertdialog')).toBeNull()
    expect(cancelCalls(fetchMock)).toHaveLength(0)
  })

  it('Escape dismisses it and returns focus to the opener; Tab stays inside', async () => {
    const fetchMock = stubBuilding(() => fakeResponse({}))
    const user = userEvent.setup()
    render(<App />)
    const opener = await screen.findByRole('button', { name: 'Cancel the build of the selected collection' })
    await user.click(opener)

    const keep = screen.getByRole('button', { name: 'Keep building' })
    const stop = screen.getByRole('button', { name: 'Cancel build' })
    await user.tab()
    expect(stop).toHaveFocus()
    await user.tab()
    expect(keep).toHaveFocus()
    await user.tab({ shift: true })
    expect(stop).toHaveFocus()

    await user.keyboard('{Escape}')
    expect(screen.queryByRole('alertdialog')).toBeNull()
    expect(opener).toHaveFocus()
    expect(cancelCalls(fetchMock)).toHaveLength(0)
  })

  it('confirming posts the cancel, closes, and the rail then offers a normal Delete', async () => {
    const fetchMock = stubBuilding(() => fakeResponse({ collection_id: 'c1', status: 'cancelling' }, 202))
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole('button', { name: 'Cancel the build of the selected collection' }))
    await user.click(screen.getByRole('button', { name: 'Cancel build' }))

    await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull())
    expect(cancelCalls(fetchMock).map((call) => String(call[0]))).toEqual(['/api/collections/c1/cancel'])
    expect(screen.getByRole('button', { name: /^retrieval augmented generation/ })).toHaveTextContent('cancelling')
    expect(screen.queryByRole('button', { name: 'Cancel the build of the selected collection' })).toBeNull()

    expect(await screen.findByRole('button', { name: 'Delete the selected collection' }, { timeout: 4000 })).toBeEnabled()
  })

  it('keeps the dialog open and shows the server message when the cancel is refused', async () => {
    stubBuilding(() =>
      fakeResponse(
        { error: { code: 'not_building', message: 'this collection is not building, so there is nothing to cancel' } },
        409,
      ),
    )
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole('button', { name: 'Cancel the build of the selected collection' }))
    const confirm = screen.getByRole('button', { name: 'Cancel build' })
    await user.click(confirm)

    expect(await screen.findByRole('alert')).toHaveTextContent('nothing to cancel')
    expect(screen.getByRole('alertdialog')).toBeVisible()
    expect(confirm).toHaveFocus()
  })
})
