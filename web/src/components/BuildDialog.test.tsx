import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { BuildDialog } from './BuildDialog'

function fakeResponse(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => JSON.stringify(body),
  } as unknown as Response
}

function job(overrides: Record<string, unknown> = {}) {
  return {
    id: 'j9',
    collection_id: 'c9',
    kind: 'build',
    status: 'running',
    progress: 0.12,
    message: 'ingested 12/100',
    error: null,
    created_at: '2026-11-13T10:00:00+00:00',
    finished_at: null,
    ...overrides,
  }
}

let polls: unknown[]

function stubBuildApi() {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)
    if (url === '/api/collections') {
      return fakeResponse({ collection_id: 'c9', job_id: 'j9', status: 'building' }, 202)
    }
    if (url === '/api/jobs/j9') {
      const next = polls.length > 1 ? polls.shift() : polls[0]
      return fakeResponse(next)
    }
    throw new Error(`unexpected request: ${url}`)
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

beforeEach(() => {
  polls = []
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('build dialog', () => {
  it('refuses a blank topic without calling the API', async () => {
    const fetchMock = stubBuildApi()
    const user = userEvent.setup()
    render(<BuildDialog onClose={vi.fn()} onCreated={vi.fn()} pollMs={5} />)

    await user.click(screen.getByRole('button', { name: 'Start build' }))

    expect(screen.getByText('Enter a topic to build.')).toBeVisible()
    expect(fetchMock).not.toHaveBeenCalled()

    const input = screen.getByLabelText('Topic')
    expect(input).toHaveAttribute('aria-invalid', 'true')
    expect(document.getElementById(input.getAttribute('aria-describedby') ?? '')).toHaveTextContent(
      'Enter a topic to build.',
    )
    expect(input).toHaveFocus()
  })

  it('reports real job progress and selects the collection when it finishes', async () => {
    polls = [job(), job({ status: 'done', progress: 1, message: 'ready' })]
    stubBuildApi()
    const onCreated = vi.fn()
    const onClose = vi.fn()
    const user = userEvent.setup()

    render(<BuildDialog onClose={onClose} onCreated={onCreated} pollMs={50} />)
    await user.type(screen.getByLabelText('Topic'), 'graph neural networks')
    await user.click(screen.getByRole('button', { name: 'Start build' }))

    expect(await screen.findByRole('status')).toHaveTextContent('ingested 12/100')
    expect(onCreated).not.toHaveBeenCalled()

    await waitFor(() => expect(onCreated).toHaveBeenCalledWith('c9'))
    expect(onClose).toHaveBeenCalled()
  })

  it('renders a failed job and does not select anything', async () => {
    polls = [job(), job({ status: 'failed', error: 'arxiv unreachable' })]
    stubBuildApi()
    const onCreated = vi.fn()
    const user = userEvent.setup()

    render(<BuildDialog onClose={vi.fn()} onCreated={onCreated} pollMs={5} />)
    await user.type(screen.getByLabelText('Topic'), 'graph neural networks')
    await user.click(screen.getByRole('button', { name: 'Start build' }))

    expect(await screen.findByText(/arxiv unreachable/)).toBeVisible()
    expect(screen.queryByText('Build did not start.')).toBeNull()
    expect(onCreated).not.toHaveBeenCalled()
  })

  it('keeps watching after a status blip and still selects the finished collection', async () => {

    let release: () => void = () => {}
    const gate = new Promise<void>((resolve) => {
      release = resolve
    })
    let jobCalls = 0
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input)
        if (url === '/api/collections') {
          return fakeResponse({ collection_id: 'c9', job_id: 'j9', status: 'building' }, 202)
        }
        if (url === '/api/jobs/j9') {
          jobCalls += 1
          const call = jobCalls
          if (call === 1) throw new TypeError('Failed to fetch')
          await gate
          if (call === 2) return fakeResponse(job())
          return fakeResponse(job({ status: 'done', progress: 1, message: 'ready' }))
        }
        throw new Error(`unexpected request: ${url}`)
      }),
    )
    const onCreated = vi.fn()
    const user = userEvent.setup()

    render(<BuildDialog onClose={vi.fn()} onCreated={onCreated} pollMs={10} />)
    await user.type(screen.getByLabelText('Topic'), 'graph neural networks')
    await user.click(screen.getByRole('button', { name: 'Start build' }))

    expect(await screen.findByText(/Lost track of the build/)).toBeVisible()

    expect(screen.getByText('Still running on the server.')).toBeVisible()
    expect(onCreated).not.toHaveBeenCalled()

    release()

    await waitFor(() => expect(onCreated).toHaveBeenCalledWith('c9'))
    expect(jobCalls).toBeGreaterThanOrEqual(3)
    expect(screen.queryByText(/Lost track of the build/)).toBeNull()
  })

  it('follows an already running build instead of starting a second one', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        fakeResponse(
          {
            error: {
              code: 'build_in_progress',
              message: 'a build for that topic is already running',
              detail: { collection_id: 'c7', job_id: 'j7' },
            },
          },
          409,
        ),
      ),
    )
    const onCreated = vi.fn()
    const onClose = vi.fn()
    const user = userEvent.setup()

    render(<BuildDialog onClose={onClose} onCreated={onCreated} pollMs={5} />)
    await user.type(screen.getByLabelText('Topic'), 'graph neural networks')
    await user.click(screen.getByRole('button', { name: 'Start build' }))

    await waitFor(() => expect(onCreated).toHaveBeenCalledWith('c7'))
    expect(onClose).toHaveBeenCalled()
  })
})

function pdf(name: string, sizeBytes = 2048, type = 'application/pdf') {
  const file = new File(['%PDF-1.7'], name, { type })

  Object.defineProperty(file, 'size', { value: sizeBytes })
  return file
}

describe('build dialog — upload mode', () => {
  it('names the file that breaks a limit and does not call the API', async () => {
    const fetchMock = stubBuildApi()
    const user = userEvent.setup({ applyAccept: false })
    render(<BuildDialog onClose={vi.fn()} onCreated={vi.fn()} pollMs={5} />)

    await user.click(screen.getByRole('radio', { name: 'Upload PDFs' }))
    const input = screen.getByLabelText(/PDF files/)

    await user.upload(input, [pdf('notes.docx', 100, 'application/msword')])
    expect(await screen.findByText(/notes\.docx is not a PDF/)).toBeVisible()

    await user.upload(input, [pdf('huge.pdf', 31 * 1024 * 1024)])
    expect(await screen.findByText(/huge\.pdf is over 30 MB/)).toBeVisible()

    await user.upload(
      input,
      Array.from({ length: 26 }, (_, index) => pdf(`paper-${index}.pdf`)),
    )
    expect(await screen.findByText(/Up to 25 PDFs \(you chose 26\)/)).toBeVisible()

    await user.click(screen.getByRole('button', { name: 'Upload and build' }))
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('refuses files whose combined size is over the per-upload limit, as the server does', async () => {
    const fetchMock = stubBuildApi()
    const user = userEvent.setup({ applyAccept: false })
    render(<BuildDialog onClose={vi.fn()} onCreated={vi.fn()} pollMs={5} />)

    await user.click(screen.getByRole('radio', { name: 'Upload PDFs' }))

    await user.upload(
      screen.getByLabelText(/PDF files/),
      Array.from({ length: 7 }, (_, index) => pdf(`big-${index}.pdf`, 29 * 1024 * 1024)),
    )
    expect(await screen.findByText(/files total 203\.0 MB, over the 200 MB limit/)).toBeVisible()

    await user.click(screen.getByRole('button', { name: 'Upload and build' }))
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('refuses to submit with no files chosen', async () => {
    const fetchMock = stubBuildApi()
    const user = userEvent.setup()
    render(<BuildDialog onClose={vi.fn()} onCreated={vi.fn()} pollMs={5} />)

    await user.click(screen.getByRole('radio', { name: 'Upload PDFs' }))
    await user.click(screen.getByRole('button', { name: 'Upload and build' }))

    expect(screen.getByText('Choose at least one PDF.')).toBeVisible()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('posts the files as multipart form data and follows the same job', async () => {
    polls = [job({ message: 'processed 1/2' }), job({ status: 'done', progress: 1 })]
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url === '/api/collections/upload') {
        expect(init?.method).toBe('POST')
        return fakeResponse({ collection_id: 'c9', job_id: 'j9', status: 'building' }, 202)
      }
      if (url === '/api/jobs/j9') {
        const next = polls.length > 1 ? polls.shift() : polls[0]
        return fakeResponse(next)
      }
      throw new Error(`unexpected request: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)
    const onCreated = vi.fn()
    const user = userEvent.setup()

    render(<BuildDialog onClose={vi.fn()} onCreated={onCreated} pollMs={20} />)
    await user.click(screen.getByRole('radio', { name: 'Upload PDFs' }))
    await user.upload(screen.getByLabelText(/PDF files/), [pdf('a.pdf'), pdf('b.pdf')])
    expect(screen.getByText(/2 files, /)).toBeVisible()
    await user.type(screen.getByLabelText(/Collection name/), 'My reading list')
    await user.click(screen.getByRole('button', { name: 'Upload and build' }))

    await waitFor(() => expect(onCreated).toHaveBeenCalledWith('c9'))

    const uploadCall = fetchMock.mock.calls.find((call) => String(call[0]) === '/api/collections/upload')
    const init = uploadCall?.[1] as RequestInit
    const body = init.body as FormData
    expect(body).toBeInstanceOf(FormData)
    expect(body.getAll('files')).toHaveLength(2)
    expect(body.get('name')).toBe('My reading list')

    expect((init.headers as Record<string, string>)['Content-Type']).toBeUndefined()
  })

  it('shows the server refusal for an invalid upload', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        fakeResponse(
          {
            error: {
              code: 'invalid_upload',
              message: 'a.pdf is not a PDF (missing the %PDF- header)',
              detail: null,
            },
          },
          422,
        ),
      ),
    )
    const user = userEvent.setup()

    render(<BuildDialog onClose={vi.fn()} onCreated={vi.fn()} pollMs={5} />)
    await user.click(screen.getByRole('radio', { name: 'Upload PDFs' }))
    await user.upload(screen.getByLabelText(/PDF files/), [pdf('a.pdf')])
    await user.click(screen.getByRole('button', { name: 'Upload and build' }))

    expect(await screen.findByText(/missing the %PDF- header/)).toBeVisible()
    expect(screen.getByText('Build did not start.')).toBeVisible()
  })
})
