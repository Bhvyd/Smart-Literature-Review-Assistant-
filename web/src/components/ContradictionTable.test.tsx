import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { Contradictions } from '../api/types'
import { ContradictionTable } from './ContradictionTable'

function fakeResponse(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => JSON.stringify(body),
  } as unknown as Response
}

const QUOTE_A = 'Dense retrieval outperforms BM25 by 9 points on Natural Questions.'
const QUOTE_B = 'BM25 remains stronger than dense retrieval on Natural Questions.'

function ref(id: string, title: string, year: number | null, url: string | null = null) {
  return { paper_id: id, title, year, url }
}

function result_fixture(overrides: Partial<Contradictions> = {}): Contradictions {
  return {
    comparisons: [
      {
        paper_a: ref('p1', 'Dense Passage Retrieval', 2020, 'https://arxiv.org/abs/2004.04906'),
        paper_b: ref('p2', 'Sparse Is Enough', 2021, null),
        verdict: 'conflict',
        quote_a: QUOTE_A,
        quote_b: QUOTE_B,
        source_a: 'full text',
        source_b: 'abstract',
        reason: 'Opposite findings on the same benchmark.',
        model: 'gemini-3.1-flash-lite',
        cached: false,
      },
      {
        paper_a: ref('p1', 'Dense Passage Retrieval', 2020),
        paper_b: ref('p3', 'Graph Neural Networks', null),
        verdict: 'no_comparable_claims',
        quote_a: null,
        quote_b: null,
        reason: null,
        note: 'the model found no pair of sentences addressing the same question',
        model: 'gemini-3.1-flash-lite',
        cached: true,
      },
      {
        paper_a: ref('p2', 'Sparse Is Enough', 2021),
        paper_b: ref('p3', 'Graph Neural Networks', null),
        verdict: 'no_comparable_claims',
        quote_a: null,
        quote_b: null,
        reason: null,
        note: 'a paper has no usable abstract sentences',
      },
    ],
    n_pairs_requested: 5,
    n_pairs_possible: 10,
    unavailable: null,
    incomplete_pair: null,
    method: 'pairs chosen by map similarity; each verdict rests on one sentence quoted verbatim',
    note: 'Abstracts only. A verdict is a lead to check, not a finding.',
    ai_generated: true,
    ...overrides,
  }
}

function stub(body: unknown, status = 200) {
  const mock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => fakeResponse(body, status))
  vi.stubGlobal('fetch', mock)
  return mock
}

function renderPanel(paperCount: number | null = 3, request = { paper_ids: ['p1', 'p2', 'p3'] }) {
  return render(
    <ContradictionTable
      collectionId="c1"
      request={request}
      subject="3 selected papers"
      paperCount={paperCount}
      onClose={vi.fn()}
    />,
  )
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('compare papers panel', () => {
  it('never runs on mount and states the cost before the click', async () => {
    const mock = stub(result_fixture())
    const user = userEvent.setup()
    renderPanel(8)

    expect(screen.getByRole('heading', { name: /Compare papers · 3 selected papers/ })).toBeVisible()
    expect(mock).not.toHaveBeenCalled()
    expect(screen.getByText(/model calls/)).toHaveTextContent('Up to 10 model calls.')

    const pairs = screen.getByLabelText('Pairs to compare')
    await user.clear(pairs)
    await user.type(pairs, '3{Enter}')

    expect(mock).not.toHaveBeenCalled()
    expect(screen.getByText(/model calls/)).toHaveTextContent('Up to 6 model calls.')

    await user.click(screen.getByRole('button', { name: 'Run comparison' }))
    expect(mock).toHaveBeenCalledTimes(1)
    const [url, init] = mock.mock.calls[0]
    expect(String(url)).toBe('/api/collections/c1/contradictions')
    expect(JSON.parse(String(init?.body))).toEqual({ paper_ids: ['p1', 'p2', 'p3'], max_pairs: 3 })
  })

  it('caps the pair count at what the papers can make and refuses an invalid number', async () => {
    const mock = stub(result_fixture())
    const user = userEvent.setup()
    renderPanel(2, { paper_ids: ['p1', 'p2'] })

    expect(screen.getByText(/model calls/)).toHaveTextContent('Up to 2 model calls.')
    expect(screen.getByText(/1 pair possible/)).toBeVisible()
    const pairs = screen.getByLabelText('Pairs to compare')
    await user.clear(pairs)
    await user.type(pairs, '4')
    expect(screen.getByText('Enter a whole number from 1 to 1.')).toBeVisible()
    expect(screen.getByRole('button', { name: 'Run comparison' })).toBeDisabled()
    expect(mock).not.toHaveBeenCalled()
  })

  it('renders verbatim quotes, verdict words and collapsed no-comparable rows with counts', async () => {
    stub(result_fixture())
    const user = userEvent.setup()
    renderPanel()

    await user.click(screen.getByRole('button', { name: 'Run comparison' }))

    const table = await screen.findByRole('table')
    const quotes = table.querySelectorAll('blockquote')
    expect(quotes).toHaveLength(2)
    expect(quotes[0].textContent).toBe(QUOTE_A)
    expect(quotes[1].textContent).toBe(QUOTE_B)
    expect(within(table).getByText('From the full text')).toBeVisible()
    expect(within(table).getByText('From the abstract')).toBeVisible()
    expect(within(table).getByText('Conflict')).toBeVisible()
    expect(within(table).getByText('Opposite findings on the same benchmark.')).toBeVisible()
    expect(within(table).getByRole('link', { name: 'Dense Passage Retrieval' })).toHaveAttribute(
      'href',
      'https://arxiv.org/abs/2004.04906',
    )

    expect(within(table).queryByRole('link', { name: 'Sparse Is Enough' })).toBeNull()
    for (const header of ['Papers', 'Verdict', 'Claim from A', 'Claim from B', 'Reason']) {
      expect(within(table).getByRole('columnheader', { name: header })).toBeVisible()
    }

    expect(screen.getByRole('status')).toHaveTextContent('1 of 3 pairs had comparable claims.')
    expect(screen.getByText('AI-generated verdicts').closest('p')).toHaveTextContent(
      'Quotes are verbatim. Check before citing.',
    )

    const none = screen.getByText('2 pairs with no comparable claims')
    expect(none.closest('details')).not.toHaveAttribute('open')
    await user.click(none)
    expect(screen.getByText(/the model found no pair of sentences addressing the same question/)).toBeVisible()
    expect(screen.getByText(/a paper has no usable abstract sentences/)).toBeVisible()
    expect(screen.getAllByText(/year unknown/).length).toBeGreaterThan(0)
  })

  it('opens the no-comparable list when no pair had comparable claims', async () => {
    const base = result_fixture()
    stub(result_fixture({ comparisons: base.comparisons.slice(1), n_pairs_requested: 2 }))
    const user = userEvent.setup()
    renderPanel()

    await user.click(screen.getByRole('button', { name: 'Run comparison' }))
    expect(await screen.findByRole('status')).toHaveTextContent('0 of 2 pairs had comparable claims.')
    expect(screen.queryByRole('table')).toBeNull()
    expect(screen.getByText('2 pairs with no comparable claims').closest('details')).toHaveAttribute('open')
  })

  it('shows the pairs that completed when the provider stopped the run', async () => {
    const base = result_fixture()
    stub(
      result_fixture({
        comparisons: base.comparisons.slice(0, 1),
        unavailable: { code: 'insufficient_quota', message: 'daily limit reached' },
      }),
    )
    const user = userEvent.setup()
    renderPanel()

    await user.click(screen.getByRole('button', { name: 'Run comparison' }))
    expect(await screen.findByText('Stopped early:')).toBeVisible()
    expect(screen.getByText(/quota exhausted/)).toHaveTextContent('1 pair completed.')
    expect(screen.getByRole('table').querySelectorAll('blockquote')[0].textContent).toBe(QUOTE_A)
  })

  it('says so when the run stopped before any pair completed', async () => {
    stub(result_fixture({ comparisons: [], unavailable: { code: 'no_api_key', message: 'no LLM API key is configured' } }))
    const user = userEvent.setup()
    renderPanel()

    await user.click(screen.getByRole('button', { name: 'Run comparison' }))
    expect(await screen.findByText(/No pair completed/)).toBeVisible()
    expect(screen.getByText(/No pair completed/)).toHaveTextContent('no API key')
    expect(screen.queryByRole('table')).toBeNull()
  })

  it.each([
    [422, 'invalid_request', 'choose between 2 and 8 papers; got 9'],
    [409, 'map_needs_more_papers', 'clusters need a map of at least 10 papers with vectors; this collection has 4'],
    [404, 'not_found', 'cluster 7 does not exist on this map'],
    [503, 'map_dependency_missing', 'umap-learn is not installed'],
  ])('reports a %i %s as a failed run', async (status, code, message) => {
    stub({ error: { code, message } }, status)
    const user = userEvent.setup()
    renderPanel()

    await user.click(screen.getByRole('button', { name: 'Run comparison' }))
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('The comparison did not run.')

    expect(alert).toHaveTextContent(message)
    expect(alert).not.toHaveTextContent(`${code}.`)
    expect(screen.queryByRole('table')).toBeNull()
    expect(screen.getByRole('button', { name: 'Run comparison again' })).toBeEnabled()
  })
})
