import { useEffect, useRef, useState } from 'react'

import { ApiError, listPapers, messageFor } from '../api/client'
import type { Collection, Mode, Paper, PapersPage, Sort } from '../api/types'
import { degradationMessage } from '../lib/format'
import { PaperRow } from './PaperRow'
import { COMPARE_MAX, COMPARE_MIN, ContradictionTable } from './ContradictionTable'
import { RelatedWorkDraft } from './RelatedWorkDraft'
import { Banner, EmptyState, ErrorState, LoadingState } from './States'

export const PAGE_SIZE = 20

const MODE_OPTIONS: { value: Mode; label: string }[] = [
  { value: 'lexical', label: 'Lexical (BM25)' },
  { value: 'tfidf', label: 'TF-IDF cosine' },
  { value: 'dense', label: 'Dense (SPECTER2)' },
  { value: 'hybrid', label: 'Hybrid (RRF)' },
]

const SORT_OPTIONS: { value: Sort; label: string }[] = [
  { value: 'relevance', label: 'Relevance' },
  { value: 'influence', label: 'Citation count' },
  { value: 'year', label: 'Year' },
]

interface Request {
  q: string
  mode: Mode
  sort: Sort
}

interface Loaded {
  key: string
  page: PapersPage | null
  papers: Paper[]
  error: string | null
  notReady: ApiError | null
}

function requestKey(collectionId: string, status: string, request: Request): string {
  return JSON.stringify([collectionId, status, request.q, request.mode, request.sort])
}

export const DRAFT_MIN = 2
export const DRAFT_MAX = 12

function draftBlocker(count: number): string | null {
  if (count < DRAFT_MIN) {
    return `Select ${DRAFT_MIN} or more papers.`
  }
  if (count > DRAFT_MAX) {
    return `Drafts take up to ${DRAFT_MAX} papers.`
  }
  return null
}

function compareBlocker(count: number): string | null {
  if (count > COMPARE_MAX) {
    return `Comparisons take up to ${COMPARE_MAX} papers.`
  }
  return null
}

export function PapersView({
  collection,
  onOpenPaper,
  generationAvailable = false,
}: {
  collection: Collection
  onOpenPaper: (paperId: string) => void

  generationAvailable?: boolean
}) {
  const [selected, setSelected] = useState<string[]>([])
  const [draftIds, setDraftIds] = useState<string[] | null>(null)
  const [compareIds, setCompareIds] = useState<string[] | null>(null)
  const [queryText, setQueryText] = useState('')
  const [request, setRequest] = useState<Request>({ q: '', mode: 'lexical', sort: 'relevance' })
  const [loaded, setLoaded] = useState<Loaded | null>(null)
  const [loadingMore, setLoadingMore] = useState(false)
  const tokens = useRef(0)
  const key = requestKey(collection.id, collection.status, request)

  const loading = loaded?.key !== key
  const page = loaded?.page ?? null
  const papers = loaded?.papers ?? []
  const error = loading ? null : (loaded?.error ?? null)
  const notReady = loading ? null : (loaded?.notReady ?? null)

  useEffect(() => {
    const token = ++tokens.current
    const forKey = requestKey(collection.id, collection.status, request)
    listPapers(collection.id, {
      q: request.q || undefined,
      mode: request.mode,
      sort: request.sort,
      limit: PAGE_SIZE,
      offset: 0,
    })
      .then((next) => {
        if (tokens.current !== token) return
        setLoaded({ key: forKey, page: next, papers: next.papers, error: null, notReady: null })
      })
      .catch((caught: unknown) => {
        if (tokens.current !== token) return
        const blocked = caught instanceof ApiError && caught.code === 'not_ready'
        setLoaded({
          key: forKey,
          page: null,
          papers: [],
          error: blocked ? null : messageFor(caught),
          notReady: blocked ? caught : null,
        })
      })
  }, [collection.id, collection.status, request])

  async function loadMore() {
    if (!page || !loaded) return
    const token = tokens.current
    setLoadingMore(true)
    try {
      const next = await listPapers(collection.id, {
        q: request.q || undefined,
        mode: request.mode,
        sort: request.sort,
        limit: PAGE_SIZE,
        offset: papers.length,
      })
      if (tokens.current !== token) return
      setLoaded((current) =>
        current && { ...current, page: next, papers: [...current.papers, ...next.papers] },
      )
    } catch (caught) {
      if (tokens.current === token) {
        setLoaded((current) => current && { ...current, error: messageFor(caught) })
      }
    } finally {
      setLoadingMore(false)
    }
  }

  const maxScore = papers.reduce(
    (highest, paper) => (paper.score !== null && paper.score > highest ? paper.score : highest),
    0,
  )
  const total = page?.total ?? 0
  const hasMore = page !== null && papers.length < total
  const detail = notReady?.detail as { status?: string; error?: string } | null
  const blocker = draftBlocker(selected.length)
  const compareReason = compareBlocker(selected.length)
  const canCompare = selected.length >= COMPARE_MIN && selected.length <= COMPARE_MAX
  const toggle = (paperId: string, checked: boolean) =>
    setSelected((current) =>
      checked ? [...current.filter((id) => id !== paperId), paperId] : current.filter((id) => id !== paperId),
    )

  return (
    <section>
      <form
        className="toolbar"
        onSubmit={(event) => {
          event.preventDefault()
          setRequest((current) => ({ ...current, q: queryText.trim() }))
        }}
      >
        <div className="field grow">
          <label className="field-label" htmlFor="papers-query">
            Query
          </label>
          <input
            id="papers-query"
            className="input"
            value={queryText}
            onChange={(event) => setQueryText(event.target.value)}
            placeholder="Search this collection"
            autoComplete="off"
          />
        </div>

        <div className="field">
          <label className="field-label" htmlFor="papers-mode">
            Mode
          </label>
          <select
            id="papers-mode"
            className="select"
            value={request.mode}
            onChange={(event) =>
              setRequest((current) => ({ ...current, mode: event.target.value as Mode }))
            }
          >
            {MODE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label className="field-label" htmlFor="papers-sort">
            Sort
          </label>
          <select
            id="papers-sort"
            className="select"
            value={request.sort}
            onChange={(event) =>
              setRequest((current) => ({ ...current, sort: event.target.value as Sort }))
            }
          >
            {SORT_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        <button type="submit" className="button">
          Search
        </button>
      </form>

      {notReady ? (
        <Banner
          tone={detail?.status === 'failed' ? 'danger' : 'warning'}
          title={
            detail?.status === 'failed'
              ? 'This collection failed to build.'
              : 'This collection is still building.'
          }
        >
          {detail?.error ?? null}
        </Banner>
      ) : null}

      {error ? <ErrorState message={error} /> : null}

      {loading && papers.length === 0 ? <LoadingState label="Ranking papers" /> : null}

      {!loading && !error && !notReady && papers.length === 0 ? (
        <EmptyState title="No papers matched">
          {request.q
            ? `Nothing matches “${request.q}”.`
            : 'This collection has no papers.'}
        </EmptyState>
      ) : null}

      {papers.length > 0 && !notReady ? (
        <>
          <p className="muted result-line" aria-live="polite">
            <span className="mono">
              {papers.length} of {total}
            </span>{' '}
            papers
            {request.q ? (
              <>
                {' '}
                for “{request.q}” · <span className="mono">{page?.mode}</span>
              </>
            ) : null}
          </p>

          {(page?.degraded ?? []).map((code) => (
            <Banner key={code} tone="warning">
              {degradationMessage(code)}
            </Banner>
          ))}

          {generationAvailable ? (
            <div className="select-bar" role="group" aria-label="Related-work draft and comparison">
              <span>
                <span className="mono">{selected.length}</span> selected
              </span>
              <span className="muted select-reason">
                <span id="draft-blocker">{blocker ?? ''}</span>
                {blocker && compareReason ? ' ' : null}
                <span id="compare-blocker">{compareReason ?? ''}</span>
              </span>
              <span className="select-actions">
                {selected.length > 0 ? (
                  <button type="button" className="button button-quiet" onClick={() => setSelected([])}>
                    Clear selection
                  </button>
                ) : null}
                <button
                  type="button"
                  className="button button-secondary"
                  disabled={blocker !== null}
                  aria-describedby="draft-blocker"
                  onClick={() => setDraftIds([...selected])}
                >
                  Draft related work
                </button>
                <button
                  type="button"
                  className="button button-secondary"
                  disabled={!canCompare}
                  aria-describedby="draft-blocker compare-blocker"
                  onClick={() => setCompareIds([...selected])}
                >
                  Compare papers
                </button>
              </span>
            </div>
          ) : null}

          {draftIds ? (
            <RelatedWorkDraft
              key={draftIds.join(',')}
              collectionId={collection.id}
              request={{ paper_ids: draftIds }}
              subject={`${draftIds.length} selected papers`}
              onClose={() => setDraftIds(null)}
            />
          ) : null}

          {compareIds ? (
            <ContradictionTable
              key={compareIds.join(',')}
              collectionId={collection.id}
              request={{ paper_ids: compareIds }}
              subject={`${compareIds.length} selected papers`}
              paperCount={compareIds.length}
              onClose={() => setCompareIds(null)}
            />
          ) : null}

          <div className="paper-list">
            {papers.map((paper, index) => (
              <PaperRow
                key={`${paper.paper_id}-${index}`}
                paper={paper}
                rank={index + 1}
                maxScore={maxScore}
                onOpen={onOpenPaper}
                selection={
                  generationAvailable
                    ? {
                        checked: selected.includes(paper.paper_id),
                        onChange: (checked) => toggle(paper.paper_id, checked),
                      }
                    : undefined
                }
              />
            ))}
          </div>

          {hasMore ? (
            <div className="load-more">
              <button
                type="button"
                className="button button-secondary"
                onClick={() => void loadMore()}
                disabled={loadingMore}
              >
                {loadingMore ? 'Loading…' : `Load more (${total - papers.length} left)`}
              </button>
            </div>
          ) : null}
        </>
      ) : null}
    </section>
  )
}
