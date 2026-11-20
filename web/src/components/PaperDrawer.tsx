import { useEffect, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'

import { getPaper, messageFor, reasonCode, safeHref, summarisePaper } from '../api/client'
import type { PaperDetail, PaperSummary } from '../api/types'
import {
  describeGenerationCode,
  formatAuthors,
  formatCount,
  generationReason,
  parseStatus,
} from '../lib/format'
import { Banner, ErrorState, LoadingState } from './States'

interface PaperDrawerProps {
  paperId: string
  onClose: () => void

  generationAvailable?: boolean

  docked?: boolean
}

type SummaryState =
  | { state: 'idle' }
  | { state: 'loading' }
  | { state: 'done'; value: PaperSummary }
  | { state: 'failed'; error: unknown }

export function PaperDrawer({
  paperId,
  onClose,
  generationAvailable = false,
  docked = false,
}: PaperDrawerProps) {
  const [paper, setPaper] = useState<PaperDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [summary, setSummary] = useState<SummaryState>({ state: 'idle' })
  const summaryFor = useRef(paperId)
  const panel = useRef<HTMLDivElement | null>(null)
  const closeButton = useRef<HTMLButtonElement | null>(null)

  useEffect(() => {

    if (docked) return
    const previous = document.activeElement as HTMLElement | null

    if (closeButton.current) closeButton.current.focus()
    else panel.current?.focus()
    return () => previous?.focus()
  }, [docked])

  const [loadedFor, setLoadedFor] = useState(paperId)
  if (loadedFor !== paperId) {
    setLoadedFor(paperId)
    setPaper(null)
    setError(null)
    setSummary({ state: 'idle' })
  }

  useEffect(() => {
    let alive = true
    summaryFor.current = paperId
    getPaper(paperId)
      .then((detail) => {
        if (alive) setPaper(detail)
      })
      .catch((caught) => {
        if (alive) setError(messageFor(caught))
      })
    return () => {
      alive = false
    }
  }, [paperId])

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === 'Escape') {
      onClose()
      return
    }
    if (docked || event.key !== 'Tab' || !panel.current) return

    const focusable = panel.current.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])',
    )
    if (focusable.length === 0) return
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault()
      first.focus()
    }
  }

  async function summarise() {
    const requested = paperId
    setSummary({ state: 'loading' })
    try {
      const value = await summarisePaper(requested)
      if (summaryFor.current === requested) setSummary({ state: 'done', value })
    } catch (caught) {
      if (summaryFor.current === requested) setSummary({ state: 'failed', error: caught })
    }
  }

  const badge = paper ? parseStatus(paper.parse_status) : null
  const sourceHref = paper ? safeHref(paper.source_url) : null
  const pdfHref = paper ? safeHref(paper.pdf_url) : null
  const isUpload = paper?.source === 'upload'

  return (
    <>
      {docked ? null : <div className="drawer-backdrop" onMouseDown={onClose} />}
      <div
        className={docked ? 'drawer docked' : 'drawer'}
        role={docked ? 'region' : 'dialog'}
        aria-modal={docked ? undefined : true}
        aria-labelledby="paper-drawer-title"
        ref={panel}
        tabIndex={-1}
        onKeyDown={onKeyDown}
      >
        <div className="drawer-head">
          <div>
            <h2 className="drawer-title" id="paper-drawer-title">
              {paper?.title ?? 'Paper'}
            </h2>
          </div>
          <button type="button" className="button button-quiet" onClick={onClose} ref={closeButton}>
            Close
          </button>
        </div>

        <div className="drawer-body">
          {error ? <ErrorState message={error} /> : null}
          {!paper && !error ? <LoadingState label="Loading paper" /> : null}

          {paper ? (
            <>
              {paper.abstract ? (
                <p className="drawer-abstract">{paper.abstract}</p>
              ) : (
                <p className="muted">No abstract is stored for this paper.</p>
              )}

              <dl className="detail-list">
                <dt>Authors</dt>
                <dd>{formatAuthors(paper.authors, 12)}</dd>
                <dt>Year</dt>
                <dd>{paper.year ?? 'unknown'}</dd>
                <dt>Venue</dt>
                <dd>{paper.venue ?? 'unknown'}</dd>
                <dt>Citations</dt>
                <dd>{formatCount(paper.citation_count)}</dd>
                <dt>Text</dt>
                <dd>
                  <span className={`badge${badge?.tone === 'warn' ? ' warn' : ''}${badge?.tone === 'bad' ? ' bad' : ''}`}>
                    {badge?.label}
                  </span>
                  {paper.chunk_count > 0 ? (
                    <span className="muted"> · {paper.chunk_count} chunks indexed</span>
                  ) : null}
                </dd>
                <dt>Source</dt>
                <dd>{isUpload ? 'Uploaded PDF' : paper.source === 'arxiv' ? 'arXiv' : paper.source}</dd>
                <dt>arXiv</dt>
                <dd className="mono">{paper.arxiv_id ?? 'none'}</dd>
              </dl>

              {paper.has_fulltext === 0 ? (
                <Banner
                  tone={paper.parse_status === 'failed' ? 'warning' : 'info'}
                  title={
                    isUpload && paper.parse_status === 'failed'
                      ? 'This PDF could not be read.'
                      : 'Ranked from the abstract only.'
                  }
                >
                  {paper.parse_note ?? null}
                </Banner>
              ) : paper.parse_status === 'low_quality' && paper.parse_note ? (
                <Banner tone="warning" title="Low-quality text.">
                  {paper.parse_note}
                </Banner>
              ) : null}

              {sourceHref || (pdfHref && pdfHref !== sourceHref) ? (
                <p className="drawer-links">
                  {sourceHref ? (
                    <a href={sourceHref} target="_blank" rel="noreferrer">
                      {isUpload ? 'Open the uploaded PDF' : 'Open the source page'}
                    </a>
                  ) : null}
                  {pdfHref && pdfHref !== sourceHref ? (
                    <a href={pdfHref} target="_blank" rel="noreferrer">
                      Open the PDF
                    </a>
                  ) : null}
                </p>
              ) : null}

              {generationAvailable ? (
                <section className="generated" aria-labelledby="summary-heading">
                  <div className="generated-head">
                    <h3 id="summary-heading" className="generated-subhead">
                      Summary
                    </h3>
                    {summary.state === 'idle' || summary.state === 'failed' ? (
                      <button
                        type="button"
                        className="button button-secondary"
                        onClick={() => void summarise()}
                      >
                        {summary.state === 'failed' ? 'Try summarising again' : 'Summarise'}
                      </button>
                    ) : null}
                  </div>
                  <div aria-live="polite">
                    {summary.state === 'loading' ? <LoadingState label="Summarising" /> : null}
                    {summary.state === 'failed' ? (
                      <Banner tone="warning" title="No summary was written.">
                        {generationReason(reasonCode(summary.error) ?? '')
                          ? `${describeGenerationCode(reasonCode(summary.error) as string)}.`
                          : messageFor(summary.error)}
                      </Banner>
                    ) : null}
                    {summary.state === 'done' ? (
                      <>
                        <p className="ai-label">
                          <span className="badge warn">AI-generated</span> model{' '}
                          <span className="mono">{summary.value.model}</span>
                          {summary.value.cached ? ' · cached' : ''}
                        </p>
                        {summary.value.unsupported_numbers.length > 0 ? (
                          <Banner tone="danger" title="Numbers not found in the source text:">
                            <span className="mono">{summary.value.unsupported_numbers.join(', ')}</span>
                          </Banner>
                        ) : null}
                        <p className="generated-text summary-text">{summary.value.summary}</p>
                        <p className="muted generated-meta">
                          From {summary.value.sources.join(', ')}
                        </p>
                      </>
                    ) : null}
                  </div>
                </section>
              ) : null}
            </>
          ) : null}
        </div>
      </div>
    </>
  )
}
