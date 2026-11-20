import { useEffect, useRef, useState } from 'react'

import { draftRelatedWork, messageFor, reasonCode, safeHref } from '../api/client'
import type { RelatedWork, RelatedWorkRequest } from '../api/types'
import { describeGenerationCode, generationReason } from '../lib/format'
import { Banner, LoadingState } from './States'

interface RelatedWorkDraftProps {
  collectionId: string
  request: RelatedWorkRequest

  subject: string
  onClose: () => void
}

export function RelatedWorkDraft({ collectionId, request, subject, onClose }: RelatedWorkDraftProps) {
  const [draft, setDraft] = useState<RelatedWork | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [copied, setCopied] = useState<'idle' | 'done' | 'failed'>('idle')
  const heading = useRef<HTMLHeadingElement | null>(null)
  const requestKey = JSON.stringify(request)

  const [draftFor, setDraftFor] = useState(`${collectionId}|${requestKey}`)
  if (draftFor !== `${collectionId}|${requestKey}`) {
    setDraftFor(`${collectionId}|${requestKey}`)
    setDraft(null)
    setError(null)
    setCopied('idle')
  }

  useEffect(() => {
    let alive = true
    draftRelatedWork(collectionId, JSON.parse(requestKey) as RelatedWorkRequest)
      .then((next) => {
        if (alive) setDraft(next)
      })
      .catch((caught: unknown) => {
        if (alive) setError(caught)
      })
    return () => {
      alive = false
    }
  }, [collectionId, requestKey])

  useEffect(() => {
    heading.current?.focus()
  }, [])

  async function copy() {
    if (!draft) return
    try {
      await navigator.clipboard.writeText(draft.paragraph)
      setCopied('done')
    } catch {
      setCopied('failed')
    }
  }

  const code = error ? reasonCode(error) : null

  return (
    <section className="generated related-work" aria-labelledby="related-work-heading">
      <div className="generated-head">
        <h2 id="related-work-heading" ref={heading} tabIndex={-1}>
          Related-work draft · {subject}
        </h2>
        <button type="button" className="button button-quiet" onClick={onClose}>
          Close draft
        </button>
      </div>

      {!draft && !error ? <LoadingState label="Drafting" /> : null}

      {error ? (
        <div role="alert">
          <Banner tone="warning" title="No draft was written.">
            {code && generationReason(code) ? `${describeGenerationCode(code)}.` : messageFor(error)}
          </Banner>
        </div>
      ) : null}

      {draft ? (
        <div aria-live="polite">
          <p className="ai-label">
            <span className="badge warn">AI-assisted draft</span> Verify before use.
          </p>

          {draft.unknown_citations.length > 0 ? (
            <Banner tone="danger" title="Unknown citations:">
              <span className="mono">{draft.unknown_citations.map((key) => `[${key}]`).join(' ')}</span>
            </Banner>
          ) : null}
          {draft.uncited_papers.length > 0 ? (
            <Banner tone="warning" title="Not cited:">
              <span className="mono">{draft.uncited_papers.map((key) => `[${key}]`).join(' ')}</span>
            </Banner>
          ) : null}

          <p className="generated-text">{draft.paragraph}</p>

          <div className="generated-actions">
            <button type="button" className="button button-secondary" onClick={() => void copy()}>
              Copy paragraph
            </button>
            <span className="muted" role="status">
              {copied === 'done'
                ? 'Copied.'
                : copied === 'failed'
                  ? 'Copy blocked. Select the text instead.'
                  : ''}
            </span>
          </div>

          <h3 className="generated-subhead">Cited papers, {draft.ordered_by}</h3>
          <ol className="source-list">
            {draft.citations.map((row) => {
              const href = safeHref(row.url)
              return (
                <li key={row.key} className="source-row">
                  <span className="mono">[{row.key}]</span>{' '}
                  {href ? (
                    <a href={href} target="_blank" rel="noreferrer">
                      {row.title}
                    </a>
                  ) : (
                    <span>{row.title}</span>
                  )}
                  <span className="muted"> · {row.year ?? 'year unknown'}</span>
                </li>
              )
            })}
          </ol>
          <p className="muted generated-meta">
            <span className="mono">{draft.word_count}</span> words · model{' '}
            <span className="mono">{draft.model}</span>
            {draft.cached ? ' · cached' : ''}
          </p>
        </div>
      ) : null}
    </section>
  )
}
