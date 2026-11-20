import { useEffect, useRef, useState, type FormEvent, type ReactNode } from 'react'

import { ask, listPapers, messageFor, safeHref } from '../api/client'
import type { AskCitation, AskResult, Health, PapersPage } from '../api/types'
import {
  degradationMessage,
  describeGenerationCode,
  formatAuthors,
  formatScore,
} from '../lib/format'
import { Banner, ErrorState } from './States'

const PREVIEW_SIZE = 8
const MIN_QUESTION = 3
const MAX_QUESTION = 1000

const REASON_COPY: Record<string, string> = {
  not_implemented: 'Not available in this build.',
  no_api_key: 'No API key is configured.',
}

const REFUSAL_COPY: Record<string, string> = {
  no_retrieval_hits: 'Nothing in this collection matches the question.',
  below_threshold: 'No passage is relevant enough to answer from.',
  model_declined: 'The collection does not cover this question.',
  answer_had_no_valid_citation: 'The answer cited no source, so it was withheld.',
}

interface AskViewProps {
  collectionId: string
  health: Health | null
}

function sourceId(n: number): string {
  return `ask-source-${n}`
}

function focusSource(n: number) {
  const card = document.getElementById(sourceId(n))
  if (!card) return
  card.focus()
  card.scrollIntoView?.({ block: 'nearest' })
}

function AnswerText({ text, citations }: { text: string; citations: AskCitation[] }) {
  const byNumber = new Map(citations.map((citation) => [citation.n, citation]))
  const parts: ReactNode[] = []
  text.split(/(\[\d+\])/).forEach((part, index) => {
    const marker = /^\[(\d+)\]$/.exec(part)
    const citation = marker ? byNumber.get(Number(marker[1])) : undefined
    if (marker && citation) {
      parts.push(
        <button
          key={index}
          type="button"
          className="cite-marker"
          aria-label={`Source ${citation.n}: ${citation.title}`}
          onClick={() => focusSource(citation.n)}
        >
          {citation.n}
        </button>,
      )
    } else if (part) {
      parts.push(part)
    }
  })
  return <p className="generated-text">{parts}</p>
}

function SourceList({ citations }: { citations: AskCitation[] }) {
  return (
    <ol className="source-list">
      {citations.map((citation) => {
        const href = safeHref(citation.url)
        return (
          <li
            key={citation.n}
            id={sourceId(citation.n)}
            className="source-card"
            tabIndex={-1}
            aria-label={`Source ${citation.n}`}
          >
            <div className="source-head">
              <span className="source-number mono">{citation.n}</span>
              <span>
                {href ? (
                  <a href={href} target="_blank" rel="noreferrer" className="source-title">
                    {citation.title}
                  </a>
                ) : (
                  <span className="source-title">{citation.title}</span>
                )}
                <span className="muted"> · {citation.year ?? 'year unknown'}</span>
              </span>
            </div>
            <blockquote className="source-quote">{citation.quote}</blockquote>
          </li>
        )
      })}
    </ol>
  )
}

function AnswerView({ result }: { result: AskResult }) {
  const retrieval = result.retrieval

  const modelCompleted =
    !result.answer_unavailable &&
    result.refusal_reason !== 'no_retrieval_hits' &&
    result.refusal_reason !== 'below_threshold'
  return (
    <div className="ask-results" aria-live="polite">
      {result.answer_unavailable ? (
        <>
          <Banner tone="warning" title="No answer generated:">
            {describeGenerationCode(result.answer_unavailable.code)}.
          </Banner>
          <h2>Retrieved evidence</h2>
          {result.citations.length > 0 ? (
            <SourceList citations={result.citations} />
          ) : (
            <p className="muted">No passages were retrieved.</p>
          )}
        </>
      ) : result.refused ? (
        <>
          <h2>No answer</h2>
          <p className="muted">
            {(result.refusal_reason && REFUSAL_COPY[result.refusal_reason]) ??
              `Refused (${result.refusal_reason ?? 'no reason given'}).`}
          </p>
          {result.answer ? <p className="muted">{result.answer}</p> : null}
          {result.citations.length > 0 ? <SourceList citations={result.citations} /> : null}
        </>
      ) : (
        <>
          <h2>Answer</h2>
          <p className="ai-label">
            <span className="badge warn">AI-generated</span>
          </p>
          {result.answer ? <AnswerText text={result.answer} citations={result.citations} /> : null}
          {result.confidence !== null ? (
            <p className="muted ask-coverage">
              <span className="mono">{Math.round(result.confidence * 100)}%</span> of sentences cite a
              source
            </p>
          ) : null}
          {result.citation_anomaly.length > 0 ? (
            <Banner tone="warning" title="Invalid citations removed:">
              <span className="mono">{result.citation_anomaly.join(', ')}</span>
            </Banner>
          ) : null}
          <h3 className="generated-subhead">Sources</h3>
          <SourceList citations={result.citations} />
        </>
      )}

      {retrieval.degraded.map((code) => (
        <Banner key={code} tone="warning">
          {degradationMessage(code)}
        </Banner>
      ))}
      <p className="muted generated-meta">
        <span className="mono">{retrieval.mode}</span> ·{' '}
        <span className="mono">{retrieval.n_passages}</span> passages ·{' '}
        {modelCompleted ? (
          <>
            model <span className="mono">{result.model}</span>
            {result.cached ? ' · cached' : ''}
          </>
        ) : result.answer_unavailable ? (
          'no model output'
        ) : (
          'no model call'
        )}{' '}
        · <span className="mono">{result.latency_ms}</span> ms
      </p>
    </div>
  )
}

export function AskView({ collectionId, health }: AskViewProps) {
  const [question, setQuestion] = useState('')
  const [preview, setPreview] = useState<PapersPage | null>(null)
  const [result, setResult] = useState<AskResult | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [loading, setLoading] = useState(false)
  const tokens = useRef(0)

  const available = health?.generation_available === true
  const lastError = health?.generation_last_error ?? null

  const [shownFor, setShownFor] = useState(collectionId)
  if (shownFor !== collectionId) {
    setShownFor(collectionId)
    setPreview(null)
    setResult(null)
    setError(null)
    setLoading(false)
  }
  useEffect(() => {

    tokens.current += 1
  }, [collectionId])

  const reason = health?.generation_reason ?? null
  const reasonCopy = (reason && REASON_COPY[reason]) ?? 'Generation is not set up.'
  const trimmed = question.trim()

  async function run<T>(load: () => Promise<T>, keep: (value: T) => void) {
    const token = ++tokens.current
    setLoading(true)
    setError(null)
    setPreview(null)
    setResult(null)
    try {
      const next = await load()
      if (tokens.current === token) keep(next)
    } catch (caught) {
      if (tokens.current === token) setError(caught)
    } finally {
      if (tokens.current === token) setLoading(false)
    }
  }

  function onPreview(event: FormEvent) {
    event.preventDefault()
    if (!trimmed) return
    void run(
      () => listPapers(collectionId, { q: trimmed, mode: 'hybrid', limit: PREVIEW_SIZE }),
      setPreview,
    )
  }

  function onAsk(event: FormEvent) {
    event.preventDefault()
    if (trimmed.length < MIN_QUESTION) return
    void run(() => ask(collectionId, trimmed), setResult)
  }

  return (
    <section className="ask">
      {available ? (
        lastError && !result ? (
          <Banner tone="warning" title="Last generation call failed:">
            {describeGenerationCode(lastError.code)}.
          </Banner>
        ) : null
      ) : (
        <Banner tone="warning" title="Answers unavailable.">
          {reasonCopy} You can still preview the evidence.
        </Banner>
      )}

      {available ? (
        <form className="ask-preview" onSubmit={onAsk}>
          <label htmlFor="ask-question">Question</label>
          <textarea
            id="ask-question"
            name="question"
            className="ask-textarea"
            rows={3}
            maxLength={MAX_QUESTION}
            value={question}
            placeholder="How does retrieval augmentation reduce hallucination?"
            aria-describedby="ask-question-hint"
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
                event.preventDefault()
                event.currentTarget.form?.requestSubmit()
              }
            }}
          />
          <div className="ask-row">
            <span id="ask-question-hint" className="field-hint">
              Ctrl+Enter to ask
            </span>
            <button
              type="submit"
              className="button"
              disabled={loading || trimmed.length < MIN_QUESTION}
            >
              {loading ? 'Answering…' : 'Ask'}
            </button>
          </div>
        </form>
      ) : (
        <form className="ask-preview" onSubmit={onPreview}>
          <label htmlFor="ask-question">Question</label>
          <div className="ask-row">
            <input
              id="ask-question"
              name="question"
              value={question}
              placeholder="How does retrieval augmentation reduce hallucination?"
              onChange={(event) => setQuestion(event.target.value)}
            />
            <button type="submit" className="button" disabled={loading || trimmed === ''}>
              {loading ? 'Retrieving…' : 'Preview retrieved evidence'}
            </button>
          </div>
        </form>
      )}

      {error ? <ErrorState message={messageFor(error)} /> : null}

      {result ? <AnswerView result={result} /> : null}

      {preview ? (
        <div className="ask-results" aria-live="polite">
          <h2>Retrieved papers</h2>
          {preview.papers.length === 0 ? (
            <p className="muted">Nothing in this collection matches.</p>
          ) : (
            <ol className="preview-list">
              {preview.papers.map((row) => (
                <li key={row.paper_id} className="preview-row">
                  <span className="preview-title">{row.title}</span>
                  <span className="muted preview-meta">
                    {formatAuthors(row.authors)} · {row.year ?? 'n.d.'} · score{' '}
                    <span className="mono">{formatScore(row.score)}</span>
                  </span>
                </li>
              ))}
            </ol>
          )}
        </div>
      ) : null}
    </section>
  )
}
