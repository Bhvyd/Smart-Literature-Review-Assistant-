import { useEffect, useRef, useState } from 'react'

import { compareContradictions, messageFor, reasonCode, safeHref } from '../api/client'
import type { ComparedPaper, Comparison, ContradictionRequest, Contradictions } from '../api/types'
import { describeGenerationCode, generationReason } from '../lib/format'
import { Banner, LoadingState } from './States'

export const COMPARE_MIN = 2
export const COMPARE_MAX = 8
export const MAX_PAIRS = 10
const DEFAULT_PAIRS = 5
const NO_VERDICT = 'no_comparable_claims'

interface ContradictionTableProps {
  collectionId: string
  request: ContradictionRequest

  subject: string

  paperCount: number | null
  onClose: () => void
}

export function ContradictionTable({
  collectionId,
  request,
  subject,
  paperCount,
  onClose,
}: ContradictionTableProps) {
  const possible = paperCount !== null ? (paperCount * (paperCount - 1)) / 2 : null
  const pairCap = Math.max(1, Math.min(MAX_PAIRS, possible ?? MAX_PAIRS))
  const [pairsText, setPairsText] = useState(String(Math.min(DEFAULT_PAIRS, pairCap)))
  const [result, setResult] = useState<Contradictions | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [loading, setLoading] = useState(false)
  const heading = useRef<HTMLHeadingElement | null>(null)
  const token = useRef(0)

  useEffect(() => {
    heading.current?.focus()

    return () => {
      token.current += 1
    }
  }, [])

  const pairs = Number(pairsText)
  const pairsValid = Number.isInteger(pairs) && pairs >= 1 && pairs <= pairCap
  const calls = pairsValid ? pairs : null

  async function run() {
    if (calls === null) return
    const mine = ++token.current
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const next = await compareContradictions(collectionId, request, calls)
      if (token.current === mine) setResult(next)
    } catch (caught) {
      if (token.current === mine) setError(caught)
    } finally {
      if (token.current === mine) setLoading(false)
    }
  }

  const code = error ? reasonCode(error) : null

  return (
    <section className="generated compare" aria-labelledby="compare-heading">
      <div className="generated-head">
        <h2 id="compare-heading" ref={heading} tabIndex={-1}>
          Compare papers · {subject}
        </h2>
        <button type="button" className="button button-quiet" onClick={onClose}>
          Close comparison
        </button>
      </div>

      <div className="compare-run">
        <div className="field">
          <label className="field-label" htmlFor="compare-pairs">
            Pairs to compare
          </label>
          <input
            id="compare-pairs"
            className="input compare-pairs"
            type="number"
            inputMode="numeric"
            min={1}
            max={pairCap}
            step={1}
            value={pairsText}
            onChange={(event) => setPairsText(event.target.value)}
            aria-describedby="compare-cost"
            aria-invalid={!pairsValid}
          />
        </div>
        <p id="compare-cost" className="compare-cost">
          {calls !== null ? (
            <>
              Up to <strong className="mono">{calls * 2}</strong> model calls.
            </>
          ) : (
            <span className="warn-text">Enter a whole number from 1 to {pairCap}.</span>
          )}
          {possible !== null ? (
            <span className="muted">
              {' '}
              {possible} {possible === 1 ? 'pair' : 'pairs'} possible.
            </span>
          ) : null}
        </p>
        <button
          type="button"
          className="button button-secondary"
          disabled={calls === null || loading}
          aria-describedby="compare-cost"
          onClick={() => void run()}
        >
          {result || error ? 'Run comparison again' : 'Run comparison'}
        </button>
      </div>

      {loading ? <LoadingState label="Comparing" /> : null}

      {error ? (
        <div role="alert">
          <Banner tone="warning" title="The comparison did not run.">
            {code && generationReason(code) ? `${describeGenerationCode(code)}.` : messageFor(error)}
          </Banner>
        </div>
      ) : null}

      {result ? <Results result={result} /> : null}
    </section>
  )
}

function Results({ result }: { result: Contradictions }) {
  const comparable = result.comparisons.filter((row) => row.verdict !== NO_VERDICT)
  const none = result.comparisons.filter((row) => row.verdict === NO_VERDICT)
  const total = result.comparisons.length
  const models = [...new Set(result.comparisons.map((row) => row.model).filter(Boolean))]
  const cached = result.comparisons.filter((row) => row.cached).length

  return (
    <div className="compare-results">
      <p className="ai-label">
        <span className="badge warn">AI-generated verdicts</span> Quotes are verbatim. Check before
        citing.
      </p>

      {result.unavailable ? (
        <Banner tone="warning" title="Stopped early:">
          {describeGenerationCode(result.unavailable.code)}.{' '}
          {total > 0 ? `${total} ${total === 1 ? 'pair' : 'pairs'} completed.` : 'No pair completed.'}
        </Banner>
      ) : null}

      {total === 0 && !result.unavailable ? (
        <p className="muted">No pairs were compared.</p>
      ) : null}

      {total > 0 ? (
        <p className="compare-summary" role="status">
          <span className="mono">
            {comparable.length} of {total}
          </span>{' '}
          {total === 1 ? 'pair' : 'pairs'} had comparable claims.
        </p>
      ) : null}

      {comparable.length > 0 ? (
        <table className="compare-table">
          <caption className="visually-hidden">
            Pairs with comparable claims, most similar first
          </caption>
          <thead>
            <tr>
              <th scope="col">Papers</th>
              <th scope="col">Verdict</th>
              <th scope="col">Claim from A</th>
              <th scope="col">Claim from B</th>
              <th scope="col">Reason</th>
            </tr>
          </thead>
          <tbody>
            {comparable.map((row) => (
              <tr key={`${row.paper_a.paper_id}-${row.paper_b.paper_id}`}>
                <td data-label="Papers">
                  <PairTitles row={row} />
                </td>
                <td data-label="Verdict">
                  <VerdictMark row={row} />
                </td>
                <td data-label="Claim from A">
                  <Quote text={row.quote_a} source={row.source_a} />
                </td>
                <td data-label="Claim from B">
                  <Quote text={row.quote_b} source={row.source_b} />
                </td>
                <td data-label="Reason">
                  {row.reason ?? <span className="muted">No reason given.</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}

      {none.length > 0 ? (
        <details className="compare-none" open={comparable.length === 0}>
          <summary>
            {none.length} {none.length === 1 ? 'pair' : 'pairs'} with no comparable claims
          </summary>
          <ul>
            {none.map((row) => (
              <li key={`${row.paper_a.paper_id}-${row.paper_b.paper_id}`}>
                <VerdictMark row={row} />
                <PairTitles row={row} />
                {row.note ? <span className="muted compare-row-note">{row.note}.</span> : null}
              </li>
            ))}
          </ul>
        </details>
      ) : null}

      {models.length > 0 ? (
        <p className="muted generated-meta">
          Model <span className="mono">{models.join(', ')}</span>
          {cached > 0 ? ` · ${cached} of ${total} from the cache` : ''}
        </p>
      ) : null}
    </div>
  )
}

function PaperLink({ label, paper }: { label: string; paper: ComparedPaper }) {
  const href = safeHref(paper.url)
  return (
    <span className="compare-paper">
      <span className="mono compare-key">{label}</span>
      <span>
        {href ? (
          <a href={href} target="_blank" rel="noreferrer">
            {paper.title}
          </a>
        ) : (
          paper.title
        )}
        <span className="muted"> · {paper.year ?? 'year unknown'}</span>
      </span>
    </span>
  )
}

function PairTitles({ row }: { row: Comparison }) {
  return (
    <span className="compare-pair">
      <PaperLink label="A" paper={row.paper_a} />
      <PaperLink label="B" paper={row.paper_b} />
    </span>
  )
}

function Quote({ text, source }: { text: string | null; source?: string | null }) {
  if (!text) return <span className="muted">No sentence quoted.</span>
  return (
    <>
      <blockquote className="claim-quote">
        <p>{text}</p>
      </blockquote>
      {source ? <p className="muted claim-source">From the {source}</p> : null}
    </>
  )
}

const VERDICT_WORDS: Record<string, string> = {
  agree: 'Agree',
  conflict: 'Conflict',
  superseded: 'Superseded',
  [NO_VERDICT]: 'No comparable claims',
}

function VerdictMark({ row }: { row: Comparison }) {
  const verdict = row.verdict
  const known = verdict in VERDICT_WORDS
  const years = [row.paper_a.year, row.paper_b.year]
  const later =
    verdict === 'superseded' && years[0] !== null && years[1] !== null && years[0] !== years[1]
      ? Math.max(years[0], years[1])
      : null
  return (
    <span className={`verdict verdict-${known ? verdict.replace(/_/g, '-') : 'other'}`}>
      <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false">
        {verdict === 'agree' ? (
          <>
            <circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
            <path d="M5 8.2l2 2 4-4.4" fill="none" stroke="currentColor" strokeWidth="1.6" />
          </>
        ) : verdict === 'conflict' ? (
          <>
            <path d="M8 1.8l6.4 11.4H1.6z" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
            <path d="M8 6v3.6M8 11.2v.6" fill="none" stroke="currentColor" strokeWidth="1.6" />
          </>
        ) : verdict === 'superseded' ? (
          <path d="M2 8h10M8.5 4.2L12.3 8l-3.8 3.8M14 3.5v9" fill="none" stroke="currentColor" strokeWidth="1.6" />
        ) : (
          <>
            <circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.2" strokeDasharray="2 2" />
            <path d="M5.2 8h5.6" fill="none" stroke="currentColor" strokeWidth="1.4" />
          </>
        )}
      </svg>
      <span>{known ? VERDICT_WORDS[verdict] : <span className="mono">{verdict}</span>}</span>
      {later !== null ? (
        <span className="verdict-hint">the {later} paper&apos;s result replaces the earlier one</span>
      ) : null}
    </span>
  )
}
