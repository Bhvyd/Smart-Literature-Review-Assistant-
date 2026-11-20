import type { Paper } from '../api/types'
import { formatAuthors, formatCount, formatScore, parseStatus } from '../lib/format'

interface PaperRowProps {
  paper: Paper
  rank: number
  maxScore: number
  onOpen: (paperId: string) => void

  selection?: { checked: boolean; onChange: (checked: boolean) => void }
}

export function PaperRow({ paper, rank, maxScore, onOpen, selection }: PaperRowProps) {
  const width = paper.score !== null && maxScore > 0 ? (paper.score / maxScore) * 100 : 0

  const parse =
    paper.source === 'upload' && (paper.parse_status === 'failed' || paper.parse_status === 'low_quality')
      ? parseStatus(paper.parse_status)
      : null

  const row = (
    <button type="button" className="paper-row" onClick={() => onOpen(paper.paper_id)}>
      <span className="paper-rank mono">{rank}</span>
      <span>
        <span className="paper-title">{paper.title}</span>
        <span className="paper-meta">
          <span>{paper.year ?? 'year unknown'}</span>
          <span>{formatAuthors(paper.authors)}</span>
          <span>{formatCount(paper.citation_count)} citations</span>
          {paper.venue ? <span>{paper.venue}</span> : null}
          {paper.source === 'upload' ? <span>uploaded PDF</span> : null}
          {parse ? (
            <span className={`badge ${parse.tone === 'bad' ? 'bad' : 'warn'}`}>{parse.label}</span>
          ) : null}
        </span>
        {parse && paper.parse_note ? <span className="paper-note">{paper.parse_note}</span> : null}
        {paper.abstract ? <span className="paper-abstract">{paper.abstract}</span> : null}
      </span>
      {paper.score !== null ? (
        <span className="paper-score">
          <span className="score-value">{formatScore(paper.score)}</span>
          <span className="score-track" aria-hidden="true">
            <span className="score-fill" style={{ width: `${width}%` }} />
          </span>
        </span>
      ) : (
        <span className="paper-score" />
      )}
    </button>
  )

  if (!selection) return row
  return (
    <div className="paper-item">
      <label className="paper-pick">
        <input
          type="checkbox"
          checked={selection.checked}
          onChange={(event) => selection.onChange(event.target.checked)}
        />
        <span className="visually-hidden">Select “{paper.title}” for a related-work draft</span>
      </label>
      {row}
    </div>
  )
}
