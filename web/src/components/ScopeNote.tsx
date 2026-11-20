import type { Health } from '../api/types'
import { MAX_UPLOAD_FILES, MAX_UPLOAD_MB, MAX_UPLOAD_TOTAL_MB } from '../lib/uploadLimits'
import { describeGenerationCode, generationReason } from '../lib/format'

export function ScopeNote({ health }: { health: Health | null }) {
  const lastError = health?.generation_last_error ?? null
  const reason = health?.generation_reason ?? null
  const model = health?.generation_model ?? health?.generation_provider ?? null

  return (
    <section className="scope-note" aria-labelledby="scope-heading">
      <h2 id="scope-heading">Scope</h2>
      <ul>
        <li>
          <strong>Generation:</strong>{' '}
          {health === null ? (
            'status unknown.'
          ) : health.generation_available ? (
            <>
              {model ? <span className="mono">{model}</span> : 'available'}
              {lastError ? <>, last call failed ({describeGenerationCode(lastError.code)})</> : null}.
            </>
          ) : (
            <>unavailable ({reason ? generationReason(reason) ?? reason : 'no reason given'}).</>
          )}
        </li>
        <li>
          <strong>Sources:</strong> arXiv and OpenAlex, or up to {MAX_UPLOAD_FILES} PDFs (
          {MAX_UPLOAD_MB} MB each, {MAX_UPLOAD_TOTAL_MB} MB total). Open access only.
        </li>
        <li>
          <strong>Compare papers:</strong> flags agreement or conflict between verbatim quotes. A
          lead to check, not a finding.
        </li>
        <li>
          <strong>Not covered:</strong> peer review, systematic-review (PRISMA) compliance,
          multi-user editing.
        </li>
      </ul>
    </section>
  )
}
