import type { ReactNode } from 'react'

import type { Collection, Health } from '../api/types'
import { describeGenerationCode } from '../lib/format'
import { CollectionRail } from './CollectionRail'

export type View = 'papers' | 'map' | 'graph' | 'ask'

export const VIEWS: { id: View; label: string; title: string; subtitle: string }[] = [
  {
    id: 'papers',
    label: 'Papers',
    title: 'Ranked papers',
    subtitle: 'Ranked by the hand-written retrieval engine.',
  },
  {
    id: 'map',
    label: 'Map',
    title: 'Similarity map',
    subtitle: 'Nearby papers are similar in content.',
  },
  {
    id: 'graph',
    label: 'Citations',
    title: 'Citation graph',
    subtitle: 'Arrows point from the citing paper.',
  },
  {
    id: 'ask',
    label: 'Ask',
    title: 'Ask the collection',
    subtitle: 'Answers cite the papers they come from.',
  },
]

interface ShellProps {
  collections: Collection[]
  selectedId: string | null
  onSelect: (collectionId: string) => void
  onBuild: () => void
  onDelete: (collection: Collection) => Promise<void>
  health: Health | null
  view: View
  onViewChange: (view: View) => void
  children: ReactNode
}

export function Shell({
  collections,
  selectedId,
  onSelect,
  onBuild,
  onDelete,
  health,
  view,
  onViewChange,
  children,
}: ShellProps) {
  const generationAvailable = health?.generation_available === true

  return (
    <div className="app">
      <aside className="rail">
        <div>
          <div className="wordmark">Literature Review</div>
        </div>

        <CollectionRail
          collections={collections}
          selectedId={selectedId}
          onSelect={onSelect}
          onBuild={onBuild}
          onDelete={onDelete}
        />

        <nav className="rail-nav" aria-label="Views">
          {VIEWS.map((item) => (
            <button
              key={item.id}
              type="button"
              className="nav-item"
              aria-current={view === item.id ? 'page' : undefined}
              onClick={() => onViewChange(item.id)}
            >
              <span>{item.label}</span>
              {item.id === 'ask' && !generationAvailable ? (
                <span className="nav-note">unavailable</span>
              ) : null}
            </button>
          ))}
        </nav>

        <div className="rail-footer">
          <span className="mono" data-testid="generation-provider">
            {health?.generation_provider
              ? [health.generation_provider, health.generation_model].filter(Boolean).join(' · ')
              : 'no generation provider'}
          </span>
          {health !== null && !generationAvailable ? (
            <span className="warn-text">
              Generation unavailable
              {health.generation_reason ? ` (${describeGenerationCode(health.generation_reason)})` : ''}
            </span>
          ) : health?.generation_last_error ? (
            <span className="warn-text">
              {describeGenerationCode(health.generation_last_error.code)}
            </span>
          ) : null}
          {health?.generation_config_note ? (
            <span className="warn-text" role="note">
              {health.generation_config_note}
            </span>
          ) : null}
        </div>
      </aside>

      <main className="main">{children}</main>
    </div>
  )
}
