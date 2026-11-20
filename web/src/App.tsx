import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiError, deleteCollection, getHealth, listCollections, listPapers, messageFor } from './api/client'
import type { Collection, Health, Paper } from './api/types'
import { BuildDialog } from './components/BuildDialog'
import { AskView } from './components/AskView'
import { GraphView } from './components/GraphView'
import { MapView } from './components/MapView'
import { PaperDrawer } from './components/PaperDrawer'
import { PapersView } from './components/PapersView'
import { ScopeNote } from './components/ScopeNote'
import { Shell, VIEWS } from './components/Shell'
import { useMediaQuery } from './lib/useMediaQuery'
import type { View } from './components/Shell'
import { Banner, EmptyState, ErrorState, LoadingState } from './components/States'

const LOOKUP_PAGE = 200

const BUILDING_POLL_MS = 2000

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [collections, setCollections] = useState<Collection[] | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [view, setView] = useState<View>('papers')
  const [error, setError] = useState<string | null>(null)
  const [building, setBuilding] = useState(false)
  const [openPaper, setOpenPaper] = useState<string | null>(null)

  const dockPaper = useMediaQuery('(min-width: 1440px)')
  const [lookup, setLookup] = useState<{ id: string; byId: Record<string, Paper> }>({
    id: '',
    byId: {},
  })
  const [includeExternal, setIncludeExternal] = useState(false)

  const applyLoaded = useCallback(
    ([nextHealth, listed]: [Health, { collections: Collection[] }]) => {
      setHealth(nextHealth)
      setCollections(listed.collections)
      setError(null)
      setSelectedId((current) => {
        if (current && listed.collections.some((row) => row.id === current)) return current
        return listed.collections[0]?.id ?? null
      })
    },
    [],
  )
  const applyFailed = useCallback((caught: unknown) => setError(messageFor(caught)), [])

  const latestFetch = useRef(0)
  const fetchAll = useCallback(() => {
    const mine = ++latestFetch.current
    return Promise.all([getHealth(), listCollections()]).then(
      (result) => {
        if (mine === latestFetch.current) applyLoaded(result)
      },
      (caught: unknown) => {
        if (mine === latestFetch.current) applyFailed(caught)
      },
    )
  }, [applyLoaded, applyFailed])

  const load = useCallback(() => {
    setError(null)
    return fetchAll()
  }, [fetchAll])

  const created = useCallback(
    (collectionId: string) => {

      setSelectedId(collectionId)
      void load()
    },
    [load],
  )

  useEffect(() => {
    void fetchAll()
  }, [fetchAll])

  const selected = collections?.find((collection) => collection.id === selectedId) ?? null

  const readyId = selected?.status === 'ready' ? selected.id : null
  useEffect(() => {
    if (readyId === null) return
    let alive = true
    listPapers(readyId, { sort: 'influence', limit: LOOKUP_PAGE })
      .then((page) => {
        if (alive) {
          setLookup({
            id: readyId,
            byId: Object.fromEntries(page.papers.map((paper) => [paper.paper_id, paper])),
          })
        }
      })
      .catch(() => {
        if (alive) setLookup({ id: readyId, byId: {} })
      })
    return () => {
      alive = false
    }
  }, [readyId])
  const papersById = lookup.id === readyId ? lookup.byId : {}

  const active = VIEWS.find((item) => item.id === view) ?? VIEWS[0]
  const generationAvailable = health?.generation_available === true

  const buildRunning = collections?.some((collection) => collection.status === 'building') ?? false
  useEffect(() => {
    if (!buildRunning) return
    const timer = setInterval(() => void load(), BUILDING_POLL_MS)
    return () => clearInterval(timer)
  }, [buildRunning, load])

  return (
    <Shell
      collections={collections ?? []}
      selectedId={selectedId}
      onSelect={(collectionId) => {
        setSelectedId(collectionId)
        setOpenPaper(null)
      }}
      onBuild={() => setBuilding(true)}
      onDelete={async (collection) => {
        latestFetch.current++
        try {
          await deleteCollection(collection.id)
        } catch (caught) {

          if (!(caught instanceof ApiError && caught.code === 'not_found')) throw caught
        }
        setOpenPaper(null)

        await load()
      }}
      health={health}
      view={view}
      onViewChange={setView}
    >
      <header className="view-head">
        <div>
          <h1 className="view-title">{active.title}</h1>
          <p className="view-subtitle">{active.subtitle}</p>
        </div>
        {selected ? (
          <div className="mono muted" data-testid="collection-heading">
            {selected.query ?? selected.id}
          </div>
        ) : null}
      </header>

      {error ? (
        <ErrorState message={error} onRetry={() => void load()} />
      ) : collections === null ? (
        <LoadingState label="Loading collections" />
      ) : collections.length === 0 ? (
        <EmptyState title="No collections yet">Build one from a topic or your PDFs.</EmptyState>
      ) : !selected ? (
        <EmptyState title="No collection selected">Choose one on the left.</EmptyState>
      ) : selected.status !== 'ready' ? (

        <Banner
          tone={selected.status === 'failed' ? 'danger' : 'warning'}
          title={
            selected.status === 'failed'
              ? 'This collection failed to build.'
              : 'This collection is still building.'
          }
        >
          {selected.error ?? null}
        </Banner>
      ) : view === 'papers' ? (
        <div className={dockPaper ? 'papers-layout' : undefined}>
          <PapersView
            key={selected.id}
            collection={selected}
            onOpenPaper={setOpenPaper}
            generationAvailable={generationAvailable}
          />
          {dockPaper ? (
            openPaper ? (
              <PaperDrawer
                docked
                paperId={openPaper}
                onClose={() => setOpenPaper(null)}
                generationAvailable={generationAvailable}
              />
            ) : (
              <aside className="paper-pane-empty" aria-label="Paper details">
                Select a paper to see its details.
              </aside>
            )
          ) : null}
        </div>
      ) : view === 'map' ? (
        <MapView
          collectionId={selected.id}
          papersById={papersById}
          onSelect={setOpenPaper}
          generationAvailable={generationAvailable}
        />
      ) : view === 'graph' ? (
        <GraphView
          key={selected.id}
          collectionId={selected.id}
          includeExternal={includeExternal}
          onToggleExternal={setIncludeExternal}
          onSelect={setOpenPaper}
        />
      ) : (
        <div className="ask-page">
          <AskView key={selected.id} collectionId={selected.id} health={health} />
          <ScopeNote health={health} />
        </div>
      )}

      {openPaper && !(dockPaper && view === 'papers' && selected?.status === 'ready') ? (
        <PaperDrawer
          paperId={openPaper}
          onClose={() => setOpenPaper(null)}
          generationAvailable={generationAvailable}
        />
      ) : null}

      {building ? (
        <BuildDialog
          onClose={() => {
            setBuilding(false)

            void load()
          }}
          onCreated={created}
        />
      ) : null}
    </Shell>
  )
}
