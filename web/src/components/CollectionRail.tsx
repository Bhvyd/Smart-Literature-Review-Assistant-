import { useEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'

import { cancelCollectionBuild, messageFor } from '../api/client'
import type { Collection } from '../api/types'
import { degradationMessage } from '../lib/format'

interface CollectionRailProps {
  collections: Collection[]
  selectedId: string | null
  onSelect: (collectionId: string) => void
  onBuild: () => void

  onDelete: (collection: Collection) => Promise<void>
}

function statusLabel(status: string): string {
  if (status === 'ready') return 'ready'
  if (status === 'building') return 'building'
  if (status === 'failed') return 'failed'
  return status
}

function buildProgress(collection: Collection): { label: string; count: string | null } {
  const job = collection.job
  if (!job || job.status === 'queued') return { label: job ? 'queued' : 'building', count: null }
  return { label: 'building', count: job.message?.match(/\d+\/\d+/)?.[0] ?? null }
}

function modeLabel(mode: string): string {
  if (mode === 'upload') return 'PDF upload'
  if (mode === 'topic') return 'topic search'
  return mode
}

const FOCUSABLE = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'

interface ConfirmCopy {
  title: string
  body: ReactNode
  confirm: string
  busy: string
  dismiss: string
}

function deleteCopy(collection: Collection): ConfirmCopy {
  return {
    title: `Delete “${collection.query ?? collection.id}”?`,
    body: (
      <>
        This permanently removes the collection and its{' '}
        <span className="mono">{collection.n_papers}</span>{' '}
        {collection.n_papers === 1 ? 'paper' : 'papers'}, with their embeddings, map and citation
        graph
        {collection.mode === 'upload' ? ', and the uploaded PDF files stored for it' : ''}. It cannot
        be undone.
      </>
    ),
    confirm: 'Delete collection',
    busy: 'Deleting…',
    dismiss: 'Cancel',
  }
}

function stopCopy(collection: Collection): ConfirmCopy {
  return {
    title: `Stop building “${collection.query ?? collection.id}”?`,
    body: <>Papers fetched so far are discarded.</>,
    confirm: 'Cancel build',
    busy: 'Cancelling…',
    dismiss: 'Keep building',
  }
}

function ConfirmDialog({
  copy,
  onCancel,
  onConfirm,
}: {
  copy: ConfirmCopy
  onCancel: () => void
  onConfirm: () => Promise<void>
}) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const dialogRef = useRef<HTMLDivElement | null>(null)
  const cancelRef = useRef<HTMLButtonElement | null>(null)
  const busyRef = useRef(false)
  const pressedOnBackdrop = useRef(false)
  const cancelRequest = useRef(onCancel)

  useEffect(() => {
    cancelRequest.current = onCancel
  }, [onCancel])

  useEffect(() => {
    cancelRef.current?.focus()
  }, [])

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        if (!busyRef.current) cancelRequest.current()
        return
      }
      if (event.key !== 'Tab') return
      const items = Array.from(
        dialogRef.current?.querySelectorAll<HTMLElement>(FOCUSABLE) ?? [],
      ).filter((item) => !item.hasAttribute('disabled'))
      if (items.length === 0) return
      const first = items[0]
      const last = items[items.length - 1]
      const active = document.activeElement
      if (!dialogRef.current?.contains(active)) {
        event.preventDefault()
        first.focus()
      } else if (event.shiftKey && active === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && active === last) {
        event.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [])

  async function confirm() {
    if (busyRef.current) return
    busyRef.current = true
    setBusy(true)
    setError(null)
    try {
      await onConfirm()
    } catch (caught) {
      setError(messageFor(caught))
      busyRef.current = false
      setBusy(false)
    }
  }

  return createPortal(
    <div
      className="dialog-backdrop"
      onMouseDown={(event) => {
        pressedOnBackdrop.current = event.target === event.currentTarget
      }}

      onClick={(event) => {
        if (event.target === event.currentTarget && pressedOnBackdrop.current && !busy) onCancel()
      }}
    >
      <div
        ref={dialogRef}
        className="dialog"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        aria-describedby="confirm-body"
      >
        <h2 id="confirm-title">{copy.title}</h2>
        <p id="confirm-body">{copy.body}</p>
        {error ? (
          <div className="banner danger" role="alert">
            <div className="banner-body">{error}</div>
          </div>
        ) : null}
        <div className="dialog-actions">
          <button
            type="button"
            className="button"
            ref={cancelRef}
            aria-disabled={busy}
            onClick={() => {
              if (!busy) onCancel()
            }}
          >
            {copy.dismiss}
          </button>
          <button
            type="button"
            className="button button-secondary button-danger"
            aria-disabled={busy}
            onClick={() => void confirm()}
          >
            {busy ? copy.busy : copy.confirm}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  )
}

export function CollectionRail({
  collections,
  selectedId,
  onSelect,
  onBuild,
  onDelete,
}: CollectionRailProps) {
  const [confirming, setConfirming] = useState<{ collection: Collection; action: 'delete' | 'stop' } | null>(
    null,
  )

  const [stopping, setStopping] = useState<ReadonlySet<string>>(new Set())
  const railRef = useRef<HTMLDivElement | null>(null)
  const opener = useRef<HTMLElement | null>(null)

  const restore = useRef<'opener' | 'selected' | null>(null)

  useEffect(() => {
    if (confirming !== null || restore.current === null) return
    const target =
      restore.current === 'opener' && opener.current?.isConnected
        ? opener.current
        : (railRef.current?.querySelector<HTMLElement>('.collection-item[aria-current="true"]') ??
          railRef.current?.querySelector<HTMLElement>('.rail-action'))
    restore.current = null
    target?.focus()
  }, [confirming])

  return (
    <div className="rail-section" ref={railRef}>
      <div className="rail-label" id="collections-label">
        Collections
      </div>

      {collections.length === 0 ? (
        <p className="rail-empty">None yet</p>
      ) : (
        <ul className="collection-list" aria-labelledby="collections-label">
          {collections.map((collection) => {
            const current = collection.id === selectedId
            const degraded = collection.degraded ?? []
            return (
              <li key={collection.id}>
                <button
                  type="button"
                  className="collection-item"
                  aria-current={current ? 'true' : undefined}
                  onClick={() => onSelect(collection.id)}
                >
                  <span className="collection-query">{collection.query ?? collection.id}</span>
                  <span className="collection-line">
                    <span className={`status-dot ${collection.status}`} aria-hidden="true" />
                    {stopping.has(collection.id) && collection.status === 'building'
                      ? 'cancelling'
                      : collection.status === 'building'
                        ? buildProgress(collection).label
                        : statusLabel(collection.status)}
                    {collection.status === 'building' && buildProgress(collection).count ? (
                      <span className="mono"> · {buildProgress(collection).count}</span>
                    ) : null}
                    {collection.status === 'ready' ? (
                      <span className="mono"> · {collection.n_papers} papers</span>
                    ) : null}
                    <span> · {modeLabel(collection.mode)}</span>
                  </span>
                  {collection.error ? (
                    <span className="collection-error">{collection.error}</span>
                  ) : null}
                </button>
                {current && collection.status === 'building' ? (
                  stopping.has(collection.id) ? null : (
                    <button
                      type="button"
                      className="button button-quiet button-danger-text"
                      aria-label="Cancel the build of the selected collection"
                      onClick={(event) => {
                        opener.current = event.currentTarget
                        setConfirming({ collection, action: 'stop' })
                      }}
                    >
                      Cancel build
                    </button>
                  )
                ) : current ? (
                  <button
                    type="button"
                    className="button button-quiet button-danger-text"
                    aria-label="Delete the selected collection"
                    onClick={(event) => {
                      opener.current = event.currentTarget
                      setConfirming({ collection, action: 'delete' })
                    }}
                  >
                    Delete
                  </button>
                ) : null}
                {degraded.length > 0 ? (
                  <details className="degraded">
                    <summary>
                      <span className="badge warn">reduced-function</span>
                      <span className="visually-hidden">
                        {' '}
                        build, {degraded.length} {degraded.length === 1 ? 'reason' : 'reasons'}
                      </span>
                    </summary>
                    <ul>
                      {degraded.map((code) => (
                        <li key={code}>
                          {degradationMessage(code)} <span className="mono">({code})</span>
                        </li>
                      ))}
                    </ul>
                  </details>
                ) : null}
              </li>
            )
          })}
        </ul>
      )}

      <button type="button" className="button button-secondary rail-action" onClick={onBuild}>
        Build a collection
      </button>

      {confirming ? (
        <ConfirmDialog
          copy={confirming.action === 'stop' ? stopCopy(confirming.collection) : deleteCopy(confirming.collection)}
          onCancel={() => {
            restore.current = 'opener'
            setConfirming(null)
          }}
          onConfirm={async () => {
            if (confirming.action === 'stop') {
              await cancelCollectionBuild(confirming.collection.id)
              setStopping((previous) => new Set(previous).add(confirming.collection.id))
            } else {
              await onDelete(confirming.collection)
            }
            restore.current = 'selected'
            setConfirming(null)
          }}
        />
      ) : null}
    </div>
  )
}
