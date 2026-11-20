import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'

import { ApiError, createCollection, getJob, messageFor, uploadCollection } from '../api/client'
import type { Job } from '../api/types'
import { MAX_UPLOAD_FILES, MAX_UPLOAD_MB, MAX_UPLOAD_TOTAL_MB } from '../lib/uploadLimits'
import { Banner } from './States'

export const POLL_MS = 1000
const EMPTY_TOPIC_ERROR = 'Enter a topic to build.'
const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
const MAX_UPLOAD_TOTAL_BYTES = MAX_UPLOAD_TOTAL_MB * 1024 * 1024

type BuildMode = 'topic' | 'upload'

function megabytes(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(1)
}

function uploadProblem(files: File[]): string | null {
  if (files.length === 0) return 'Choose at least one PDF.'
  if (files.length > MAX_UPLOAD_FILES) {
    return `Up to ${MAX_UPLOAD_FILES} PDFs (you chose ${files.length}).`
  }
  for (const file of files) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      return `${file.name} is not a PDF.`
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      return `${file.name} is over ${MAX_UPLOAD_MB} MB.`
    }
  }
  const total = files.reduce((sum, file) => sum + file.size, 0)
  if (total > MAX_UPLOAD_TOTAL_BYTES) {
    return `The files total ${megabytes(total)} MB, over the ${MAX_UPLOAD_TOTAL_MB} MB limit.`
  }
  return null
}

interface BuildDialogProps {
  onClose: () => void

  onCreated: (collectionId: string) => void

  pollMs?: number
}

export function BuildDialog({ onClose, onCreated, pollMs = POLL_MS }: BuildDialogProps) {
  const [mode, setMode] = useState<BuildMode>('topic')
  const [query, setQuery] = useState('')
  const [maxPapers, setMaxPapers] = useState(100)
  const [files, setFiles] = useState<File[]>([])
  const [name, setName] = useState('')
  const [jobId, setJobId] = useState<string | null>(null)
  const [collectionId, setCollectionId] = useState<string | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)

  const alive = useRef(true)
  const settled = useRef(false)
  const onCreatedRef = useRef(onCreated)
  const onCloseRef = useRef(onClose)
  const queryRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    onCreatedRef.current = onCreated
    onCloseRef.current = onClose
  }, [onCreated, onClose])

  useEffect(() => {
    alive.current = true
    queryRef.current?.focus()
    return () => {
      alive.current = false
    }
  }, [])

  useEffect(() => {
    if (!jobId || !collectionId) return

    let timer: ReturnType<typeof setInterval> | undefined
    const stop = () => {
      if (timer !== undefined) {
        clearInterval(timer)
        timer = undefined
      }
    }

    const finish = (collection: string) => {
      if (settled.current) return
      settled.current = true
      onCreatedRef.current(collection)
      onCloseRef.current()
    }

    const poll = async () => {
      try {
        const next = await getJob(jobId)
        if (!alive.current) return
        setJob(next)
        if (next.status === 'done') {
          finish(collectionId)
          stop()
          return
        }
        if (next.status === 'failed') {
          settled.current = true
          setError(next.error ?? 'The build failed.')
          stop()
          return
        }

        setError(null)
      } catch (caught) {

        if (!alive.current) return
        setError(`Lost track of the build: ${messageFor(caught)}`)
      }
    }

    timer = setInterval(() => {
      void poll()
    }, pollMs)
    void poll()
    return () => stop()
  }, [jobId, collectionId, pollMs])

  const topicInvalid = error === EMPTY_TOPIC_ERROR

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    let start: () => Promise<{ collection_id: string; job_id: string }>
    if (mode === 'topic') {
      const topic = query.trim()
      if (!topic) {
        setError(EMPTY_TOPIC_ERROR)
        queryRef.current?.focus()
        return
      }
      start = () => createCollection(topic, maxPapers)
    } else {
      const problem = uploadProblem(files)
      if (problem) {
        setError(problem)
        return
      }
      start = () => uploadCollection(files, name)
    }

    setError(null)
    setStarting(true)
    try {
      const created = await start()
      if (!alive.current) return
      settled.current = false
      setCollectionId(created.collection_id)
      setJobId(created.job_id)
    } catch (caught) {
      if (!alive.current) return
      if (caught instanceof ApiError && caught.code === 'build_in_progress') {

        const running = (caught.detail as { collection_id?: string } | null)?.collection_id
        if (running) {
          onCreated(running)
          onClose()
          return
        }
      }
      setError(messageFor(caught))
    } finally {
      if (alive.current) setStarting(false)
    }
  }

  const running = job !== null && (job.status === 'queued' || job.status === 'running')
  const locked = starting || running

  const blip = jobId !== null && job?.status !== 'failed'
  const progress = Math.round((job?.progress ?? 0) * 100)
  const totalBytes = files.reduce((sum, file) => sum + file.size, 0)

  return (
    <div
      className="dialog-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <div
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="build-title"
        onKeyDown={(event) => {
          if (event.key === 'Escape') onClose()
        }}
      >
        <h2 id="build-title">Build a collection</h2>
        <p className="muted">Runs in the background and can take several minutes.</p>

        <form className="dialog-form" onSubmit={submit} noValidate>
          <fieldset className="segmented" disabled={locked}>
            <legend className="field-label">Source</legend>
            {(
              [
                ['topic', 'Topic search'],
                ['upload', 'Upload PDFs'],
              ] as const
            ).map(([value, label]) => (
              <label key={value} className="segment">
                <input
                  type="radio"
                  name="build-mode"
                  value={value}
                  checked={mode === value}
                  onChange={() => {
                    setMode(value)
                    setError(null)
                  }}
                />
                <span>{label}</span>
              </label>
            ))}
          </fieldset>

          {mode === 'topic' ? (
            <>
              <div className="field">
                <label className="field-label" htmlFor="build-query">
                  Topic
                </label>
                <input
                  id="build-query"
                  ref={queryRef}
                  className="input"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="retrieval augmented generation"
                  autoComplete="off"
                  aria-invalid={topicInvalid || undefined}
                  aria-describedby={topicInvalid ? 'build-error' : undefined}
                />
              </div>

              <div className="field">
                <label className="field-label" htmlFor="build-max">
                  Papers to fetch (10 to 200)
                </label>
                <input
                  id="build-max"
                  className="input"
                  type="number"
                  min={10}
                  max={200}
                  value={maxPapers}
                  onChange={(event) => setMaxPapers(Number(event.target.value) || 0)}
                />
              </div>
            </>
          ) : (
            <>
              <div className="field">
                <label className="field-label" htmlFor="build-files">
                  PDF files (up to {MAX_UPLOAD_FILES}, {MAX_UPLOAD_MB} MB each, {MAX_UPLOAD_TOTAL_MB} MB total)
                </label>
                <input
                  id="build-files"
                  className="file-input"
                  type="file"
                  accept=".pdf,application/pdf"
                  multiple
                  disabled={locked}
                  aria-describedby="build-files-summary"
                  onChange={(event) => {
                    const chosen = Array.from(event.target.files ?? [])
                    setFiles(chosen)
                    setError(chosen.length > 0 ? uploadProblem(chosen) : null)
                  }}
                />
                <span id="build-files-summary" className="field-hint">
                  {files.length === 0
                    ? ''
                    : `${files.length} ${files.length === 1 ? 'file' : 'files'}, ${megabytes(totalBytes)} MB in total.`}
                </span>
                {files.length > 0 ? (
                  <ul className="file-list" aria-label="Chosen files">
                    {files.map((file, index) => (
                      <li key={`${file.name}-${index}`}>
                        <span className="file-name">{file.name}</span>
                        <span className="mono muted">{megabytes(file.size)} MB</span>
                      </li>
                    ))}
                  </ul>
                ) : null}
              </div>

              <div className="field">
                <label className="field-label" htmlFor="build-name">
                  Collection name (optional)
                </label>
                <input
                  id="build-name"
                  className="input"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder={files.length ? `Uploaded PDFs (${files.length})` : 'Uploaded PDFs'}
                  autoComplete="off"
                />
              </div>
            </>
          )}

          {error ? (
            <div id="build-error" role="alert">
              <Banner
                tone={blip ? 'warning' : 'danger'}
                title={
                  jobId === null
                    ? 'Build did not start.'
                    : blip
                      ? 'Still running on the server.'
                      : 'The build stopped.'
                }
              >
                {error}
              </Banner>
            </div>
          ) : null}

          {job ? (
            <div>
              <div className="progress-track" aria-hidden="true">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>
              <div className="job-message" role="status" aria-live="polite">
                {job.message ?? job.status}
              </div>
            </div>
          ) : null}

          <div className="dialog-actions">
            <button type="button" className="button button-secondary" onClick={onClose}>
              {running ? 'Close' : 'Cancel'}
            </button>
            <button type="submit" className="button" disabled={starting || running}>
              {running ? 'Building…' : mode === 'topic' ? 'Start build' : 'Upload and build'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
