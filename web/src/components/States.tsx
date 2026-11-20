import type { ReactNode } from 'react'

export function LoadingState({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="skeleton" role="status" aria-live="polite">
      <span className="rail-label">{label}</span>
      <div className="skeleton-line wide" />
      <div className="skeleton-line" />
      <div className="skeleton-line narrow" />
    </div>
  )
}

export function EmptyState({
  title,
  children,
  action,
}: {
  title: string
  children?: ReactNode
  action?: ReactNode
}) {
  return (
    <div className="empty">
      <h2>{title}</h2>
      {children ? <p>{children}</p> : null}
      {action}
    </div>
  )
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="empty" role="alert">
      <h2>Could not load this view</h2>
      <p>{message}</p>
      {onRetry ? (
        <button type="button" className="button button-secondary" onClick={onRetry}>
          Try again
        </button>
      ) : null}
    </div>
  )
}

export function Banner({
  tone = 'info',
  title,
  children,
}: {
  tone?: 'info' | 'warning' | 'danger'
  title?: string
  children: ReactNode
}) {
  const className = tone === 'info' ? 'banner' : `banner ${tone}`
  return (
    <div className={className}>
      <div className="banner-body">
        {title ? <strong>{title}</strong> : null}
        {title ? ' ' : null}
        {children}
      </div>
    </div>
  )
}
