import React from 'react'

export default function SourcesDrawer({open, sources = [], selectedSource = null, onClose}) {
  const items = selectedSource ? [selectedSource] : sources

  return (
    <div className={`fixed right-0 top-0 z-30 h-full w-[28rem] max-w-full border-l border-app-border bg-app-surface shadow-2xl transition-transform duration-200 ${open ? 'translate-x-0' : 'translate-x-full'}`}>
      <div className="flex items-center justify-between border-b border-app-border p-4">
        <div>
          <div className="text-sm font-semibold">Source passage</div>
          <div className="font-mono text-xs text-app-muted">{selectedSource ? (selectedSource.filename || selectedSource.source || 'Selected source') : 'All sources'}</div>
        </div>
        <button type="button" onClick={onClose} aria-label="Close source drawer" className="rounded-md border border-app-border-strong px-2 py-1 text-app-muted hover:bg-app-surface-2 hover:text-app-text">Close</button>
      </div>
      <div className="h-[calc(100%-61px)] overflow-auto p-4">
        {items && items.length ? items.map((source, index) => {
          const fullText = source?.source || source?.full_text || source?.text || source?.preview || source?.snippet || 'No full passage available.'
          const filename = source?.document_name || source?.filename || source?.source || `Source ${index + 1}`
          const page = source?.page ?? '?' 
          return (
            <article key={`${filename}-${page}-${index}`} className="mb-5 rounded-xl border border-app-border bg-app-surface-2 p-3">
              <div className="text-sm font-medium text-app-text">{filename}</div>
              <div className="mt-1 font-mono text-xs text-app-muted">Page {page}</div>
              <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-app-muted">{fullText}</p>
            </article>
          )
        }) : <p className="text-sm text-app-muted">No sources yet.</p>}
      </div>
    </div>
  )
}
