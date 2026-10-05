import React, {useRef, useState} from 'react'
import { FileText, UploadCloud, X } from 'lucide-react'
import {apiUpload} from '../api'

export default function UploadPanel({conversationId, documents, summary, onEnsureConversation, onDocumentStatus, onRemoveDocument, onUploaded}) {
  const [drag, setDrag] = useState(false)
  const [error, setError] = useState('')
  const [confirmRemoveId, setConfirmRemoveId] = useState(null)
  const inputRef = useRef(null)

  async function upload(files) {
    const selected = Array.from(files).filter(file => file.name)
    if (!selected.length) return
    setError('')
    let id = conversationId
    try {
      if (!id) id = await onEnsureConversation()
    } catch (failure) {
      setError(failure.message || 'Could not start a new chat.')
      return
    }

    const localItems = selected.map((file, index) => ({
      id: `upload-${Date.now()}-${index}`,
      filename: file.name,
      status: 'uploading',
      progress: 0,
      error_message: '',
    }))
    localItems.forEach(item => onDocumentStatus(item))

    const form = new FormData()
    selected.forEach(file => form.append('files', file, file.name))
    let response
    try {
      response = await apiUpload(`/api/conversations/${id}/documents`, form, progress => {
        const status = progress >= 100 ? 'processing' : 'uploading'
        localItems.forEach(item => onDocumentStatus({...item, status, progress}))
      })
    } catch (failure) {
      const reason = failure.message || 'The upload was interrupted. Check your connection and try again.'
      localItems.forEach(item => onDocumentStatus({...item, status: 'failed', error_message: reason}))
      setError(reason)
      if (inputRef.current) inputRef.current.value = ''
      return
    }

    let payload = {}
    try { payload = await response.json() } catch { /* show a plain fallback below */ }
    if (response.ok) {
      const acceptedByFilename = new Map()
      const rejectedByFilename = new Map()
      for (const item of payload.accepted || []) acceptedByFilename.set(item.filename, [...(acceptedByFilename.get(item.filename) || []), item])
      for (const item of payload.rejected || []) rejectedByFilename.set(item.filename, [...(rejectedByFilename.get(item.filename) || []), item])
      selected.forEach((file, index) => {
        const accepted = acceptedByFilename.get(file.name)?.shift()
        const rejected = rejectedByFilename.get(file.name)?.shift()
        if (accepted) onDocumentStatus({...accepted, progress: 100})
        else if (rejected) onDocumentStatus({
          id: `rejected-${Date.now()}-${index}`,
          filename: file.name,
          status: 'failed',
          error_code: rejected.error_code,
          error_message: rejected.reason,
        })
      })
      setError('')
      onUploaded?.()
    } else {
      const reason = typeof payload.detail?.message === 'string'
        ? payload.detail.message
        : typeof payload.detail === 'string'
          ? payload.detail
          : 'We could not add these files. Check your connection and try again.'
      localItems.forEach(item => onDocumentStatus({...item, status: 'failed', error_message: reason}))
      setError(reason)
    }
    if (inputRef.current) inputRef.current.value = ''
  }

  const statusText = {
    queued: 'Waiting to process',
    processing: 'Processing…',
    ready: 'Ready',
    failed: 'Could not process',
    uploading: 'Uploading',
  }
  const readyDocuments = documents.filter(document => document.status === 'ready')

  return (
    <div>
      <button type="button" onClick={() => inputRef.current?.click()} className="mb-2 text-xs font-medium text-app-cyan hover:underline">Add files</button>
      <button type="button" disabled={false} onClick={() => inputRef.current?.click()} onDragOver={event => {event.preventDefault(); setDrag(true)}} onDragLeave={() => setDrag(false)} onDrop={event => {event.preventDefault(); setDrag(false); upload(event.dataTransfer.files)}} className={`w-full rounded-xl border-2 border-dashed p-4 text-left transition-colors ${drag ? 'border-app-cyan bg-app-cyan-soft' : 'border-app-border-strong hover:border-app-cyan hover:bg-app-surface-2'}`}>
        <span className="flex items-center gap-3">
          <UploadCloud size={20} className="shrink-0 text-app-cyan" />
          <span>
            <span className="block text-sm font-medium text-app-text">Drop files here or browse</span>
            <span className="mt-1 block text-xs text-app-muted">PDF or plain text · multiple files</span>
          </span>
        </span>
      </button>
      <input ref={inputRef} type="file" accept=".pdf,.txt,application/pdf,text/plain" multiple hidden aria-label="Add PDF or text files" onChange={event => upload(event.target.files)} />

      {documents.length > 0 && <ul className="mt-3 space-y-3" aria-live="polite">
        {documents.map(document => <li key={document.id} className="flex items-start gap-2 text-sm">
          <FileText size={16} className="mt-0.5 shrink-0 text-app-muted" />
          <span className="min-w-0 flex-1">
            <span className="block truncate text-app-text">{document.filename}</span>
            <span className={`font-mono text-xs ${document.status === 'failed' ? 'text-red-400' : document.status === 'ready' ? 'text-app-green' : 'text-app-muted'}`}>
              {statusText[document.status] || 'Waiting'}{document.status === 'uploading' ? ` · ${document.progress || 0}%` : ''}
              {document.status === 'ready' ? ` · ${document.pages} pages, ${document.chunks} sections` : ''}
            </span>
            {document.error_message && <span className="block text-xs text-red-300">{document.error_message}</span>}
          </span>
          {document.id && !document.id.startsWith('upload-') && !document.id.startsWith('rejected-') && (confirmRemoveId === document.id
            ? <span className="flex items-center gap-2 text-xs"><button type="button" onClick={() => {onRemoveDocument(document); setConfirmRemoveId(null)}} className="font-medium text-red-400">Remove</button><button type="button" onClick={() => setConfirmRemoveId(null)} className="text-app-muted">Cancel</button></span>
            : <button type="button" aria-label={`Remove ${document.filename}`} onClick={() => setConfirmRemoveId(document.id)} className="rounded-md p-1 text-app-muted hover:bg-app-surface-3 hover:text-red-400"><X size={16}/></button>)}
        </li>)}
      </ul>}

      {readyDocuments.length > 0 && summary && <p className="mt-3 border-t border-app-border pt-3 font-mono text-xs text-app-muted">
        {summary.pages} pages read · {summary.chunks} sections searchable
      </p>}
      {!readyDocuments.length && <p className="mt-3 text-xs text-app-muted">Add a ready file before asking a question.</p>}
      {error && <p role="alert" className="mt-3 rounded bg-red-950 p-2 text-sm text-red-200">{error}</p>}
    </div>
  )
}
