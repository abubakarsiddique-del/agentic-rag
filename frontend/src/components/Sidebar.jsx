import React, {useEffect, useMemo, useState} from 'react'
import { Download, Pencil, Plus, Trash2, X } from 'lucide-react'
import { formatConversationTimestamp, groupHistoryByLocalDate } from '../conversationUtils.mjs'
import UploadPanel from './UploadPanel'

export default function Sidebar({
  conversationId,
  documents,
  documentSummary,
  onEnsureConversation,
  onUploaded,
  onDocumentStatus,
  onRemoveDocument,
  history,
  activeId,
  onNewChat,
  onSelectChat,
  onRenameChat,
  onDeleteChat,
  onDownload,
  settings,
  onSettingsChange,
  memoryEnabled,
  conversationMemoryEnabled,
  onGlobalMemoryChange,
  onConversationMemoryChange,
  onClearAllMemory,
  searchMode,
  onSearchModeChange,
  selectedDocumentIds,
  onSelectedDocumentIdsChange,
  open,
  onClose,
}) {
  const [query, setQuery] = useState('')
  const [renamingId, setRenamingId] = useState(null)
  const [renameValue, setRenameValue] = useState('')
  const [confirmDeleteId, setConfirmDeleteId] = useState(null)
  const readyDocuments = documents.filter(document => document.status === 'ready')
  const filteredHistory = useMemo(() => history.filter(item => (item.title || '').toLowerCase().includes(query.trim().toLowerCase())), [history, query])
  const groups = groupHistoryByLocalDate(filteredHistory)

  useEffect(() => {
    if (!open) return undefined
    const closeOnEscape = event => { if (event.key === 'Escape') onClose?.() }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [open, onClose])

  function startRename(conversation) {
    setRenamingId(conversation.id)
    setRenameValue(conversation.title || '')
  }

  async function submitRename(event, id) {
    event.preventDefault()
    const title = renameValue.trim()
    if (title) await onRenameChat(id, title)
    setRenamingId(null)
  }

  const panel = <aside className="flex h-full w-[min(88vw,22rem)] shrink-0 flex-col gap-4 overflow-y-auto border-r border-app-border bg-app-surface p-4 dark:bg-app-surface-2 md:w-80">
    <header className="flex items-center justify-between border-b border-app-border pb-3">
      <h1 className="text-lg font-semibold text-app-text">Agentic RAG</h1>
      <div className="flex items-center gap-1">
        {conversationId && <button type="button" aria-label="Download this chat" title="Download this chat" onClick={onDownload} className="rounded-md border border-app-border-strong p-2 text-app-muted hover:border-app-muted hover:bg-app-surface-2"><Download size={17}/></button>}
        <button type="button" aria-label="New chat" title="New chat" onClick={onNewChat} className="flex items-center gap-1 rounded-md border border-app-border-strong px-2 py-1.5 text-sm font-medium text-app-text hover:border-app-cyan hover:bg-app-cyan-soft"><Plus size={18}/><span>New chat</span></button>
        <button type="button" aria-label="Close sidebar" onClick={onClose} className="rounded-md border border-app-border-strong p-2 text-app-muted hover:bg-app-surface-2 md:hidden"><X size={18}/></button>
      </div>
    </header>

    <section aria-labelledby="documents-heading">
      <div className="mb-3 flex items-center justify-between">
        <h2 id="documents-heading" className="font-mono text-xs font-medium uppercase tracking-wide text-app-muted">Documents ({documents.length})</h2>
      </div>
      <UploadPanel
        conversationId={conversationId}
        documents={documents}
        summary={documentSummary}
        onEnsureConversation={onEnsureConversation}
        onUploaded={onUploaded}
        onDocumentStatus={onDocumentStatus}
        onRemoveDocument={onRemoveDocument}
      />
      <fieldset className="mt-4 space-y-2">
        <legend className="mb-2 text-sm font-medium text-app-text">Search in</legend>
        <label className="flex items-center gap-2 text-sm text-app-muted">
          <input type="radio" className="accent-app-cyan" name="document-scope" checked={searchMode === 'all'} onChange={() => onSearchModeChange('all')} />
          All documents
        </label>
        <label className="flex items-center gap-2 text-sm text-app-muted">
          <input type="radio" className="accent-app-cyan" name="document-scope" checked={searchMode === 'selected'} onChange={() => onSearchModeChange('selected')} disabled={!readyDocuments.length} />
          Selected
        </label>
        {searchMode === 'selected' && readyDocuments.map(document => <label key={document.id} className="ml-5 flex items-center gap-2 text-xs text-app-muted">
          <input type="checkbox" className="accent-app-cyan" checked={selectedDocumentIds.includes(document.id)} onChange={event => onSelectedDocumentIdsChange(event.target.checked ? [...selectedDocumentIds, document.id] : selectedDocumentIds.filter(id => id !== document.id))} />
          <span className="truncate">{document.filename}</span>
        </label>)}
      </fieldset>
    </section>

    <section className="flex min-h-0 flex-1 flex-col border-t border-app-border pt-4" aria-labelledby="history-heading">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 id="history-heading" className="font-mono text-xs font-medium uppercase tracking-wide text-app-muted">History</h2>
        <input aria-label="Search chat history" value={query} onChange={event => setQuery(event.target.value)} placeholder="Search" className="w-28 rounded-md border border-app-border-strong bg-app-surface-2 px-2 py-1 text-xs text-app-text placeholder:text-app-muted" />
      </div>
      {!filteredHistory.length && <p className="text-sm text-slate-500">No chats yet. Add a document to start.</p>}
      <div className="space-y-4 overflow-y-auto">
        {Object.entries(groups).map(([label, items]) => items.length > 0 && <section key={label}>
          <h3 className="mb-1 font-mono text-xs font-medium text-app-muted">{label}</h3>
          <ul className="space-y-1">
            {items.map(conversation => <li key={conversation.id} className={`group rounded-lg border p-2 ${activeId === conversation.id ? 'border-app-cyan/40 bg-app-cyan-soft' : 'border-transparent hover:border-app-border hover:bg-app-surface-2'}`}>
              {renamingId === conversation.id ? <form onSubmit={event => submitRename(event, conversation.id)} className="flex gap-1">
                <input autoFocus aria-label="Chat title" value={renameValue} onChange={event => setRenameValue(event.target.value)} className="min-w-0 flex-1 rounded-md border border-app-border-strong bg-app-surface px-2 py-1 text-sm text-app-text" />
                <button type="submit" className="text-xs font-medium text-app-cyan">Save</button>
                <button type="button" onClick={() => setRenamingId(null)} className="text-xs text-app-muted">Cancel</button>
              </form> : <div className="flex items-start gap-2">
                <button type="button" onClick={() => {onSelectChat(conversation.id); onClose?.()}} aria-current={activeId === conversation.id ? 'page' : undefined} className="min-w-0 flex-1 text-left">
                  <span className="block truncate text-sm font-medium text-app-text">{conversation.title || 'Untitled chat'}</span>
                  <span className="mt-1 block font-mono text-xs text-app-muted">{conversation.document_count} docs · {formatConversationTimestamp(conversation)}</span>
                </button>
                <div className="flex shrink-0 gap-1 opacity-100 md:opacity-0 md:group-hover:opacity-100 md:group-focus-within:opacity-100">
                  <button type="button" aria-label={`Rename ${conversation.title || 'chat'}`} onClick={() => startRename(conversation)} className="rounded-md p-1 text-app-muted hover:bg-app-surface-3 hover:text-app-text"><Pencil size={14}/></button>
                  <button type="button" aria-label={`Delete ${conversation.title || 'chat'}`} onClick={() => setConfirmDeleteId(conversation.id)} className="rounded-md p-1 text-app-muted hover:bg-app-surface-3 hover:text-red-400"><Trash2 size={14}/></button>
                </div>
              </div>}
              {confirmDeleteId === conversation.id && <div className="mt-2 rounded-lg border border-red-500/30 bg-red-950/50 p-2 text-xs text-red-200">
                <p>Delete this chat and its files?</p>
                <div className="mt-2 flex justify-end gap-3">
                  <button type="button" onClick={() => setConfirmDeleteId(null)}>Cancel</button>
                  <button type="button" className="font-semibold" onClick={async () => {await onDeleteChat(conversation.id); setConfirmDeleteId(null)}}>Delete</button>
                </div>
              </div>}
            </li>)}
          </ul>
        </section>)}
      </div>
    </section>

    <details className="border-t border-app-border pt-3">
      <summary className="cursor-pointer text-sm font-semibold">Agent settings</summary>
      <div className="mt-3 space-y-3">
        <fieldset>
          <legend className="mb-1 font-mono text-xs text-app-muted">Answer mode</legend>
          <div className="grid grid-cols-2 gap-2">
            <button type="button" aria-pressed={settings.answer_mode === 'agentic'} onClick={() => onSettingsChange({...settings, answer_mode: 'agentic'})} className={`rounded-md border px-2 py-1.5 text-xs ${settings.answer_mode === 'agentic' ? 'border-app-cyan bg-app-cyan-soft text-app-cyan' : 'border-app-border-strong text-app-muted hover:bg-app-surface-2'}`}>Agentic</button>
            <button type="button" aria-pressed={settings.answer_mode === 'traditional'} onClick={() => onSettingsChange({...settings, answer_mode: 'traditional'})} className={`rounded-md border px-2 py-1.5 text-xs ${settings.answer_mode === 'traditional' ? 'border-app-cyan bg-app-cyan-soft text-app-cyan' : 'border-app-border-strong text-app-muted hover:bg-app-surface-2'}`}>Traditional</button>
          </div>
        </fieldset>
        <label className="block text-xs text-app-muted">Maximum retries
          <select value={settings.max_retries} onChange={event => onSettingsChange({...settings, max_retries: Number(event.target.value)})} className="mt-1 block w-full rounded-md border border-app-border-strong bg-app-surface-2 px-2 py-1.5 text-app-text">{[0,1,2,3].map(value => <option key={value}>{value}</option>)}</select>
        </label>
        <label className="block text-xs text-app-muted">Passages per search
          <input type="number" min="1" max="12" value={settings.passages_per_search} onChange={event => onSettingsChange({...settings, passages_per_search: Math.max(1, Math.min(12, Number(event.target.value) || 1))})} className="mt-1 block w-full rounded-md border border-app-border-strong bg-app-surface-2 px-2 py-1.5 text-app-text" />
        </label>
        <label className="flex items-center gap-2 text-xs text-app-muted">
          <input type="checkbox" className="accent-app-cyan" checked={settings.rerank_enabled} onChange={event => onSettingsChange({...settings, rerank_enabled: event.target.checked})} />
          Re-rank passages
        </label>
        <label className="block text-xs text-app-muted">Top passages after re-ranking
          <input type="number" min="1" max="20" value={settings.rerank_top_n} disabled={!settings.rerank_enabled} onChange={event => onSettingsChange({...settings, rerank_top_n: Math.max(1, Math.min(20, Number(event.target.value) || 1))})} className="mt-1 block w-full rounded-md border border-app-border-strong bg-app-surface-2 px-2 py-1.5 text-app-text disabled:opacity-50" />
        </label>
        <label className="block text-xs text-app-muted">Broad question handling
          <select value={settings.map_reduce_mode} onChange={event => onSettingsChange({...settings, map_reduce_mode: event.target.value})} className="mt-1 block w-full rounded-md border border-app-border-strong bg-app-surface-2 px-2 py-1.5 text-app-text">
            <option value="auto">Auto</option>
            <option value="off">Off</option>
            <option value="force">Force</option>
          </select>
        </label>
        <label className="flex items-center gap-2 text-xs text-app-muted">
          <input type="checkbox" className="accent-app-cyan" checked={memoryEnabled} onChange={event => onGlobalMemoryChange(event.target.checked)} />
          Cross-session memory
        </label>
        <label className="flex items-center gap-2 text-xs text-app-muted">
          <input type="checkbox" className="accent-app-cyan" checked={conversationMemoryEnabled} disabled={!memoryEnabled || !conversationId} onChange={event => onConversationMemoryChange(event.target.checked)} />
          Use memory in this chat
        </label>
        <button type="button" onClick={onClearAllMemory} className="rounded-md border border-app-border-strong px-2 py-1.5 text-left text-xs text-app-muted hover:bg-app-surface-2">
          Clear all memory
        </button>
        <label className="flex items-center gap-2 text-xs text-app-muted"><input type="checkbox" className="accent-app-cyan" checked={settings.show_reasoning_steps} onChange={event => onSettingsChange({...settings, show_reasoning_steps: event.target.checked})} />Show reasoning steps</label>
      </div>
    </details>
  </aside>

  return <>
    <div className={`fixed inset-0 z-40 bg-black/30 transition-opacity md:hidden ${open ? 'opacity-100' : 'pointer-events-none opacity-0'}`} onClick={onClose} aria-hidden="true" />
    <div className={`fixed inset-y-0 left-0 z-50 transform transition-transform md:static md:z-auto md:translate-x-0 ${open ? 'translate-x-0' : '-translate-x-full'}`}>{panel}</div>
  </>
}
