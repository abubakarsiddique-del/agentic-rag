import React, {useMemo, useState} from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeSanitize from 'rehype-sanitize'
import { Copy, ThumbsDown, ThumbsUp } from 'lucide-react'
import SourcesDrawer from './SourcesDrawer'
import {
  buildReasoningSteps,
  deriveStepStates,
  parseCitationReferences,
  normalizeSourceContent,
  formatSourceTitle,
} from '../conversationUtils.mjs'

export default function MessageBubble({message, idx, showReasoningSteps = true, onFeedback = () => {}}) {
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [selectedSource, setSelectedSource] = useState(null)
  const isUser = message.role === 'user'
  const trace = message.trace || []
  const effort = message.status === 'error'
    ? 'Answer could not be completed'
    : message.status === 'stopped'
      ? 'Stopped early'
      : deriveStepStates(trace).effort
  const reasoning = buildReasoningSteps(trace)
  const citationEntries = useMemo(
    () => parseCitationReferences(message.text || '', message.sources || []),
    [message.text, message.sources],
  )
  const sourceGroups = useMemo(() => {
    return (message.sources || []).reduce((groups, source) => {
      const key = formatSourceTitle(source)
      groups[key] = [...(groups[key] || []), source]
      return groups
    }, {})
  }, [message.sources])

  const openSourceDrawer = (source) => {
    setSelectedSource(source || (message.sources || [])[0] || null)
    setDrawerOpen(true)
  }

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.text || '')
    } catch (error) {
      console.warn('Could not copy answer text.', error)
    }
  }

  return (
    <article className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`${isUser ? 'border-app-cyan/30 bg-app-surface-2' : 'border-app-border bg-app-surface'} max-w-[85%] rounded-2xl border p-4 shadow-lg`}>
        <div className="answer-content text-sm">
          <ReactMarkdown rehypePlugins={[rehypeSanitize]}>{message.text}</ReactMarkdown>
        </div>
        {!isUser && <p className="mt-3 font-mono text-xs font-medium text-app-muted" aria-label="Answer effort">{effort}</p>}
        {!isUser && message.status === 'stopped' && <p className="mt-2 text-xs text-amber-300">Answer stopped. The text above is what was available so far.</p>}
        {!isUser && message.status === 'error' && <p className="mt-2 text-xs text-red-300">This answer could not be completed.</p>}

        {!isUser && message.outcome === 'direct' && <p className="mt-2 text-sm text-app-muted">I answered directly without searching your files.</p>}
        {!isUser && message.outcome === 'abstain' && <p className="mt-2 text-sm text-amber-200">I couldn’t find enough support in the loaded files. Try rephrasing your question or checking the files above.</p>}

        {!isUser && showReasoningSteps && reasoning.length > 0 && <details className="mt-3 border-t border-app-border pt-3">
          <summary className="cursor-pointer text-sm font-medium text-app-text">How the agent got here</summary>
          <ol className="mt-2 list-decimal space-y-1 pl-5 text-sm text-app-muted">
            {reasoning.map((step, index) => <li key={`${idx}-reason-${index}`}>{step}</li>)}
          </ol>
        </details>}

        {!isUser && <section className="mt-3 border-t border-app-border pt-3" aria-label="Sources">
          <div className="mb-2 flex items-center justify-between gap-2">
            <h3 className="font-mono text-xs font-medium uppercase tracking-wide text-app-muted">Sources</h3>
            {citationEntries.length > 0 && <button type="button" onClick={() => openSourceDrawer(citationEntries[0].source || (message.sources || [])[0])} className="rounded-full border border-app-cyan/40 bg-app-cyan-soft px-2.5 py-1 text-xs font-medium text-app-cyan hover:border-app-cyan">View full passage</button>}
          </div>
          {Object.keys(sourceGroups).length > 0 ? <div className="mt-2 space-y-3">
            {Object.entries(sourceGroups).map(([filename, sources]) => <section key={filename}>
              <h4 className="text-sm font-medium text-app-text">{filename}</h4>
              <ul className="mt-1 space-y-2">{sources.map((source, index) => <li key={`${filename}-${source.page}-${index}`} title={source.rerank_score == null ? undefined : `Rerank score: ${Number(source.rerank_score).toFixed(3)}`} className="rounded-lg border border-app-border bg-app-surface-2 p-2 text-sm">
                <p className="font-mono text-xs text-app-muted">Page {source.page ?? '?'}</p>
                <p className="mt-1 border-l-2 border-app-cyan/50 pl-2 text-xs text-app-muted">{normalizeSourceContent(source)}</p>
              </li>)}</ul>
            </section>)}
          </div> : <p className="mt-1 text-xs text-slate-500">No document passages were used for this answer.</p>}
        </section>}

        {!isUser && <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-app-border pt-2">
          <button type="button" onClick={handleCopy} className="inline-flex items-center gap-1 rounded-md border border-app-border-strong px-2 py-1 text-xs text-app-muted hover:bg-app-surface-2 hover:text-app-text"><Copy size={12}/>Copy</button>
          <button type="button" onClick={() => onFeedback(message.id, 1, 'Helpful')} className={`inline-flex items-center gap-1 rounded-md border px-2 py-1 text-xs ${message.rating === 1 ? 'border-app-green/50 bg-app-green/10 text-app-green' : 'border-app-border-strong text-app-muted hover:bg-app-surface-2 hover:text-app-text'}`}><ThumbsUp size={12}/>Helpful</button>
          <button type="button" onClick={() => onFeedback(message.id, -1, 'Not helpful')} className={`inline-flex items-center gap-1 rounded-md border px-2 py-1 text-xs ${message.rating === -1 ? 'border-amber-500/50 bg-amber-950/40 text-amber-200' : 'border-app-border-strong text-app-muted hover:bg-app-surface-2 hover:text-app-text'}`}><ThumbsDown size={12}/>Not helpful</button>
        </div>}
      </div>
      {!isUser && <SourcesDrawer open={drawerOpen} sources={message.sources || []} selectedSource={selectedSource} onClose={() => setDrawerOpen(false)} />}
    </article>
  )
}
