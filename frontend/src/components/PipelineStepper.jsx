import React from 'react'
import { deriveStepStates } from '../conversationUtils.mjs'

const steps = [
  ['contextualize', 'Understand'],
  ['route', 'Choose path'],
  ['retrieve', 'Search files'],
  ['grade', 'Check passages'],
  ['rewrite', 'Refine search'],
  ['generate', 'Write answer'],
]

export default function PipelineStepper({events = [], collapsed, onToggle}) {
  const result = deriveStepStates(events)
  const showRerank = events.some(event => event.step === 'rerank' && event.enabled !== false)
  const showMapReduce = events.some(event => event.step === 'map_progress' || event.step === 'map_reduce')
  const optionalSteps = [
    ...(showRerank ? [['rerank', 'Rerank']] : []),
    ...(showMapReduce ? [['map_reduce', 'Map → Reduce']] : []),
  ]
  const visibleSteps = [...steps.slice(0, 3), ...optionalSteps, ...steps.slice(3)]
  const colors = {
    pending: 'border-app-border text-app-muted',
    active: 'border-app-cyan bg-app-cyan-soft text-app-text',
    done: 'border-app-green/40 bg-app-green/10 text-app-green',
    skipped: 'border-app-border text-app-muted',
  }
  const summary = result.effort && result.effort !== 'Ready for your question' ? result.effort : null

  return (
    <section className="border-b border-app-border bg-app-surface px-4 py-3" aria-label="Answer progress">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold">Answer progress</h2>
        <button type="button" onClick={onToggle} className="rounded-md px-2 py-1 text-xs text-app-muted hover:bg-app-surface-2 hover:text-app-text">{collapsed ? 'Show' : 'Hide'}</button>
      </div>
      {collapsed ? (
        summary ? (
          <button type="button" onClick={onToggle} className="mt-2 flex w-full items-center justify-between rounded-xl border border-app-border bg-app-surface-2 px-3 py-2 text-left text-sm hover:border-app-border-strong">
            <span className="font-medium text-app-text">{summary}</span>
            <span className="font-mono text-xs text-app-muted">Expand</span>
          </button>
        ) : null
      ) : (
        <>
          <ol className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-5">
            {visibleSteps.map(([key, label], index) => (
              <li key={key} className={`flex min-w-0 items-center gap-2 rounded-lg border bg-app-surface-2 px-2 py-2 ${colors[result.states[key]]}`} aria-current={result.states[key] === 'active' ? 'step' : undefined}>
                <span className="flex size-6 shrink-0 items-center justify-center rounded-full border border-current font-mono text-xs">{result.states[key] === 'done' ? '✓' : result.states[key] === 'skipped' ? '–' : index + 1}</span>
                <span className="min-w-0 truncate text-xs font-medium">{label}</span>
                {key === 'retrieve' && result.counts.retrieve > 1 && <span className="ml-auto font-mono text-xs">{result.counts.retrieve}x</span>}
                {key === 'grade' && result.counts.grade > 1 && <span className="ml-auto font-mono text-xs">{result.counts.grade}x</span>}
                {key === 'map_reduce' && result.mapProgress?.stage === 'map' && <span className="ml-auto font-mono text-xs">{result.mapProgress.mapped}/{result.mapProgress.total}</span>}
              </li>
            ))}
          </ol>
          {result.outcome === 'direct' && <p role="status" className="mt-3 text-sm text-app-green">Answered directly. This question did not need a file search.</p>}
          {result.outcome === 'abstain' && <p role="status" className="mt-3 text-sm text-amber-200">Couldn’t confirm an answer. Try rephrasing or check which files are loaded.</p>}
        </>
      )}
    </section>
  )
}
