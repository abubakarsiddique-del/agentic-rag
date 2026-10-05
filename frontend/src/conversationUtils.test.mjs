import {describe, expect, it, vi} from 'vitest'

import {
  buildReasoningSteps,
  buildSuggestedQuestions,
  createActiveConversationManager,
  deriveStepStates,
  deriveConversationTitle,
  formatConversationTimestamp,
  groupHistoryByLocalDate,
  parseCitationReferences,
  updateDocumentStatus,
} from './conversationUtils.mjs'

const response = (payload, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => payload,
})

describe('active conversation manager', () => {
  it('creates at most once when StrictMode replays first-content creation', async () => {
    const calls = []
    const storage = {getItem: () => null, setItem: vi.fn(), removeItem: vi.fn()}
    const conversation = {id: 'created-id'}
    const fetcher = vi.fn(async (_url, options = {}) => {
      calls.push(options.method ?? 'GET')
      return response(conversation)
    })
    const manager = createActiveConversationManager(fetcher, storage)

    const [first, second] = await Promise.all([
      manager.create(),
      manager.create(),
    ])

    expect(first).toBe('created-id')
    expect(second).toBe('created-id')
    expect(calls).toEqual(['POST'])
    expect(storage.setItem).toHaveBeenCalledWith(
      'agentic-rag.activeConversationId',
      'created-id',
    )
  })

  it('validates a stored ID and stays a draft when it is missing', async () => {
    const storage = {getItem: () => 'stale-id', setItem: vi.fn(), removeItem: vi.fn()}
    const fetcher = vi.fn(async () => response({detail: 'Conversation not found'}, 404))
    const manager = createActiveConversationManager(fetcher, storage)

    await expect(manager.restore()).resolves.toBeNull()
    expect(fetcher).toHaveBeenCalledTimes(1)
    expect(storage.removeItem).toHaveBeenCalledWith('agentic-rag.activeConversationId')
  })

  it('restores a valid ID without creating a server row', async () => {
    const storage = {getItem: () => 'existing-id', setItem: vi.fn(), removeItem: vi.fn()}
    const fetcher = vi.fn(async () => response({id: 'existing-id'}))
    const manager = createActiveConversationManager(fetcher, storage)

    await expect(manager.restore()).resolves.toBe('existing-id')
    expect(fetcher).toHaveBeenCalledTimes(1)
    expect(storage.setItem).not.toHaveBeenCalled()
  })
})

describe('history and document helpers', () => {
  it('builds suggestions from ready documents only', () => {
    expect(buildSuggestedQuestions([
      {filename: 'team_report-2026.pdf', status: 'ready'},
      {filename: 'draft.txt', status: 'processing'},
    ])).toEqual([
      'Summarize the main points in team report 2026.',
      'What are the key takeaways from team report 2026?',
      'List the important dates, risks, or actions in team report 2026.',
      'Who is this document written for?',
      'What decisions or recommendations does it make?',
    ])
  })

  it('adds a comparison suggestion when multiple files are ready', () => {
    const suggestions = buildSuggestedQuestions([
      {filename: 'one.pdf', status: 'ready'},
      {filename: 'two.txt', status: 'ready'},
    ])
    expect(suggestions).toHaveLength(5)
    expect(suggestions.at(-1)).toBe('Compare the main ideas across 2 uploaded files.')
  })

  it('does not suggest questions before any document is ready', () => {
    expect(buildSuggestedQuestions([{filename: 'draft.pdf', status: 'processing'}])).toEqual([])
  })

  it('derives a short title at a word boundary', () => {
    expect(deriveConversationTitle('  Compare constitutions across their amendment systems  ', 24)).toBe('Compare constitutions')
  })

  it('groups conversations by local calendar day', () => {
    const now = new Date(2026, 8, 28, 12)
    const at = days => new Date(2026, 8, 28 - days, 8).toISOString()
    const conversations = [0, 1, 4, 9].map((days, index) => ({id: String(index), updated_at: at(days)}))
    const groups = groupHistoryByLocalDate(conversations, now)
    expect(groups.Today.map(item => item.id)).toEqual(['0'])
    expect(groups.Yesterday.map(item => item.id)).toEqual(['1'])
    expect(groups['Previous 7 days'].map(item => item.id)).toEqual(['2'])
    expect(groups.Older.map(item => item.id)).toEqual(['3'])
  })

  it('updates one document status without merging same-name files', () => {
    const files = [{id: 'one', filename: 'report.pdf', status: 'queued'}, {id: 'two', filename: 'report.pdf', status: 'ready'}]
    expect(updateDocumentStatus(files, {id: 'one', status: 'processing'})[0].status).toBe('processing')
    expect(updateDocumentStatus(files, {id: 'one', status: 'processing'})[1].status).toBe('ready')
  })

  it('replaces a temporary upload row when the server assigns its document ID', () => {
    const files = [{id: 'upload-1', filename: 'report.pdf', status: 'processing'}]
    const result = updateDocumentStatus(files, {id: 'server-id', filename: 'report.pdf', status: 'queued'})
    expect(result).toEqual([{id: 'server-id', filename: 'report.pdf', status: 'queued'}])
  })
})

describe('citation references', () => {
  it('maps numbered page citations to the matching displayed source', () => {
    const sources = [
      {filename: 'first.pdf', page: 1},
      {filename: 'second.pdf', page: 2},
    ]

    expect(parseCitationReferences('Second claim [Source 2: Page 2].', sources)).toEqual([
      {key: '2: Page 2', source: sources[1], label: 'Source 2'},
    ])
  })
})

describe('deriveStepStates', () => {
  it('reports a normal retrieval path', () => {
    const result = deriveStepStates([
      {step: 'contextualize', standalone_question: 'question', skipped: true},
      {step: 'route', route: 'retrieve'},
      {step: 'retrieve', query: 'question', passage_count: 4},
      {step: 'grade', sufficient: true},
      {step: 'generate'},
      {step: 'complete'},
    ])
    expect(result.states).toEqual({contextualize: 'skipped', route: 'done', retrieve: 'done', grade: 'done', rewrite: 'skipped', generate: 'done'})
    expect(result.retryCount).toBe(0)
  })

  it('counts retries and completes rewritten searches', () => {
    const result = deriveStepStates([
      {step: 'contextualize', standalone_question: 'second constitution amendment', skipped: false},
      {step: 'route', route: 'retrieve'},
      {step: 'retrieve', query: 'first'},
      {step: 'grade', sufficient: false},
      {step: 'rewrite', query: 'refined'},
      {step: 'retrieve', query: 'refined'},
      {step: 'grade', sufficient: true},
      {step: 'generate'},
      {step: 'complete'},
    ])
    expect(result.retryCount).toBe(1)
    expect(result.states.contextualize).toBe('done')
    expect(result.counts.retrieve).toBe(2)
    expect(result.counts.grade).toBe(2)
    expect(result.states.rewrite).toBe('done')
    expect(result.effort).toBe('Answered after 1 retry')
    expect(buildReasoningSteps([
      {step: 'retrieve', query: 'first', passage_count: 2},
      {step: 'rewrite', query: 'refined'},
      {step: 'retrieve', query: 'refined', passage_count: 3},
    ])).toContain('Searches: 2; retries: 1.')
  })

  it('skips retrieval for direct answers', () => {
    const result = deriveStepStates([
      {step: 'contextualize', standalone_question: 'hello', skipped: true},
      {step: 'route', route: 'direct'},
      {step: 'direct_answer'},
    ])
    expect(result.states).toEqual({contextualize: 'skipped', route: 'done', retrieve: 'skipped', grade: 'skipped', rewrite: 'skipped', generate: 'skipped'})
    expect(result.outcome).toBe('direct')
    expect(result.effort).toBe('Answered directly')
  })

  it('shows abstention instead of generation', () => {
    const result = deriveStepStates([
      {step: 'contextualize', standalone_question: 'unknown', skipped: true},
      {step: 'route', route: 'retrieve'},
      {step: 'retrieve'},
      {step: 'grade', sufficient: false},
      {step: 'abstain'},
    ])
    expect(result.states.generate).toBe('skipped')
    expect(result.outcome).toBe('abstain')
    expect(result.effort).toBe("Couldn't confirm an answer")
  })

  it('starts with every step pending before a question', () => {
    const result = deriveStepStates([])
    expect(Object.values(result.states)).toEqual(['pending', 'pending', 'pending', 'pending', 'pending', 'pending'])
    expect(result.retryCount).toBe(0)
  })

  it('records rerank as an optional completed step without changing base steps', () => {
    const result = deriveStepStates([
      {step: 'retrieve', passage_count: 20},
      {step: 'rerank', enabled: true, candidate_count: 20, kept_count: 5},
    ])
    expect(result.states.rerank).toBe('done')
    expect(result.states.grade).toBe('active')
  })

  it('tracks map progress and completion as an optional step', () => {
    const active = deriveStepStates([
      {step: 'map_progress', stage: 'map', mapped: 3, total: 8},
    ])
    expect(active.states.map_reduce).toBe('active')
    expect(active.mapProgress).toMatchObject({mapped: 3, total: 8})
    expect(deriveStepStates([{step: 'map_reduce', fallback: false}]).states.map_reduce).toBe('done')
  })
})

it('formats recent timestamps relative to the current day', () => {
  const now = new Date(2026, 8, 28, 12)
  expect(formatConversationTimestamp({updated_at: new Date(2026, 8, 28, 9, 30).toISOString()}, now)).toMatch(/^Today/)
  expect(formatConversationTimestamp({updated_at: new Date(2026, 8, 27, 9, 30).toISOString()}, now)).toMatch(/^Yesterday/)
  expect(formatConversationTimestamp({updated_at: new Date(2026, 8, 20, 9, 30).toISOString()}, now)).toMatch(/Sep|Sept|2026/)
})

it('timestamp formatting never exposes Invalid Date', () => {
  expect(
    formatConversationTimestamp({created_at: '2026-09-28T12:00:00Z'}),
  ).not.toBe('Invalid Date')
  expect(formatConversationTimestamp({})).toBe('Time unavailable')
  expect(formatConversationTimestamp({updated_at: 'not-a-date'})).toBe('Time unavailable')
})