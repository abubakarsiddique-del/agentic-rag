export const DEFAULT_AGENT_SETTINGS = {
  answer_mode: 'agentic',
  max_retries: 2,
  passages_per_search: 4,
  show_reasoning_steps: true,
  rerank_enabled: true,
  rerank_candidates: 20,
  rerank_top_n: 5,
  map_reduce_mode: 'auto',
}

const ACTIVE_CONVERSATION_KEY = 'agentic-rag.activeConversationId'
const AGENT_SETTINGS_KEY = 'agentic-rag.settings'
const STEP_NAMES = ['contextualize', 'route', 'retrieve', 'grade', 'rewrite', 'generate']

export function readAgentSettings(storage = globalThis.localStorage) {
  try {
    const saved = JSON.parse(storage.getItem(AGENT_SETTINGS_KEY) || '{}')
    return {
      answer_mode: saved.answer_mode === 'traditional' ? 'traditional' : 'agentic',
      max_retries: clampInteger(saved.max_retries, 0, 3, DEFAULT_AGENT_SETTINGS.max_retries),
      passages_per_search: clampInteger(saved.passages_per_search, 1, 12, DEFAULT_AGENT_SETTINGS.passages_per_search),
      show_reasoning_steps: saved.show_reasoning_steps !== false,
      rerank_enabled: saved.rerank_enabled !== false,
      rerank_candidates: clampInteger(saved.rerank_candidates, 1, 100, DEFAULT_AGENT_SETTINGS.rerank_candidates),
      rerank_top_n: clampInteger(saved.rerank_top_n, 1, 20, DEFAULT_AGENT_SETTINGS.rerank_top_n),
      map_reduce_mode: ['auto', 'off', 'force'].includes(saved.map_reduce_mode) ? saved.map_reduce_mode : 'auto',
    }
  } catch {
    return {...DEFAULT_AGENT_SETTINGS}
  }
}

export function saveAgentSettings(settings, storage = globalThis.localStorage) {
  storage.setItem(AGENT_SETTINGS_KEY, JSON.stringify(settings))
}

function clampInteger(value, min, max, fallback) {
  const number = Number(value)
  if (!Number.isInteger(number)) return fallback
  return Math.min(max, Math.max(min, number))
}

export function createActiveConversationManager(fetcher = fetch, storage = globalThis.localStorage) {
  let pendingCreation

  async function create() {
    if (!pendingCreation) {
      pendingCreation = createConversation()
    }
    try {
      return await pendingCreation
    } catch (error) {
      pendingCreation = null
      throw error
    } finally {
      pendingCreation = null
    }
  }

  async function createConversation() {
    const response = await fetcher('/api/conversations', {method: 'POST'})
    if (!response.ok) throw new Error('Could not create a new workspace')
    const conversation = await response.json()
    storage.setItem(ACTIVE_CONVERSATION_KEY, conversation.id)
    return conversation.id
  }

  async function restore() {
    const savedId = storage.getItem(ACTIVE_CONVERSATION_KEY)
    if (!savedId) return null
    const response = await fetcher(`/api/conversations/${savedId}`)
    if (response.ok) return savedId
    storage.removeItem(ACTIVE_CONVERSATION_KEY)
    return null
  }

  function remember(id) {
    if (id) storage.setItem(ACTIVE_CONVERSATION_KEY, id)
    else storage.removeItem(ACTIVE_CONVERSATION_KEY)
  }

  return {restore, create, remember}
}

export function deriveConversationTitle(question, maximum = 50) {
  const normalized = String(question || '').trim().replace(/\s+/g, ' ')
  if (normalized.length <= maximum) return normalized || 'New conversation'
  const prefix = normalized.slice(0, maximum + 1)
  const boundary = prefix.lastIndexOf(' ')
  return (boundary > 0 ? prefix.slice(0, boundary) : prefix.slice(0, maximum)).trim()
}

export function groupHistoryByLocalDate(conversations, now = new Date()) {
  const labels = ['Today', 'Yesterday', 'Previous 7 days', 'Older']
  const groups = Object.fromEntries(labels.map(label => [label, []]))
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  for (const conversation of [...conversations].sort((a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at))) {
    const updated = new Date(conversation.updated_at)
    if (Number.isNaN(updated.getTime())) {
      groups.Older.push(conversation)
      continue
    }
    const day = new Date(updated.getFullYear(), updated.getMonth(), updated.getDate()).getTime()
    const difference = Math.floor((today - day) / 86_400_000)
    const group = difference <= 0 ? 'Today' : difference === 1 ? 'Yesterday' : difference < 7 ? 'Previous 7 days' : 'Older'
    groups[group].push(conversation)
  }
  return groups
}

export function updateDocumentStatus(documents, update) {
  const index = documents.findIndex(document => document.id === update.id)
  if (index < 0) {
    const withoutTemporary = documents.filter(document => !(String(document.id).startsWith('upload-') && document.filename === update.filename && !String(update.id).startsWith('upload-')))
    return [...withoutTemporary, {...update}]
  }
  return documents
    .filter((document, current) => !(current !== index && document.id.startsWith('upload-') && document.filename === update.filename && !String(update.id).startsWith('upload-')))
    .map(document => document.id === update.id ? {...document, ...update} : document)
}

export function buildSuggestedQuestions(documents = []) {
  const readyDocuments = documents.filter(document => document.status === 'ready')
  if (!readyDocuments.length) return []

  const names = readyDocuments.map(document =>
    String(document.filename || 'document')
      .replace(/\.[^.]+$/, '')
      .replace(/[_-]+/g, ' ')
      .trim() || 'document',
  )
  const primary = names[0]
  const suggestions = [
    `Summarize the main points in ${primary}.`,
    `What are the key takeaways from ${primary}?`,
    `List the important dates, risks, or actions in ${primary}.`,
    'Who is this document written for?',
    'What decisions or recommendations does it make?',
  ]

  if (names.length > 1) {
    return suggestions.slice(0, 4).concat(
      `Compare the main ideas across ${names.length} uploaded files.`,
    )
  }
  return suggestions
}

export function deriveStepStates(events = []) {
  const states = Object.fromEntries(STEP_NAMES.map(step => [step, 'pending']))
  const counts = {retrieve: 0, grade: 0, rewrite: 0}
  let outcome = null
  let finalPath = null
  let mapProgress = null

  for (const event of events) {
    if (!event || typeof event !== 'object') continue
    switch (event.step) {
      case 'contextualize':
        states.contextualize = event.skipped ? 'skipped' : 'done'
        break
      case 'route':
        if (states.contextualize === 'pending') states.contextualize = 'skipped'
        states.route = 'done'
        if (event.route === 'direct') {
          states.retrieve = 'skipped'
          states.grade = 'skipped'
          states.rewrite = 'skipped'
        } else if (states.retrieve === 'pending') {
          states.retrieve = 'active'
        }
        break
      case 'retrieve':
        counts.retrieve += 1
        states.route = 'done'
        states.retrieve = 'done'
        states.grade = 'active'
        break
      case 'grade':
        counts.grade += 1
        states.retrieve = 'done'
        states.grade = 'done'
        if (event.sufficient === false) states.rewrite = 'active'
        else states.generate = 'active'
        break
      case 'rewrite':
        counts.rewrite += 1
        states.grade = 'done'
        states.rewrite = 'done'
        states.retrieve = 'active'
        break
      case 'rerank':
        states.rerank = event.enabled === false ? 'skipped' : 'done'
        break
      case 'map_progress':
        states.map_reduce = 'active'
        mapProgress = event
        break
      case 'map_reduce':
        states.map_reduce = event.fallback ? 'skipped' : event.cancelled ? 'skipped' : 'done'
        break
      case 'generate':
        states.route = 'done'
        states.retrieve = counts.retrieve ? 'done' : 'skipped'
        states.grade = counts.grade ? 'done' : 'skipped'
        states.rewrite = counts.rewrite ? 'done' : 'skipped'
        states.generate = 'active'
        finalPath = 'generate'
        break
      case 'direct_answer':
        if (states.contextualize === 'pending') states.contextualize = 'skipped'
        states.route = 'done'
        states.retrieve = 'skipped'
        states.grade = 'skipped'
        states.rewrite = 'skipped'
        states.generate = 'skipped'
        outcome = 'direct'
        finalPath = 'direct_answer'
        break
      case 'abstain':
        if (states.contextualize === 'pending') states.contextualize = 'skipped'
        states.route = 'done'
        states.retrieve = counts.retrieve ? 'done' : 'skipped'
        states.grade = counts.grade ? 'done' : 'skipped'
        states.rewrite = counts.rewrite ? 'done' : 'skipped'
        states.generate = 'skipped'
        outcome = 'abstain'
        finalPath = 'abstain'
        break
      case 'complete':
      case 'answer':
        for (const step of STEP_NAMES) {
          if (states[step] === 'active') states[step] = 'done'
        }
        break
      default:
        break
    }
  }

  const retryCount = Math.max(0, counts.retrieve - 1)
  const effort = outcome === 'direct'
    ? 'Answered directly'
    : outcome === 'abstain'
      ? "Couldn't confirm an answer"
      : retryCount > 0
        ? `Answered after ${retryCount} ${retryCount === 1 ? 'retry' : 'retries'}`
        : events.length > 0
          ? 'Answered from your documents'
          : 'Ready for your question'

  return {states, counts, retryCount, outcome, finalPath, effort, mapProgress}
}

export function buildReasoningSteps(events = []) {
  const searches = []
  const steps = []
  for (const event of events) {
    if (!event || typeof event !== 'object') continue
    if (event.step === 'contextualize') {
      steps.push(event.skipped
        ? 'There was no earlier chat to refer to.'
        : `I understood your follow-up as: “${event.standalone_question}”.`)
    } else if (event.step === 'route') {
      steps.push(event.route === 'direct'
        ? 'This question could be answered directly without searching your files.'
        : 'This question needs information from your files, so I searched them.')
    } else if (event.step === 'retrieve') {
      const query = event.query || 'the original question'
      searches.push(query)
      steps.push(`Search ${searches.length}: “${query}” (${event.passage_count ?? 0} passages found).`)
    } else if (event.step === 'grade') {
      steps.push(event.sufficient
        ? 'The passages were enough to answer the question.'
        : 'The passages were not enough, so I tried another search.')
    } else if (event.step === 'rewrite') {
      steps.push(`I refined the search to: “${event.query || event.detail || ''}”.`)
    } else if (event.step === 'generate') {
      steps.push('I wrote the answer using the selected passages.')
    } else if (event.step === 'direct_answer') {
      steps.push('I answered directly without using document passages.')
    } else if (event.step === 'abstain') {
      steps.push('I could not find enough support in the files, so I did not guess.')
    }
  }
  const retries = Math.max(0, searches.length - 1)
  if (searches.length) steps.push(`Searches: ${searches.length}; retries: ${retries}.`)
  return steps
}

export function parseCitationReferences(answer = '', sources = []) {
  const pattern = /\[(?:source|passage|doc)\s*(?:[:#-]\s*)?([^\]]+)\]/gi
  const markers = [...(answer || '').matchAll(pattern)].map(match => String(match[1] || '').trim()).filter(Boolean)
  if (!markers.length) {
    return (sources || []).map((source, index) => ({
      key: String(index + 1),
      source,
      label: `Source ${index + 1}`,
    }))
  }

  const idMap = new Map()
  for (const source of sources || []) {
    const candidates = [
      source.id,
      source.passage_id,
      source.document_id,
      source.source,
      source.filename,
      source.document_name,
      source.title,
      `${source.filename || source.document_name || 'document'}:${source.page ?? ''}`,
    ].filter(Boolean).map(String)
    for (const candidate of candidates) {
      idMap.set(candidate, source)
    }
  }

  return [...new Set(markers)].map((marker, index) => {
    const sourceNumber = marker.match(/^\s*(\d+)(?:\s*[:#,-]|\s+page\b)/i)?.[1]
    const source = idMap.get(marker)
      || (sourceNumber ? (sources || [])[Number(sourceNumber) - 1] : null)
      || (sources || [])[index]
      || null
    return {
      key: marker,
      source,
      label: `Source ${sourceNumber || index + 1}`,
    }
  })
}

export function normalizeSourceContent(source = {}) {
  return source.full_text || source.text || source.content || source.snippet || source.preview || 'No excerpt available.'
}

export function formatSourceTitle(source = {}) {
  return source.document_name || source.filename || source.source || source.title || 'Source'
}

export function formatConversationTimestamp(conversation, now = new Date()) {
  const value = conversation.updated_at ?? conversation.created_at
  if (!value) return 'Time unavailable'

  const timestamp = new Date(value)
  if (Number.isNaN(timestamp.getTime())) return 'Time unavailable'

  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const dayStart = new Date(timestamp.getFullYear(), timestamp.getMonth(), timestamp.getDate()).getTime()
  const differenceInDays = Math.floor((todayStart - dayStart) / 86_400_000)

  const timeLabel = new Intl.DateTimeFormat(undefined, {
    hour: 'numeric',
    minute: '2-digit',
  }).format(timestamp)

  if (differenceInDays <= 0) return `Today • ${timeLabel}`
  if (differenceInDays === 1) return `Yesterday • ${timeLabel}`

  const dateLabel = new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
  }).format(timestamp)
  return `${dateLabel} • ${timeLabel}`
}
