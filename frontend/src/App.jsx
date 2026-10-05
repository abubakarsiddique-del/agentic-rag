import React, {useEffect, useRef, useState} from 'react'
import { LogOut, Menu } from 'lucide-react'
import Sidebar from './components/Sidebar'
import MessageBubble from './components/MessageBubble'
import PipelineStepper from './components/PipelineStepper'
import InputBar from './components/InputBar'
import ThemeToggle from './components/ThemeToggle'
import AuthPanel from './components/AuthPanel'
import Landing from './components/Landing'
import {apiFetch, notifySessionExpired} from './api'
import {
  buildSuggestedQuestions,
  createActiveConversationManager,
  deriveConversationTitle,
  DEFAULT_AGENT_SETTINGS,
  readAgentSettings,
  saveAgentSettings,
  updateDocumentStatus,
} from './conversationUtils.mjs'

function parseSseFrame(frame) {
  let type = 'message'
  const data = []
  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith('event:')) type = line.slice(6).trim()
    else if (line.startsWith('data:')) data.push(line.slice(5).trimStart())
  }
  if (!data.length) return null
  const raw = data.join('\n')
  if (raw === '[DONE]') return {type: 'done'}
  try { return {type, data: JSON.parse(raw)} } catch { return {type, data: raw} }
}

function summarizeDocuments(documents) {
  const ready = documents.filter(document => document.status === 'ready')
  if (!ready.length) return null
  return {
    pages: ready.reduce((sum, document) => sum + (document.pages || 0), 0),
    chunks: ready.reduce((sum, document) => sum + (document.chunks || 0), 0),
  }
}

export default function App() {
  const [currentUser, setCurrentUser] = useState(null)
  const [authLoading, setAuthLoading] = useState(true)
  const [authMode, setAuthMode] = useState('signin')
  const [currentPath, setCurrentPath] = useState(() => window.location.pathname)
  const [activeConv, setActiveConv] = useState(null)
  const [history, setHistory] = useState([])
  const [documents, setDocuments] = useState([])
  const [documentSummary, setDocumentSummary] = useState(null)
  const [messages, setMessages] = useState([])
  const [traceEvents, setTraceEvents] = useState([])
  const [searchMode, setSearchMode] = useState('all')
  const [selectedDocumentIds, setSelectedDocumentIds] = useState([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [voiceOutput, setVoiceOutput] = useState(false)
  const [speechStatus, setSpeechStatus] = useState('')
  const [collapsed, setCollapsed] = useState(true)
  const [dark, setDark] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [settings, setSettings] = useState(() => readAgentSettings())
  const [memoryEnabled, setMemoryEnabled] = useState(false)
  const [conversationMemoryEnabled, setConversationMemoryEnabled] = useState(true)
  const [memoryHits, setMemoryHits] = useState([])
  const [pageError, setPageError] = useState('')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [passwordStatus, setPasswordStatus] = useState(null)
  const [passwordInput, setPasswordInput] = useState('')
  const [passwordStatusMessage, setPasswordStatusMessage] = useState('')
  const [passwordStatusError, setPasswordStatusError] = useState('')
  const [passwordSaving, setPasswordSaving] = useState(false)
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false)
  const managerRef = useRef(null)
  const controllerRef = useRef(null)
  const audioContextRef = useRef(null)
  const audioDecodeQueueRef = useRef(Promise.resolve())
  const audioPendingChunksRef = useRef(0)
  const audioNextStartRef = useRef(0)
  const audioGenerationDoneRef = useRef(false)
  const audioFailureRef = useRef(false)
  const voiceOutputRef = useRef(false)
  const activeConvRef = useRef(null)
  const announcedReadyDocumentIdsRef = useRef(new Set())
  const uploadNoticeIndexRef = useRef(0)

  if (!managerRef.current) managerRef.current = createActiveConversationManager(apiFetch)

  function navigateTo(path, {replace = false} = {}) {
    const method = replace ? 'replaceState' : 'pushState'
    window.history[method]({}, '', path)
    setCurrentPath(window.location.pathname)
    window.scrollTo(0, 0)
  }

  async function refreshHistory() {
    const response = await apiFetch('/api/conversations')
    if (!response.ok) throw new Error('Could not load chat history.')
    const rows = await response.json()
    setHistory(rows)
    return rows
  }

  function setActiveId(id) {
    activeConvRef.current = id
    setActiveConv(id)
    managerRef.current.remember(id)
  }

  async function openConversation(id) {
    setPageError('')
    const response = await apiFetch(`/api/conversations/${id}`)
    if (!response.ok) throw new Error('This chat is no longer available.')
    const detail = await response.json()
    setActiveId(detail.id)
    const loadedDocuments = detail.documents || []
    announcedReadyDocumentIdsRef.current = new Set(
      loadedDocuments.filter(document => document.status === 'ready').map(document => document.id),
    )
    setDocuments(loadedDocuments)
    setDocumentSummary(summarizeDocuments(loadedDocuments))
    setSelectedDocumentIds(loadedDocuments.filter(document => document.status === 'ready').map(document => document.id))
    const loadedMessages = (detail.messages || []).map(message => ({
      id: message.id,
      role: message.role,
      text: message.content,
      sources: message.sources || [],
      trace: message.trace || [],
      rating: message.rating ?? null,
      feedback: message.feedback ?? '',
      status: message.status || 'complete',
      outcome: message.trace?.some(item => item.step === 'abstain') ? 'abstain' : message.trace?.some(item => item.step === 'direct_answer') ? 'direct' : null,
    }))
    setMessages(loadedMessages)
    const latestCompletedTrace = [...loadedMessages].reverse().find(
      message => message.role === 'assistant' && message.status === 'complete' && message.trace.length > 0,
    )?.trace || []
    setTraceEvents(latestCompletedTrace)
    setCollapsed(true)
    setMemoryHits([])
    try {
      const memoryResponse = await apiFetch(`/api/conversations/${id}/memory`)
      if (memoryResponse.ok) {
        const memorySettings = await memoryResponse.json()
        setConversationMemoryEnabled(memorySettings.enabled)
      }
    } catch { /* memory settings are optional when opening old conversations */ }
  }

  async function loadWorkspace() {
    await refreshHistory()
    const memoryResponse = await apiFetch('/api/memory/settings')
    if (memoryResponse.ok) setMemoryEnabled((await memoryResponse.json()).enabled)
    const savedId = await managerRef.current.restore()
    if (savedId) await openConversation(savedId)
  }

  function clearWorkspace() {
    controllerRef.current?.abort()
    controllerRef.current = null
    activeConvRef.current = null
    managerRef.current?.remember(null)
    setActiveConv(null)
    setHistory([])
    setDocuments([])
    announcedReadyDocumentIdsRef.current = new Set()
    setDocumentSummary(null)
    setMessages([])
    setTraceEvents([])
    setSelectedDocumentIds([])
    setMemoryHits([])
    setIsStreaming(false)
  }

  useEffect(() => {
    function handlePopState() {
      setCurrentPath(window.location.pathname)
    }
    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  useEffect(() => {
    function handleUnauthorized() {
      clearWorkspace()
      setCurrentUser(null)
      setAuthMode('signin')
      setPageError('Your session has expired. Sign in to continue.')
      navigateTo('/login', {replace: true})
    }
    window.addEventListener('rag:unauthorized', handleUnauthorized)
    return () => window.removeEventListener('rag:unauthorized', handleUnauthorized)
  }, [])

  useEffect(() => {
    let mounted = true
    document.title = 'Agentic RAG'
    async function restore() {
      try {
        const response = await apiFetch('/api/auth/me')
        if (!response.ok) return
        const user = await response.json()
        if (!mounted) return
        setCurrentUser(user)
        await loadWorkspace()
      } catch (error) {
        if (mounted) setPageError(error.message || 'Could not open chat history.')
      } finally {
        if (mounted) setAuthLoading(false)
      }
    }
    restore()
    return () => { mounted = false }
  }, [])

  useEffect(() => {
    if (!authLoading && currentPath === '/app' && !currentUser) {
      navigateTo('/login', {replace: true})
    }
  }, [authLoading, currentPath, currentUser])

  async function onAuthenticated(user) {
    clearWorkspace()
    setCurrentUser(user)
    navigateTo('/app')
    setPageError('')
    setAuthLoading(true)
    try {
      await loadWorkspace()
    } catch (error) {
      setPageError(error.message || 'Could not load your workspace.')
    } finally {
      setAuthLoading(false)
    }
  }

  async function logout() {
    const activeId = activeConvRef.current
    if (activeId) {
      try { await apiFetch(`/api/conversations/${activeId}/cancel`, {method: 'POST'}) } catch { /* logout still revokes the session */ }
    }
    try { await apiFetch('/api/auth/logout', {method: 'POST'}) } finally {
      clearWorkspace()
      setCurrentUser(null)
      setAuthMode('signin')
      setPageError('')
      navigateTo('/')
    }
  }

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
  }, [dark])

  useEffect(() => {
    try { saveAgentSettings(settings) } catch { /* storage may be disabled */ }
  }, [settings])

  useEffect(() => {
    setDocumentSummary(summarizeDocuments(documents))
  }, [documents])

  useEffect(() => {
    function handleKeydown(event) {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        setCommandPaletteOpen(value => !value)
      }
      if (event.key === 'Escape') {
        setCommandPaletteOpen(false)
        setSettingsOpen(false)
      }
    }
    window.addEventListener('keydown', handleKeydown)
    return () => window.removeEventListener('keydown', handleKeydown)
  }, [])

  const hasPendingDocuments = documents.some(document => ['queued', 'processing'].includes(document.status))
  function announceReadyDocuments(latest) {
    const newlyReady = latest.filter(document =>
      document.status === 'ready' && !announcedReadyDocumentIdsRef.current.has(document.id),
    )
    if (!newlyReady.length) return
    newlyReady.forEach(document => announcedReadyDocumentIdsRef.current.add(document.id))
    const notices = [
      'Your document is ready. Ask a question about it whenever you’re ready.',
      'Upload complete. You can now ask about the document’s details or main points.',
      'The file is ready to search. What would you like to find in it?',
    ]
    const readyNotices = newlyReady.map(document => {
      const variant = uploadNoticeIndexRef.current % notices.length
      uploadNoticeIndexRef.current += 1
      return {
        role: 'assistant',
        text: `${notices[variant]}${newlyReady.length > 1 ? ` (${document.filename})` : ''}`,
        outcome: 'chitchat',
        sources: [],
        trace: [{
          step: 'route',
          route: 'chitchat',
          chitchat_category: 'document_just_uploaded',
          confidence: 1,
          retrieval_skipped: true,
          detail: 'Notified once when the uploaded document became ready.',
        }],
      }
    })
    setMessages(current => [...current, ...readyNotices])
  }

  useEffect(() => {
    if (!activeConv || !hasPendingDocuments) return undefined
    let mounted = true
    const poll = async () => {
      try {
        const response = await apiFetch(`/api/conversations/${activeConv}/documents`)
        if (!response.ok) return
        const latest = await response.json()
        if (!mounted) return
        announceReadyDocuments(latest)
        setDocuments(latest)
        setDocumentSummary(summarizeDocuments(latest))
        setSelectedDocumentIds(current => {
          const readyIds = latest.filter(document => document.status === 'ready').map(document => document.id)
          return current.filter(id => readyIds.includes(id)).concat(readyIds.filter(id => !current.includes(id)))
        })
      } catch (error) {
        if (mounted) console.error('Could not refresh document status', error)
      }
    }
    const timer = setInterval(poll, 1000)
    poll()
    return () => {mounted = false; clearInterval(timer)}
  }, [activeConv, hasPendingDocuments])

  function updateDocument(document) {
    setDocuments(current => updateDocumentStatus(current, document))
  }

  async function ensureConversation() {
    if (activeConvRef.current) return activeConvRef.current
    const id = await managerRef.current.create()
    setActiveId(id)
    setDocuments([])
    setMessages([])
    await refreshHistory()
    return id
  }

  async function onUploadComplete() {
    if (!activeConvRef.current) return
    const [documentResponse] = await Promise.all([
      apiFetch(`/api/conversations/${activeConvRef.current}/documents`),
      refreshHistory(),
    ])
    if (documentResponse.ok) {
      const latest = await documentResponse.json()
      announceReadyDocuments(latest)
      setDocuments(latest)
      setDocumentSummary(summarizeDocuments(latest))
      setSelectedDocumentIds(latest.filter(document => document.status === 'ready').map(document => document.id))
    }
  }

  async function changeGlobalMemory(enabled) {
    const response = await apiFetch('/api/memory/settings', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({enabled}),
    })
    if (!response.ok) throw new Error('Could not update cross-session memory.')
    setMemoryEnabled((await response.json()).enabled)
    if (!enabled) setMemoryHits([])
  }

  async function changeConversationMemory(enabled) {
    if (!activeConv) return
    const response = await apiFetch(`/api/conversations/${activeConv}/memory`, {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({enabled}),
    })
    if (!response.ok) throw new Error('Could not update memory for this chat.')
    setConversationMemoryEnabled((await response.json()).enabled)
    if (!enabled) setMemoryHits([])
  }

  async function clearAllMemory() {
    const response = await apiFetch('/api/memory', {method: 'DELETE'})
    if (!response.ok) throw new Error('Could not clear cross-session memory.')
    setMemoryHits([])
  }

  async function submitMessageFeedback(messageId, rating, comment = '') {
    if (!messageId || !activeConv) return
    const response = await apiFetch(`/api/conversations/${activeConv}/messages/${messageId}/feedback`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({rating, comment}),
    })
    if (!response.ok) {
      const body = await response.json().catch(() => ({}))
      throw new Error(body.detail || 'Could not record feedback.')
    }
    const payload = await response.json()
    setMessages(current => current.map(message => message.id === messageId ? {
      ...message,
      rating: payload.rating,
      feedback: payload.feedback ?? message.feedback,
    } : message))
  }

  function newChat() {
    controllerRef.current?.abort()
    setActiveId(null)
    setDocuments([])
    setDocumentSummary(null)
    setMessages([])
    setTraceEvents([])
    setSelectedDocumentIds([])
    setMemoryHits([])
    setConversationMemoryEnabled(true)
    setPageError('')
    setSidebarOpen(false)
    setSettingsOpen(false)
    setCommandPaletteOpen(false)
  }

  async function renameChat(id, title) {
    const response = await apiFetch(`/api/conversations/${id}`, {
      method: 'PATCH',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({title}),
    })
    if (!response.ok) throw new Error('Could not rename this chat.')
    await refreshHistory()
  }

  async function deleteChat(id) {
    const response = await apiFetch(`/api/conversations/${id}`, {method: 'DELETE'})
    if (!response.ok) throw new Error('Could not delete this chat.')
    if (activeConvRef.current === id) newChat()
    await refreshHistory()
  }

  async function removeDocument(document) {
    if (!activeConvRef.current || !document.id) return
    const response = await apiFetch(`/api/conversations/${activeConvRef.current}/documents/${document.id}`, {method: 'DELETE'})
    if (!response.ok) {
      setPageError('Could not remove this file.')
      return
    }
    setDocuments(current => current.filter(item => item.id !== document.id))
    setSelectedDocumentIds(current => current.filter(id => id !== document.id))
    setDocumentSummary(current => summarizeDocuments(documents.filter(item => item.id !== document.id)))
    await refreshHistory()
  }

  async function downloadChat() {
    if (!activeConvRef.current) return
    const response = await apiFetch(`/api/conversations/${activeConvRef.current}/export`)
    if (!response.ok) throw new Error('Could not export this chat.')
    const objectUrl = URL.createObjectURL(await response.blob())
    const link = document.createElement('a')
    link.href = objectUrl
    link.download = 'agentic-rag-chat.txt'
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(objectUrl)
  }

  async function toggleVoiceOutput() {
    if (voiceOutput) {
      voiceOutputRef.current = false
      setVoiceOutput(false)
      try {
        if (audioContextRef.current?.state === 'running') await audioContextRef.current.suspend()
      } catch (error) {
        console.error('Could not mute speech playback.', error)
        setSpeechStatus(error.message || 'Could not mute speech playback.')
      }
      return
    }

    try {
      const AudioContextConstructor = window.AudioContext || window.webkitAudioContext
      if (!AudioContextConstructor) throw new Error('Audio playback is unavailable in this browser.')
      const audioContext = audioContextRef.current || new AudioContextConstructor()
      audioContextRef.current = audioContext
      await audioContext.resume()
      voiceOutputRef.current = true
      setVoiceOutput(true)
      setSpeechStatus('')
    } catch (error) {
      console.error('Could not enable speech playback.', error)
      setSpeechStatus(error.message || 'Audio playback is unavailable in this browser.')
    }
  }

  function settleAudioChunk() {
    audioPendingChunksRef.current = Math.max(0, audioPendingChunksRef.current - 1)
    if (audioGenerationDoneRef.current && audioPendingChunksRef.current === 0 && !audioFailureRef.current) {
      setSpeechStatus('')
    }
  }

  function queueSpeechChunk(payload) {
    audioPendingChunksRef.current += 1
    audioDecodeQueueRef.current = audioDecodeQueueRef.current.then(async () => {
      if (!voiceOutputRef.current) {
        settleAudioChunk()
        return
      }
      const audioContext = audioContextRef.current
      if (!audioContext) throw new Error('Audio playback was not initialized.')
      const binary = window.atob(payload.audio_base64 || '')
      const bytes = new Uint8Array(binary.length)
      for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index)
      const audioBuffer = await audioContext.decodeAudioData(bytes.buffer)
      if (!voiceOutputRef.current) {
        settleAudioChunk()
        return
      }
      const source = audioContext.createBufferSource()
      source.buffer = audioBuffer
      source.connect(audioContext.destination)
      const startsAt = Math.max(audioContext.currentTime + 0.04, audioNextStartRef.current)
      audioNextStartRef.current = startsAt + audioBuffer.duration
      source.addEventListener('ended', settleAudioChunk, {once: true})
      source.start(startsAt)
      setSpeechStatus('Playing generated speech…')
    }).catch(error => {
      settleAudioChunk()
      audioFailureRef.current = true
      console.error('Speech audio playback failed.', error)
      setSpeechStatus('Speech playback failed. The text answer is still available.')
    })
  }

  async function sendQuestion(text) {
    const question = text.trim()
    const hasReadyDocument = documents.some(document => document.status === 'ready')
    if (!question || (!hasReadyDocument && documents.length > 0)) return
    let id
    try {
      id = await ensureConversation()
    } catch (error) {
      setPageError(error.message || 'Could not start this chat.')
      return
    }
    setPageError('')
    audioGenerationDoneRef.current = false
    audioFailureRef.current = false
    setSpeechStatus(voiceOutput ? 'Generating speech…' : '')
    setTraceEvents([])
    setMemoryHits([])
    setMessages(current => [...current, {role: 'user', text: question}])
    setCollapsed(false)
    setIsStreaming(true)
    controllerRef.current = new AbortController()
    let assistantText = ''
    let answerPayload = null
    let frameBuffer = ''
    try {
      const response = await apiFetch(`/api/conversations/${id}/questions`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          question,
          ...settings,
          ...(voiceOutput ? {voice_output: true} : {}),
          document_ids: searchMode === 'selected' ? selectedDocumentIds : null,
        }),
        signal: controllerRef.current.signal,
      })
      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail?.message || body.detail || 'The answer could not be started.')
      }
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let receivedDone = false
      while (true) {
        const {done, value} = await reader.read()
        if (done) break
        frameBuffer += decoder.decode(value, {stream: true})
        const frames = frameBuffer.split(/\r?\n\r?\n/)
        frameBuffer = frames.pop() || ''
        for (const frame of frames) {
          const event = parseSseFrame(frame)
          if (!event) continue
          if (import.meta.env.DEV) console.debug('[Agentic RAG SSE]', event)
          if (event.type === 'trace') {
            setTraceEvents(current => [...current, event.data])
          } else if (event.type === 'memory_hits') {
            setMemoryHits(event.data.hits || [])
          } else if (event.type === 'map_progress') {
            setTraceEvents(current => [...current, {step: 'map_progress', ...event.data}])
          } else if (event.type === 'audio_chunk') {
            queueSpeechChunk(event.data)
          } else if (event.type === 'audio_error') {
            audioFailureRef.current = true
            console.error('Speech generation failed.', event.data)
            setSpeechStatus(event.data.error || 'Speech generation failed. The text answer is still available.')
          } else if (event.type === 'token') {
            assistantText += event.data.token || ''
            setMessages(current => {
              const next = [...current]
              const last = next[next.length - 1]
              if (last?.role === 'assistant' && last.streaming) next[next.length - 1] = {...last, text: assistantText}
              else next.push({role: 'assistant', text: assistantText, streaming: true, sources: []})
              return next
            })
          } else if (event.type === 'answer') {
            answerPayload = event.data
            assistantText = event.data.answer || assistantText
          } else if (event.type === 'error') {
            throw new Error(event.data.error || 'The answer could not be completed.')
          } else if (event.type === 'done') {
            receivedDone = true
            setTraceEvents(current => [...current, {step: 'complete'}])
            break
          }
        }
      }
      if (voiceOutput) {
        audioGenerationDoneRef.current = true
        audioDecodeQueueRef.current.then(() => {
          if (audioPendingChunksRef.current === 0 && !audioFailureRef.current) setSpeechStatus('')
        })
      }
      if (!receivedDone && !controllerRef.current?.signal.aborted) {
        notifySessionExpired()
        throw new Error('Your session expired while the answer was streaming. Sign in again.')
      }
      if (answerPayload) {
        const trace = answerPayload.trace || []
        const message = {
          role: 'assistant',
          text: answerPayload.answer || assistantText,
          sources: answerPayload.sources || [],
          trace,
          status: 'complete',
          outcome: trace.some(item => item.step === 'abstain') ? 'abstain' : trace.some(item => item.step === 'direct_answer') ? 'direct' : null,
        }
        setMessages(current => {
          const next = [...current]
          const last = next[next.length - 1]
          if (last?.role === 'assistant' && last.streaming) next[next.length - 1] = message
          else next.push(message)
          return next
        })
        await refreshHistory()
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        setMessages(current => current.map(message => message.streaming ? {...message, streaming: false, status: 'stopped'} : message))
      } else {
        const message = {role: 'assistant', text: error.message || 'Something went wrong. Please try again.', error: true, status: 'error', sources: []}
        setMessages(current => [...current, message])
        setPageError(message.text)
      }
    } finally {
      controllerRef.current = null
      setCollapsed(true)
      setIsStreaming(false)
    }
  }

  async function stopStream() {
    const id = activeConvRef.current
    if (id) apiFetch(`/api/conversations/${id}/cancel`, {method: 'POST'}).catch(() => {})
    controllerRef.current?.abort()
    controllerRef.current = null
    setIsStreaming(false)
  }

  async function submitRename(id, title) {
    try { await renameChat(id, title) } catch (error) { setPageError(error.message) }
  }

  async function toggleSettings() {
    const opening = !settingsOpen
    setSettingsOpen(opening)
    if (!opening || passwordStatus !== null) return
    try {
      const response = await apiFetch('/api/auth/password-status')
      if (!response.ok) throw new Error('Could not load password settings.')
      const payload = await response.json()
      setPasswordStatus(Boolean(payload.has_password))
    } catch (error) {
      setPasswordStatusError(error.message || 'Could not load password settings.')
    }
  }

  async function addPassword(event) {
    event.preventDefault()
    setPasswordStatusError('')
    setPasswordStatusMessage('')
    setPasswordSaving(true)
    try {
      const response = await apiFetch('/api/auth/add-password', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({password: passwordInput}),
      })
      const payload = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(payload.detail || 'Could not add a password.')
      setPasswordStatus(true)
      setPasswordInput('')
      setPasswordStatusMessage('Password sign-in is now enabled for this account.')
    } catch (error) {
      setPasswordStatusError(error.message || 'Could not add a password.')
    } finally {
      setPasswordSaving(false)
    }
  }

  const ready = documents.some(document => document.status === 'ready')
  const noDocuments = documents.length === 0
  const pending = documents.some(document => ['queued', 'processing', 'uploading'].includes(document.status))
  const canSend = noDocuments || (ready && (searchMode !== 'selected' || selectedDocumentIds.length > 0))
  const hasUserMessages = messages.some(message => message.role === 'user')
  const suggestions = canSend && !isStreaming && !hasUserMessages
    ? buildSuggestedQuestions(documents)
    : []

  if (authLoading) {
    return <main className="flex min-h-screen items-center justify-center bg-app-bg text-sm text-app-muted" role="status">Checking your session…</main>
  }
  const resetTokenPresent = new URLSearchParams(window.location.search).has('reset_token')
  if (currentPath === '/' && !resetTokenPresent) {
    return <Landing isAuthenticated={Boolean(currentUser)} onNavigate={navigateTo} />
  }
  if (!currentUser || currentPath === '/login' || resetTokenPresent) {
    return <AuthPanel initialMode={authMode} onAuthenticated={onAuthenticated} pageError={pageError} />
  }

  return <div className="flex h-screen min-h-0 bg-app-bg text-app-text">
    <Sidebar
      conversationId={activeConv}
      documents={documents}
      documentSummary={documentSummary}
      onEnsureConversation={ensureConversation}
      onUploaded={onUploadComplete}
      onDocumentStatus={updateDocument}
      onRemoveDocument={removeDocument}
      history={history}
      activeId={activeConv}
      onNewChat={newChat}
      onSelectChat={id => openConversation(id).catch(error => setPageError(error.message))}
      onRenameChat={submitRename}
      onDeleteChat={deleteChat}
      onDownload={downloadChat}
      settings={settings}
      onSettingsChange={setSettings}
      memoryEnabled={memoryEnabled}
      conversationMemoryEnabled={conversationMemoryEnabled}
      onGlobalMemoryChange={enabled => changeGlobalMemory(enabled).catch(error => setPageError(error.message))}
      onConversationMemoryChange={enabled => changeConversationMemory(enabled).catch(error => setPageError(error.message))}
      onClearAllMemory={() => clearAllMemory().catch(error => setPageError(error.message))}
      searchMode={searchMode}
      onSearchModeChange={setSearchMode}
      selectedDocumentIds={selectedDocumentIds}
      onSelectedDocumentIdsChange={setSelectedDocumentIds}
      open={sidebarOpen}
      onClose={() => setSidebarOpen(false)}
    />
    <main className="flex min-w-0 flex-1 flex-col bg-app-bg">
      <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-app-border bg-app-surface px-3 py-3 dark:bg-app-surface-2 md:px-4">
          <div className="flex items-center gap-2">
            <button type="button" aria-label="Open sidebar" onClick={() => setSidebarOpen(true)} className="rounded-md border border-app-border-strong p-2 text-app-muted hover:bg-app-surface-2 md:hidden"><Menu size={20}/></button>
            <h1 className="text-lg font-semibold">Agentic RAG</h1>
          </div>
          <div className="relative flex items-center gap-2">
            <button type="button" onClick={() => toggleSettings().catch(error => setPasswordStatusError(error.message))} className="rounded-md border border-app-border-strong bg-transparent px-3 py-1.5 text-sm text-app-text hover:border-app-muted hover:bg-app-surface-2">Settings</button>
            <button type="button" onClick={() => setCommandPaletteOpen(true)} className="rounded-md border border-app-border-strong bg-transparent px-3 py-1.5 text-sm text-app-text hover:border-app-muted hover:bg-app-surface-2">Commands</button>
            <span className="hidden max-w-48 truncate font-mono text-xs text-app-muted sm:block" title={currentUser.email}>{currentUser.email}</span>
            <button type="button" onClick={() => logout().catch(error => setPageError(error.message))} className="rounded-md border border-app-border-strong p-2 text-app-muted hover:border-app-muted hover:bg-app-surface-2" aria-label="Sign out" title="Sign out"><LogOut size={17}/></button>
            {settingsOpen && <div className="absolute right-0 top-12 z-20 w-64 rounded-xl border border-app-border bg-app-surface-2 p-3 shadow-lg">
              <label className="flex items-center gap-2 text-sm text-app-text">
                <input type="checkbox" checked={settings.show_reasoning_steps} onChange={event => setSettings(current => ({...current, show_reasoning_steps: event.target.checked}))} />
                Show reasoning steps
              </label>
              <div className="mt-3 border-t border-app-border pt-3">
                {passwordStatus === false ? <form onSubmit={addPassword} className="space-y-2">
                  <label className="block text-xs text-app-muted" htmlFor="account-add-password">Add password for email sign-in</label>
                  <input
                    id="account-add-password"
                    type="password"
                    autoComplete="new-password"
                    minLength={12}
                    maxLength={72}
                    required
                    value={passwordInput}
                    onChange={event => setPasswordInput(event.target.value)}
                    className="w-full rounded-md border border-app-border-strong bg-app-surface px-2 py-1.5 text-sm text-app-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-app-cyan"
                  />
                  <button type="submit" disabled={passwordSaving} className="w-full rounded-md bg-app-cyan px-2 py-1.5 text-sm font-medium text-slate-950 disabled:opacity-60">
                    {passwordSaving ? 'Saving…' : 'Add password'}
                  </button>
                </form> : passwordStatus === true
                  ? <p role="status" className="text-xs text-app-muted">Password sign-in is enabled.</p>
                  : <p role="status" className="text-xs text-app-muted">Loading password settings…</p>}
                {passwordStatusMessage && <p role="status" className="mt-2 text-xs text-app-green">{passwordStatusMessage}</p>}
                {passwordStatusError && <p role="alert" className="mt-2 text-xs text-red-300">{passwordStatusError}</p>}
              </div>
              <button type="button" onClick={() => { setCommandPaletteOpen(true); setSettingsOpen(false) }} className="mt-3 w-full rounded-md bg-app-surface-3 px-2 py-1.5 text-left text-sm text-app-text hover:bg-app-surface">Open command palette</button>
              <button type="button" onClick={() => setSettingsOpen(false)} className="mt-2 w-full rounded-md bg-app-surface px-2 py-1.5 text-left text-sm text-app-muted hover:bg-app-surface-3">Close</button>
            </div>}
            <ThemeToggle dark={dark} onToggle={() => setDark(value => !value)} />
          </div>
        </header>
        {commandPaletteOpen && <div className="fixed inset-0 z-40 flex items-start justify-center bg-slate-950/40 p-6 backdrop-blur-sm" onClick={() => setCommandPaletteOpen(false)}>
          <div className="w-full max-w-xl rounded-xl border border-app-border bg-app-surface p-3 shadow-2xl" onClick={event => event.stopPropagation()}>
            <div className="mb-2 flex items-center justify-between font-mono text-xs uppercase tracking-wide text-app-muted">
              <span>Command palette</span>
              <span>⌘K</span>
            </div>
            <div className="space-y-1">
              {[
                {label: 'New chat', action: () => { newChat(); setCommandPaletteOpen(false) }},
                {label: 'Toggle theme', action: () => { setDark(value => !value); setCommandPaletteOpen(false) }},
                {label: 'Open sidebar', action: () => { setSidebarOpen(true); setCommandPaletteOpen(false) }},
                {label: 'Refresh history', action: async () => { try { await refreshHistory(); setCommandPaletteOpen(false) } catch (error) { setPageError(error.message) } }},
              ].map(command => <button key={command.label} type="button" onClick={command.action} className="flex w-full items-center justify-between rounded-md px-3 py-2 text-left text-sm text-app-text hover:bg-app-surface-2"><span>{command.label}</span><span className="font-mono text-xs text-app-muted">↵</span></button>)}
            </div>
          </div>
        </div>}
        {(isStreaming || traceEvents.length > 0) && <PipelineStepper events={traceEvents} collapsed={collapsed} onToggle={() => setCollapsed(value => !value)} />}
        {pageError && <p role="alert" className="mx-4 mt-3 rounded bg-red-950 p-3 text-sm text-red-200">{pageError}</p>}
        <div className="flex-1 space-y-3 overflow-auto p-4" aria-live="polite">
          {memoryHits.length > 0 && <aside className="mb-3 rounded-xl border border-app-border bg-app-surface p-3" aria-label="Related past chats">
            <p className="mb-2 text-xs font-medium text-app-muted">Related past chats</p>
            <div className="flex flex-wrap gap-2">
              {memoryHits.map(hit => <button key={hit.memory_id} type="button" onClick={() => openConversation(hit.conversation_id).catch(error => setPageError(error.message))} className="max-w-full truncate rounded-md border border-app-border-strong px-2 py-1 text-left text-xs text-app-text hover:border-app-cyan hover:bg-app-cyan-soft" title={hit.question}>
                {hit.title || hit.question}
              </button>)}
            </div>
          </aside>}
          {!messages.length && <p className="mt-10 text-center text-sm text-app-muted">{history.length ? 'Choose a chat or add a document to start a new one.' : noDocuments ? 'No chats yet. Say hello or upload a document to start.' : 'Add a document to start.'}</p>}
          {messages.map((message, index) => <MessageBubble key={`${activeConv || 'draft'}-${message.id || index}`} message={message} idx={index} showReasoningSteps={settings.show_reasoning_steps} onFeedback={(messageId, rating, comment = '') => submitMessageFeedback(messageId, rating, comment).catch(error => setPageError(error.message))} />)}
        </div>
        {!ready && pending && <p className="border-t border-slate-200 px-4 py-2 text-xs text-slate-500 dark:border-slate-800">Your files are still being prepared. You can ask when one is ready.</p>}
        {!ready && !pending && documents.length > 0 && <p className="border-t border-slate-200 px-4 py-2 text-xs text-slate-500 dark:border-slate-800">Add or reprocess a file before asking a question.</p>}
        <InputBar onSend={sendQuestion} onStop={stopStream} disabled={!canSend} streaming={isStreaming} suggestions={suggestions} placeholder={noDocuments ? 'Say hello or ask a question' : ready ? 'Ask a question about your files' : 'Upload a document to ask a question'} voiceOutput={voiceOutput} onVoiceOutputToggle={toggleVoiceOutput} speechStatus={speechStatus} />
      </div>
    </main>
  </div>
}
