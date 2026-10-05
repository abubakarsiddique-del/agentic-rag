import React, {useRef, useEffect, useState} from 'react'
import { Mic, Volume2, VolumeX } from 'lucide-react'
import {apiFetch} from '../api'

const RECORDING_MIME_TYPES = ['audio/webm;codecs=opus', 'audio/mp4;codecs=mp4a.40.2', 'audio/mp4']

export default function InputBar({onSend, onStop, disabled, streaming, suggestions = [], placeholder, voiceOutput = false, onVoiceOutputToggle, speechStatus = ''}){
  const [text, setText] = useState('')
  const [recording, setRecording] = useState(false)
  const [requestingMic, setRequestingMic] = useState(false)
  const [transcribing, setTranscribing] = useState(false)
  const [voiceStatus, setVoiceStatus] = useState('')
  const taRef = useRef(null)
  const isComposingRef = useRef(false)
  const heldRef = useRef(false)
  const startingRef = useRef(false)
  const disposedRef = useRef(false)
  const recordingErrorRef = useRef(false)
  const recorderRef = useRef(null)
  const streamRef = useRef(null)
  const chunksRef = useRef([])
  const maxDurationRef = useRef(null)
  const recordingMimeType = typeof MediaRecorder !== 'undefined' && typeof MediaRecorder.isTypeSupported === 'function'
    ? RECORDING_MIME_TYPES.find(type => MediaRecorder.isTypeSupported(type))
    : null
  const recordingSupported = Boolean(recordingMimeType) && typeof navigator !== 'undefined' && Boolean(navigator.mediaDevices?.getUserMedia)

  useEffect(()=>{
    if(taRef.current){
      taRef.current.style.height = 'auto'
      taRef.current.style.height = taRef.current.scrollHeight + 'px'
    }
  },[text])

  useEffect(() => () => {
    disposedRef.current = true
    heldRef.current = false
    clearTimeout(maxDurationRef.current)
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
    streamRef.current?.getTracks().forEach(track => track.stop())
  }, [])

  function submit(e){
    e.preventDefault()
    if(!text.trim()) return
    onSend(text)
    setText('')
  }

  async function transcribeRecording(blob) {
    if (!blob.size) {
      setVoiceStatus('The recording was empty. Try again.')
      return
    }
    setTranscribing(true)
    setVoiceStatus('Transcribing…')
    try {
      const form = new FormData()
      const filename = blob.type.toLowerCase().includes('mp4') ? 'recording.mp4' : 'recording.webm'
      form.append('audio', blob, filename)
      const response = await apiFetch('/api/voice/transcribe', {method: 'POST', body: form})
      const payload = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(payload.detail || `Transcription failed (${response.status}).`)
      setText(payload.transcript || '')
      setVoiceStatus(payload.transcript ? 'Transcript ready to review.' : 'No speech was recognized. Try recording again.')
      if (import.meta.env.DEV) console.info('[Voice transcription response]', payload)
    } catch (error) {
      console.error('Voice transcription request failed.', error)
      setVoiceStatus(error.message || 'Transcription failed. Try again or type your message.')
    } finally {
      setTranscribing(false)
    }
  }

  async function startRecording() {
    if (!recordingSupported || transcribing || streaming || disabled || startingRef.current || recorderRef.current) return
    startingRef.current = true
    setRequestingMic(true)
    setVoiceStatus('Requesting microphone access…')
    let stream
    try {
      stream = await navigator.mediaDevices.getUserMedia({audio: true})
      if (!heldRef.current || disposedRef.current) {
        stream.getTracks().forEach(track => track.stop())
        stream = null
        if (!disposedRef.current) setVoiceStatus('Recording canceled. Hold to record again.')
        return
      }
      streamRef.current = stream
      const recorder = new MediaRecorder(stream, {mimeType: recordingMimeType})
      recorderRef.current = recorder
      chunksRef.current = []
      recordingErrorRef.current = false
      recorder.addEventListener('dataavailable', event => {
        if (event.data.size) chunksRef.current.push(event.data)
      })
      recorder.addEventListener('stop', () => {
        clearTimeout(maxDurationRef.current)
        const blob = new Blob(chunksRef.current, {type: recorder.mimeType || recordingMimeType})
        stream.getTracks().forEach(track => track.stop())
        streamRef.current = null
        recorderRef.current = null
        if (disposedRef.current || recordingErrorRef.current) return
        setRecording(false)
        transcribeRecording(blob)
      }, {once: true})
      recorder.addEventListener('error', event => {
        recordingErrorRef.current = true
        console.error('Audio recording failed.', event.error)
        stream.getTracks().forEach(track => track.stop())
        streamRef.current = null
        recorderRef.current = null
        setRecording(false)
        setVoiceStatus(event.error?.message || 'Recording failed. Try again or type your message.')
      }, {once: true})
      recorder.start()
      setRecording(true)
      setVoiceStatus(`Recording… release to transcribe (10 second maximum, ${recorder.mimeType || recordingMimeType}).`)
      maxDurationRef.current = setTimeout(() => {
        if (recorder.state === 'recording') {
          heldRef.current = false
          setVoiceStatus('Maximum recording time reached. Transcribing…')
          recorder.stop()
        }
      }, 10000)
    } catch (error) {
      stream?.getTracks().forEach(track => track.stop())
      streamRef.current = null
      recorderRef.current = null
      heldRef.current = false
      if (!disposedRef.current) {
        console.error('Could not start audio recording.', error)
        setVoiceStatus(error.message || 'Microphone unavailable. You can type your message instead.')
      }
    } finally {
      startingRef.current = false
      if (!disposedRef.current) setRequestingMic(false)
    }
  }

  function stopRecording() {
    heldRef.current = false
    clearTimeout(maxDurationRef.current)
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
  }

  function beginPointerRecording(event) {
    if (event.button !== 0 || disabled || streaming || transcribing) return
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    heldRef.current = true
    startRecording()
  }

  function handleRecordKeyDown(event) {
    if (![' ', 'Enter'].includes(event.key) || event.repeat || disabled || streaming || transcribing) return
    event.preventDefault()
    heldRef.current = true
    startRecording()
  }

  function handleRecordKeyUp(event) {
    if (![' ', 'Enter'].includes(event.key)) return
    event.preventDefault()
    stopRecording()
  }

  return (
    <div className="border-t border-app-border bg-app-surface dark:bg-app-surface-2">
      {suggestions.length > 0 && !disabled && <section aria-label="Suggested questions" className="px-4 pt-3">
        <p className="mb-2 text-xs font-medium text-app-muted">Suggested questions</p>
        <div className="flex flex-wrap gap-2">
          {suggestions.map((suggestion, index) => <button
            key={`${index}-${suggestion}`}
            type="button"
            onClick={() => onSend(suggestion)}
            className="rounded-lg border border-app-border-strong bg-app-surface-2 px-3 py-1.5 text-left text-xs font-medium text-app-text hover:border-app-cyan hover:bg-app-cyan-soft"
          >{suggestion}</button>)}
        </div>
      </section>}
      <form onSubmit={submit} className="flex items-end gap-3 p-4">
        <textarea
          ref={taRef}
          value={text}
          disabled={disabled || streaming || recording || requestingMic || transcribing}
          onChange={e => setText(e.target.value)}
          placeholder={placeholder || (disabled ? 'Upload a document to ask a question' : 'Ask a question about your files')}
          className="flex-1 resize-none rounded-xl border border-app-border-strong bg-app-surface-2 p-3 text-app-text placeholder:text-app-muted focus:border-app-cyan disabled:opacity-60"
          rows={1}
          onCompositionStart={() => { isComposingRef.current = true }}
          onCompositionEnd={() => { isComposingRef.current = false }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !isComposingRef.current) {
              e.preventDefault()
              if (!streaming && !recording && !requestingMic && !transcribing) submit(e)
            }
          }}
        />
        <div className="flex items-center gap-2">
          {onVoiceOutputToggle && <button type="button" onClick={onVoiceOutputToggle} disabled={streaming || disabled} aria-label={voiceOutput ? 'Mute voice output' : 'Enable voice output'} aria-pressed={voiceOutput} title={voiceOutput ? 'Mute voice output' : 'Enable voice output'} className={`rounded-md border p-2 disabled:opacity-50 ${voiceOutput ? 'border-app-cyan bg-app-cyan-soft text-app-cyan' : 'border-app-border-strong text-app-muted hover:border-app-cyan hover:bg-app-surface-2'}`}>
            {voiceOutput ? <Volume2 size={18}/> : <VolumeX size={18}/>}
          </button>}
          {recordingSupported && <button type="button" onPointerDown={beginPointerRecording} onPointerUp={stopRecording} onPointerCancel={stopRecording} onKeyDown={handleRecordKeyDown} onKeyUp={handleRecordKeyUp} disabled={transcribing || streaming || disabled} aria-label="Hold to record voice input" aria-pressed={recording || requestingMic} title="Hold to record" className={`rounded-md border p-2 disabled:opacity-50 ${recording || requestingMic ? 'border-red-500 bg-red-950/50 text-red-200' : 'border-app-border-strong text-app-muted hover:border-app-cyan hover:bg-app-surface-2'}`}>
            <Mic size={18}/>
          </button>}
          {streaming ? (
            <button type="button" onClick={onStop} className="rounded-md border border-app-border-strong bg-app-surface-2 px-4 py-2 text-app-text hover:border-app-cyan">Stop</button>
          ) : (
            <button type="submit" disabled={!text.trim() || disabled || recording || requestingMic || transcribing} className="rounded-md bg-app-cyan px-4 py-2 font-semibold text-app-cyan-ink transition-opacity hover:opacity-90 disabled:opacity-50">Send</button>
          )}
        </div>
      </form>
      {voiceStatus && <p role="status" className="px-4 pb-3 text-xs text-app-muted">{voiceStatus}</p>}
      {speechStatus && <p role="status" className="px-4 pb-3 text-xs text-app-muted">{speechStatus}</p>}
    </div>
  )
}
