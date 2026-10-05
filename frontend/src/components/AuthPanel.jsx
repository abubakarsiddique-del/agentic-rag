import React, {useState} from 'react'
import {KeyRound, LogIn, UserPlus} from 'lucide-react'
import {apiFetch} from '../api'
import './AuthPanel.css'

const RESET_MESSAGE = 'If the account exists, reset instructions have been sent.'

function responseMessage(payload, fallback) {
  if (typeof payload.detail === 'string') return payload.detail
  if (Array.isArray(payload.detail)) return payload.detail[0]?.msg || fallback
  return fallback
}

export default function AuthPanel({initialMode = 'signin', onAuthenticated, pageError = ''}) {
  const searchParams = new URLSearchParams(window.location.search)
  const resetToken = searchParams.get('reset_token') || ''
  const oauthError = searchParams.get('auth_error') === 'google_signin_failed'
  const [mode, setMode] = useState(resetToken ? 'reset-password' : initialMode)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [statusMessage, setStatusMessage] = useState('')
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const modeTitle = {
    signin: 'Sign in',
    signup: 'Create account',
    'request-reset': 'Reset password',
    'reset-password': 'Choose a new password',
  }[mode]

  async function submit(event) {
    event.preventDefault()
    setError('')
    setStatusMessage('')
    setSubmitting(true)
    try {
      if (mode === 'signin' || mode === 'signup') {
        const response = await apiFetch(`/api/auth/${mode}`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({email, password}),
        })
        const payload = await response.json().catch(() => ({}))
        if (!response.ok) throw new Error(responseMessage(payload, 'Authentication failed.'))
        await onAuthenticated(payload)
      } else if (mode === 'request-reset') {
        const response = await apiFetch('/api/auth/request-reset', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({email}),
        })
        const payload = await response.json().catch(() => ({}))
        if (!response.ok) throw new Error(responseMessage(payload, 'Could not request a reset.'))
        setStatusMessage(payload.message || RESET_MESSAGE)
      } else {
        const response = await apiFetch('/api/auth/reset-password', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({token: resetToken, password}),
        })
        const payload = await response.json().catch(() => ({}))
        if (!response.ok) throw new Error(responseMessage(payload, 'Reset token is invalid or expired.'))
        window.history.replaceState({}, '', window.location.pathname)
        setPassword('')
        setMode('signin')
        setStatusMessage('Password changed. Sign in with your new password.')
      }
    } catch (failure) {
      setError(failure.message || 'The request could not be completed.')
    } finally {
      setSubmitting(false)
    }
  }

  function changeMode(nextMode) {
    setMode(nextMode)
    setError('')
    setStatusMessage('')
  }

  return <main className="auth-page">
    <section className="auth-card" aria-labelledby="auth-heading">
      <a className="auth-brand" href="/" aria-label="Agentic RAG home">
        <span className="auth-brand-mark" aria-hidden="true"><KeyRound size={18}/></span>
        <span>Agentic RAG</span>
      </a>
      <div className="auth-heading">
        <p>Welcome back</p>
        <h1 id="auth-heading">{modeTitle}</h1>
      </div>

      {pageError && <p role="alert" className="auth-message auth-message-warning">{pageError}</p>}
      {oauthError && <p role="alert" className="auth-message auth-message-error">Google sign-in could not be completed. Please try again or use your email and password.</p>}
      {statusMessage && <p role="status" className="auth-message auth-message-success">{statusMessage}</p>}
      {error && <p role="alert" className="auth-message auth-message-error">{error}</p>}

      {(mode === 'signin' || mode === 'signup') && <>
      <form className="auth-form" onSubmit={submit}>
        <label className="auth-field">
          Email
          <input
            autoComplete="email"
            className="auth-input"
            type="email"
            required
            maxLength={254}
            value={email}
            onChange={event => setEmail(event.target.value)}
          />
        </label>
        <label className="auth-field">
          Password
          <input
            autoComplete={mode === 'signin' ? 'current-password' : 'new-password'}
            className="auth-input"
            type="password"
            required
            minLength={12}
            maxLength={72}
            value={password}
            onChange={event => setPassword(event.target.value)}
          />
        </label>
        <button className="btn btn-primary auth-submit" type="submit" disabled={submitting}>
          {mode === 'signup' ? <UserPlus size={17}/> : <LogIn size={17}/>}
          {submitting ? 'Working…' : modeTitle}
        </button>
      </form>
      <div className="auth-divider" role="separator"><span>or continue with</span></div>
      <a className="google-auth-link" href="/api/auth/google/login">
        <img src="/google-g-logo.png" alt="" aria-hidden="true" />
        <span>{mode === 'signup' ? 'Sign up with Google' : 'Sign in with Google'}</span>
      </a>
      </>}

      {(mode === 'request-reset' || mode === 'reset-password') && <form className="auth-form" onSubmit={submit}>
        {mode === 'request-reset' && <label className="auth-field">
          Email
          <input
            autoComplete="email"
            className="auth-input"
            type="email"
            required
            maxLength={254}
            value={email}
            onChange={event => setEmail(event.target.value)}
          />
        </label>}
        {mode === 'reset-password' && <label className="auth-field">
          New password
          <input
            autoComplete="new-password"
            className="auth-input"
            type="password"
            required
            minLength={12}
            maxLength={72}
            value={password}
            onChange={event => setPassword(event.target.value)}
          />
        </label>}
        <button className="btn btn-primary auth-submit" type="submit" disabled={submitting}>
          {submitting ? 'Working…' : modeTitle}
        </button>
      </form>}

      <div className="auth-actions">
        {mode === 'signin' && <>
          <button type="button" className="auth-link" onClick={() => changeMode('signup')}>Create an account</button>
          <button type="button" className="auth-link auth-link-muted" onClick={() => changeMode('request-reset')}>Forgot password?</button>
        </>}
        {mode === 'signup' && <button type="button" className="auth-link" onClick={() => changeMode('signin')}>Back to sign in</button>}
        {mode === 'request-reset' && <button type="button" className="auth-link" onClick={() => changeMode('signin')}>Back to sign in</button>}
        {mode === 'reset-password' && !resetToken && <button type="button" className="auth-link" onClick={() => changeMode('request-reset')}>Request a reset link</button>}
        {mode === 'reset-password' && resetToken && <button type="button" className="auth-link" onClick={() => { window.history.replaceState({}, '', window.location.pathname); changeMode('signin') }}>Back to sign in</button>}
      </div>
    </section>
  </main>
}
