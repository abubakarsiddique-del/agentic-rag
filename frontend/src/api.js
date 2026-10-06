const API_BASE = process.env.NEXT_PUBLIC_API_BASE
const CSRF_EXEMPT_PATHS = new Set(['/api/auth/signup', '/api/auth/signin'])
let csrfToken = null
let csrfRequest = null


function isUnsafe(method) {
  return !['GET', 'HEAD', 'OPTIONS'].includes(String(method || 'GET').toUpperCase())
}

function needsCsrf(path, method) {
  if (isUnsafe(method)) return true
  const parts = path.replace(/^\//, '').split('/')
  if (String(method).toUpperCase() !== 'GET' || parts[0] !== 'api' || parts[1] !== 'conversations') return false
  return parts.length === 3 || (parts.length === 4 && ['documents', 'status'].includes(parts[3]))
}

function notifyUnauthorized(path) {
  if (path === '/api/auth/signin' || path === '/api/auth/signup' || path === '/api/auth/me') return
  csrfToken = null
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('rag:unauthorized'))
  }
}

async function getCsrfToken() {
  if (csrfToken) return csrfToken
  if (!csrfRequest) {
    csrfRequest = fetch(`${API_BASE}/api/auth/csrf`, {credentials: 'include'})
      .then(async response => {
        if (!response.ok) throw new Error('Could not prepare a secure request.')
        const payload = await response.json()
        csrfToken = payload.csrf_token
        return csrfToken
      })
      .finally(() => { csrfRequest = null })
  }
  return csrfRequest
}

export function invalidateCsrfToken() {
  csrfToken = null
  csrfRequest = null
}

export async function apiFetch(path, init = {}) {
  const method = String(init.method || 'GET').toUpperCase()
  const headers = new Headers(init.headers || {})
  if (needsCsrf(path, method) && !CSRF_EXEMPT_PATHS.has(path)) {
    headers.set('X-CSRF-Token', await getCsrfToken())
  }
  const response = await fetch(`${API_BASE}/${path}`, {
    ...init,
    method,
    headers,
    credentials: 'include',
  })
  if (response.status === 401) notifyUnauthorized(path)
  if (path === '/api/auth/signin' || path === '/api/auth/signup' ||
      path === '/api/auth/request-reset' || path === '/api/auth/reset-password' ||
      path === '/api/auth/logout') {
    invalidateCsrfToken()
  }
  return response
}

export async function apiUpload(path, form, onProgress) {
  const token = await getCsrfToken()
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest()
    request.open('POST', `${API_BASE}/${path}`)
    request.withCredentials = true
    request.setRequestHeader('X-CSRF-Token', token)
    request.upload.addEventListener('progress', event => {
      if (event.lengthComputable && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100))
      }
    })
    request.onload = () => {
      if (request.status === 401) notifyUnauthorized(path)
      resolve({
        status: request.status,
        ok: request.status >= 200 && request.status < 300,
        json: async () => JSON.parse(request.responseText),
        text: request.responseText,
      })
    }
    request.onerror = () => reject(new Error('The upload was interrupted. Check your connection and try again.'))
    request.send(form)
  })
}

export function notifySessionExpired() {
  notifyUnauthorized('/api/auth/session-expired')
}
