import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest'
import {apiFetch, invalidateCsrfToken} from './api.js'

function jsonResponse(payload, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => payload,
  }
}

describe('shared API client', () => {
  beforeEach(() => {
    invalidateCsrfToken()
    globalThis.fetch = vi.fn()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('includes cookies and a synchronizer header on state-changing calls', async () => {
    globalThis.fetch
      .mockResolvedValueOnce(jsonResponse({csrf_token: 'csrf-value'}))
      .mockResolvedValueOnce(jsonResponse({ok: true}))

    await apiFetch('/api/conversations', {method: 'POST', body: '{}'})

    expect(globalThis.fetch).toHaveBeenNthCalledWith(1, '/api/auth/csrf', {credentials: 'include'})
    const [, request] = globalThis.fetch.mock.calls[1]
    expect(request.credentials).toBe('include')
    expect(request.headers.get('X-CSRF-Token')).toBe('csrf-value')
  })

  it('refreshes a one-use pre-auth token after each reset operation', async () => {
    globalThis.fetch
      .mockResolvedValueOnce(jsonResponse({csrf_token: 'request-token'}))
      .mockResolvedValueOnce(jsonResponse({message: 'generic'}))
      .mockResolvedValueOnce(jsonResponse({csrf_token: 'reset-token'}))
      .mockResolvedValueOnce(jsonResponse({password_reset: true}))

    await apiFetch('/api/auth/request-reset', {method: 'POST', body: '{}'})
    await apiFetch('/api/auth/reset-password', {method: 'POST', body: '{}'})

    expect(globalThis.fetch).toHaveBeenNthCalledWith(1, '/api/auth/csrf', {credentials: 'include'})
    expect(globalThis.fetch).toHaveBeenNthCalledWith(3, '/api/auth/csrf', {credentials: 'include'})
    expect(globalThis.fetch.mock.calls[1][1].headers.get('X-CSRF-Token')).toBe('request-token')
    expect(globalThis.fetch.mock.calls[3][1].headers.get('X-CSRF-Token')).toBe('reset-token')
  })
})
