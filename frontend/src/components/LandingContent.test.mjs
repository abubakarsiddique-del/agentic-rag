import {readFileSync} from 'node:fs'
import {describe, expect, it} from 'vitest'

const markup = readFileSync(new URL('./LandingContent.html', import.meta.url), 'utf8')
const rootStyles = readFileSync(new URL('../index.css', import.meta.url), 'utf8')
const landingStyles = readFileSync(new URL('./Landing.css', import.meta.url), 'utf8')
const authStyles = readFileSync(new URL('./AuthPanel.css', import.meta.url), 'utf8')
const links = [...markup.matchAll(/<a\b[^>]*\bhref=(["'])(.*?)\1[^>]*>([\s\S]*?)<\/a>/g)]
const ids = new Set([...markup.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]))

describe('landing page integration', () => {
  it('has no bare placeholder anchors and every in-page link targets an element', () => {
    expect(links.some(([, , href]) => href === '#')).toBe(false)
    for (const [, , href] of links) {
      if (href.startsWith('#')) expect(ids.has(href.slice(1)), href).toBe(true)
    }
    for (const href of ['#features', '#how', '#security', '#get-started']) {
      expect(links.some(([, , actualHref]) => actualHref === href)).toBe(true)
    }
  })

  it('routes both Start free calls to the existing auth page', () => {
    const startLinks = links.filter(([, , , label]) => label.includes('Start free'))
    expect(startLinks).toHaveLength(2)
    expect(startLinks.every(([, , href]) => href === '/login')).toBe(true)
  })

  it('uses explicit intentional destinations for the unavailable company pages', () => {
    for (const [label, href] of [['Contact', '/contact'], ['Privacy', '/privacy'], ['Terms', '/terms']]) {
      expect(links.some(([, , actualHref, content]) => actualHref === href && content.includes(label))).toBe(true)
    }
  })

  it('keeps footer headings at the next valid heading level', () => {
    expect(markup).toContain('<h3>Product</h3>')
    expect(markup).toContain('<h3>Company</h3>')
    expect(markup).not.toContain('<h4>')
  })

  it('shares the landing page palette and type tokens with the auth screens', () => {
    const expected = {
      '--bg': '#07090d',
      '--surface': '#0d1117',
      '--surface-2': '#121821',
      '--surface-3': '#171f29',
      '--border': 'rgba(255, 255, 255, .085)',
      '--border-strong': 'rgba(255, 255, 255, .15)',
      '--text': '#f4f7fa',
      '--muted': '#99a4b1',
      '--dim': '#66717f',
      '--cyan': '#38d9f2',
      '--cyan-soft': 'rgba(56, 217, 242, .12)',
      '--violet': '#8b8cf7',
      '--green': '#69e5a1',
    }
    for (const [token, value] of Object.entries(expected)) {
      const declared = rootStyles.match(new RegExp(`${token}:\\s*([^;]+);`))?.[1].trim()
      expect(declared, token).toBe(value)
    }
    expect(rootStyles).toContain('--font-manrope:')
    expect(rootStyles).toContain('--font-mono:')
    expect(landingStyles).toContain('font-family:var(--font-manrope)')
    expect(landingStyles).toContain('var(--font-mono)')
    expect(authStyles).toContain('font-family: var(--font-manrope)')
    expect(authStyles).toContain('var(--font-mono)')
    expect(authStyles).toContain('background: var(--cyan)')
    expect(authStyles).toContain('background: var(--surface-2)')
  })
})
