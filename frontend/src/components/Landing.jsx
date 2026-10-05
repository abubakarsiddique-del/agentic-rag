import React, {useEffect, useRef} from 'react'
import landingMarkup from './LandingContent.html?raw'
import './Landing.css'

const pageTitle = 'Agentic RAG — Ask your documents anything'

export default function Landing({isAuthenticated = false, onNavigate}) {
  const pageRef = useRef(null)
  const navigationRef = useRef({isAuthenticated, onNavigate})
  navigationRef.current = {isAuthenticated, onNavigate}

  useEffect(() => {
    const page = pageRef.current
    const nav = page?.querySelector('#nav')
    const menuButton = page?.querySelector('#menuBtn')
    const panel = page?.querySelector('#mobilePanel')
    if (!page || !nav || !menuButton || !panel) return undefined

    const closeMenu = () => {
      panel.classList.remove('open')
      menuButton.setAttribute('aria-expanded', 'false')
      menuButton.setAttribute('aria-label', 'Open menu')
    }
    const toggleMenu = () => {
      const open = !panel.classList.contains('open')
      panel.classList.toggle('open', open)
      menuButton.setAttribute('aria-expanded', String(open))
      menuButton.setAttribute('aria-label', open ? 'Close menu' : 'Open menu')
    }
    const handleClick = event => {
      if (!page.contains(event.target)) {
        closeMenu()
        return
      }
      const link = event.target.closest('a')
      if (!link) {
        if (panel.classList.contains('open') && !panel.contains(event.target) && !menuButton.contains(event.target)) {
          closeMenu()
        }
        return
      }
      if (panel.contains(link)) closeMenu()
      if (link.getAttribute('href') === '/login') {
        event.preventDefault()
        const {isAuthenticated: authenticated, onNavigate: navigate} = navigationRef.current
        navigate(authenticated ? '/app' : '/login')
      }
    }
    const handleScroll = () => nav.classList.toggle('scrolled', window.scrollY > 12)
    const handleMenuClick = event => {
      event.stopPropagation()
      toggleMenu()
    }
    menuButton.addEventListener('click', handleMenuClick)
    document.addEventListener('click', handleClick)
    window.addEventListener('scroll', handleScroll, {passive: true})
    handleScroll()

    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const reveals = page.querySelectorAll('.reveal')
    let revealObserver
    if (!reduceMotion && 'IntersectionObserver' in window) {
      revealObserver = new IntersectionObserver(entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('in')
            revealObserver.unobserve(entry.target)
          }
        })
      }, {threshold: 0.12, rootMargin: '0px 0px -30px'})
      reveals.forEach(element => revealObserver.observe(element))
    } else {
      reveals.forEach(element => element.classList.add('in'))
    }

    const navLinks = [...page.querySelectorAll('.nav-links a')]
    const sections = [...page.querySelectorAll('main section[id]')]
    let sectionObserver
    if ('IntersectionObserver' in window) {
      sectionObserver = new IntersectionObserver(entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            navLinks.forEach(link => {
              link.classList.toggle('active', link.getAttribute('href') === `#${entry.target.id}`)
            })
          }
        })
      }, {rootMargin: '-35% 0px -55% 0px', threshold: 0})
      sections.forEach(section => sectionObserver.observe(section))
    }

    document.title = pageTitle
    const pageUrl = `${window.location.origin}/`
    let canonical = document.querySelector('link[rel="canonical"]')
    if (!canonical) {
      canonical = document.createElement('link')
      canonical.rel = 'canonical'
      document.head.append(canonical)
    }
    canonical.href = pageUrl
    let openGraphUrl = document.querySelector('meta[property="og:url"]')
    if (!openGraphUrl) {
      openGraphUrl = document.createElement('meta')
      openGraphUrl.setAttribute('property', 'og:url')
      document.head.append(openGraphUrl)
    }
    openGraphUrl.content = pageUrl

    return () => {
      menuButton.removeEventListener('click', handleMenuClick)
      document.removeEventListener('click', handleClick)
      window.removeEventListener('scroll', handleScroll)
      revealObserver?.disconnect()
      sectionObserver?.disconnect()
    }
  }, [])

  return <div
    className="landing-page"
    ref={pageRef}
    dangerouslySetInnerHTML={{__html: landingMarkup}}
  />
}
