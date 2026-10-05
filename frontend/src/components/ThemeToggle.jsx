import React from 'react'
import { Moon, Sun } from 'lucide-react'

export default function ThemeToggle({dark, onToggle}){
  return (
    <button type="button" onClick={onToggle} aria-label="Toggle theme" aria-pressed={dark} className="rounded-md border border-app-border-strong p-2 text-app-muted hover:border-app-cyan hover:bg-app-cyan-soft hover:text-app-cyan">
      {dark? <Sun/> : <Moon/>}
    </button>
  )
}
