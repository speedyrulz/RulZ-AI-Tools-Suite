import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './App'
import './styles.css'

// The main process passes the saved look so the first paint matches it.
const initialTheme = new URLSearchParams(location.search).get('theme')
if (initialTheme === 'light' || initialTheme === 'dark') document.documentElement.classList.toggle('dark', initialTheme === 'dark')

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
)
