import type { ThemeName } from '@shared/types'
import logoDark from '../assets/logo.png'
import logoLight from '../assets/logo-light.png'
import markDark from '../assets/mark.png'
import markLight from '../assets/mark-light.png'

/** The theme the main process opened the window with (dark until one is chosen). */
export const initialTheme: ThemeName = new URLSearchParams(location.search).get('theme') === 'light' ? 'light' : 'dark'

export function applyThemeClass(theme: ThemeName): void {
  document.documentElement.classList.toggle('light', theme === 'light')
  document.documentElement.style.colorScheme = theme
}

// The dark UI uses the logo with white lettering; the light UI uses the original black lettering.
export const LOGOS: Record<ThemeName, { logo: string; mark: string }> = {
  dark: { logo: logoDark, mark: markDark },
  light: { logo: logoLight, mark: markLight }
}
