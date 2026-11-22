import type { CapacitorConfig } from '@capacitor/cli'

const config: CapacitorConfig = {
  appId: 'com.rulz.writer.android',
  appName: 'RulZ Writer',
  webDir: 'dist',
  // The app connects to the computer with ws:// over Tailscale (which encrypts it), so the page itself is
  // served over http to keep that from counting as mixed content.
  server: { androidScheme: 'http' },
  android: { backgroundColor: '#050505' },
  plugins: {
    SystemBars: { insetsHandling: 'css', initialViewportFitValueHint: 'cover' }
  }
}

export default config
