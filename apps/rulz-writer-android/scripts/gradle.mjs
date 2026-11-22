// Runs the Android build (Gradle) in android/: node scripts/gradle.mjs assembleDebug
// Needs Java 21 (JAVA_HOME) and the Android SDK (ANDROID_HOME, or sdk.dir in android/local.properties).
import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { join } from 'node:path'

const android = join(import.meta.dirname, '..', 'android')
const hasSdk = process.env.ANDROID_HOME || process.env.ANDROID_SDK_ROOT || existsSync(join(android, 'local.properties'))
if (!hasSdk) {
  console.error('Android SDK not found. Set ANDROID_HOME, or put sdk.dir=C:\\\\path\\\\to\\\\Sdk in android/local.properties.')
  process.exit(1)
}
const windows = process.platform === 'win32'
const r = spawnSync(windows ? '.\\gradlew.bat' : './gradlew', process.argv.slice(2), { cwd: android, stdio: 'inherit', shell: windows })
if (r.status === 0 && process.argv.includes('assembleDebug')) {
  console.log(`\nAPK: ${join(android, 'app', 'build', 'outputs', 'apk', 'debug', 'app-debug.apk')}`)
}
process.exit(r.status ?? 1)
