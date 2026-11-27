// electron-vite: `import p from './worker?modulePath'` bundles the file as its own entry and gives its path.
declare module '*?modulePath' {
  const path: string
  export default path
}
