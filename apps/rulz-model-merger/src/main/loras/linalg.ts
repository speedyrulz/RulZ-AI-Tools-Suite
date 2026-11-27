// Small dense linear algebra for LoRA merging (Float64Array, row-major). Matrices here are rank x rank (a few hundred
// at most); the big out x in matrix of a layer is never formed.

const EPS = 2 ** -52

/**
 * Eigen-decomposition of a symmetric n x n matrix: Householder reduction to tridiagonal form, then implicit QL
 * (the EISPACK tred2/tql2 route). Cyclic Jacobi has better relative accuracy but costs several times more at the
 * ranks a merge can reach (n up to 1024), and the callers only need absolute accuracy.
 * Values come back descending; `vectors` holds the eigenvectors as columns (vectors[row * n + col]).
 */
export function symmetricEigen(a: Float64Array, n: number): { values: Float64Array; vectors: Float64Array } {
  if (a.length !== n * n) throw new Error('symmetricEigen: the matrix size does not match n.')
  const values = new Float64Array(n)
  const vectors = new Float64Array(n * n)
  if (n === 0) return { values, vectors }
  if (n === 1) {
    values[0] = a[0]
    vectors[0] = 1
    return { values, vectors }
  }
  const V = Float64Array.from(a) // V[r*n + c]
  const d = new Float64Array(n)
  const e = new Float64Array(n)
  tred2(V, d, e, n)
  // Transpose so the QL rotations (which mix two eigenvectors) walk contiguous memory: T[i*n + k] = V[k][i].
  const T = new Float64Array(n * n)
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) T[c * n + r] = V[r * n + c]
  tql2(T, d, e, n)
  const order = Array.from({ length: n }, (_, i) => i).sort((x, y) => d[y] - d[x])
  for (let j = 0; j < n; j++) {
    const src = order[j]
    values[j] = d[src]
    for (let k = 0; k < n; k++) vectors[k * n + j] = T[src * n + k]
  }
  return { values, vectors }
}

function tred2(V: Float64Array, d: Float64Array, e: Float64Array, n: number): void {
  for (let j = 0; j < n; j++) d[j] = V[(n - 1) * n + j]
  for (let i = n - 1; i > 0; i--) {
    let scale = 0
    let h = 0
    for (let k = 0; k < i; k++) scale += Math.abs(d[k])
    if (scale === 0) {
      e[i] = d[i - 1]
      for (let j = 0; j < i; j++) {
        d[j] = V[(i - 1) * n + j]
        V[i * n + j] = 0
        V[j * n + i] = 0
      }
    } else {
      for (let k = 0; k < i; k++) {
        d[k] /= scale
        h += d[k] * d[k]
      }
      let f = d[i - 1]
      let g = Math.sqrt(h)
      if (f > 0) g = -g
      e[i] = scale * g
      h -= f * g
      d[i - 1] = f - g
      for (let j = 0; j < i; j++) e[j] = 0
      for (let j = 0; j < i; j++) {
        f = d[j]
        V[j * n + i] = f
        g = e[j] + V[j * n + j] * f
        for (let k = j + 1; k <= i - 1; k++) {
          g += V[k * n + j] * d[k]
          e[k] += V[k * n + j] * f
        }
        e[j] = g
      }
      f = 0
      for (let j = 0; j < i; j++) {
        e[j] /= h
        f += e[j] * d[j]
      }
      const hh = f / (h + h)
      for (let j = 0; j < i; j++) e[j] -= hh * d[j]
      for (let j = 0; j < i; j++) {
        f = d[j]
        g = e[j]
        for (let k = j; k <= i - 1; k++) V[k * n + j] -= f * e[k] + g * d[k]
        d[j] = V[(i - 1) * n + j]
        V[i * n + j] = 0
      }
    }
    d[i] = h
  }
  for (let i = 0; i < n - 1; i++) {
    V[(n - 1) * n + i] = V[i * n + i]
    V[i * n + i] = 1
    const h = d[i + 1]
    if (h !== 0) {
      for (let k = 0; k <= i; k++) d[k] = V[k * n + i + 1] / h
      for (let j = 0; j <= i; j++) {
        let g = 0
        for (let k = 0; k <= i; k++) g += V[k * n + i + 1] * V[k * n + j]
        for (let k = 0; k <= i; k++) V[k * n + j] -= g * d[k]
      }
    }
    for (let k = 0; k <= i; k++) V[k * n + i + 1] = 0
  }
  for (let j = 0; j < n; j++) {
    d[j] = V[(n - 1) * n + j]
    V[(n - 1) * n + j] = 0
  }
  V[(n - 1) * n + n - 1] = 1
  e[0] = 0
}

/** QL iteration. T holds the eigenvectors as rows (T[i*n + k] is component k of vector i). */
function tql2(T: Float64Array, d: Float64Array, e: Float64Array, n: number): void {
  for (let i = 1; i < n; i++) e[i - 1] = e[i]
  e[n - 1] = 0
  let f = 0
  let tst1 = 0
  for (let l = 0; l < n; l++) {
    tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]))
    let m = l
    while (m < n) {
      if (Math.abs(e[m]) <= EPS * tst1) break
      m++
    }
    if (m > l) {
      let iter = 0
      do {
        if (++iter > 200) throw new Error('symmetricEigen: the iteration did not converge.')
        let g = d[l]
        let p = (d[l + 1] - g) / (2 * e[l])
        let r = Math.hypot(p, 1)
        if (p < 0) r = -r
        d[l] = e[l] / (p + r)
        d[l + 1] = e[l] * (p + r)
        const dl1 = d[l + 1]
        let h = g - d[l]
        for (let i = l + 2; i < n; i++) d[i] -= h
        f += h
        p = d[m]
        let c = 1
        let c2 = c
        let c3 = c
        const el1 = e[l + 1]
        let s = 0
        let s2 = 0
        for (let i = m - 1; i >= l; i--) {
          c3 = c2
          c2 = c
          s2 = s
          g = c * e[i]
          h = c * p
          r = Math.hypot(p, e[i])
          e[i + 1] = s * r
          s = e[i] / r
          c = p / r
          p = c * d[i] - s * g
          d[i + 1] = h + s * (c * g + s * d[i])
          const ri = i * n
          const rj = (i + 1) * n
          for (let k = 0; k < n; k++) {
            const hv = T[rj + k]
            T[rj + k] = s * T[ri + k] + c * hv
            T[ri + k] = c * T[ri + k] - s * hv
          }
        }
        p = (-s * s2 * c3 * el1 * e[l]) / dl1
        e[l] = s * p
        d[l] = c * p
      } while (Math.abs(e[l]) > EPS * tst1)
    }
    d[l] += f
    e[l] = 0
  }
}

/**
 * Thin SVD of an m x n matrix: A = U diag(S) Vt with r = min(m, n), u m x r, s length r (descending), vt r x n.
 * Small matrices use one-sided Jacobi (accurate); larger ones go through the eigenvectors of the smaller Gram matrix,
 * with s taken as the norm of A v so that U S Vt always reproduces A. Columns of u for a zero singular value are zero.
 */
export function svdSmall(a: Float64Array, m: number, n: number): { u: Float64Array; s: Float64Array; vt: Float64Array } {
  if (a.length !== m * n) throw new Error('svdSmall: the matrix size does not match m x n.')
  if (m === 0 || n === 0) return { u: new Float64Array(0), s: new Float64Array(0), vt: new Float64Array(0) }
  if (m < n) {
    // A = (At)t: svd(At) = U' S V'^t gives A = V' S U'^t.
    const at = transpose(a, m, n)
    const r = svdSmall(at, n, m)
    return { u: transpose(r.vt, m, m), s: r.s, vt: transpose(r.u, n, m) }
  }
  return n <= 64 ? svdJacobi(a, m, n) : svdViaGram(a, m, n)
}

function transpose(a: Float64Array, rows: number, cols: number): Float64Array {
  const out = new Float64Array(a.length)
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[c * rows + r] = a[r * cols + c]
  return out
}

/** One-sided Jacobi for m >= n. */
function svdJacobi(a: Float64Array, m: number, n: number): { u: Float64Array; s: Float64Array; vt: Float64Array } {
  const U = transpose(a, m, n) // row j = column j of A (length m)
  const V = new Float64Array(n * n) // row j = column j of V (length n)
  for (let i = 0; i < n; i++) V[i * n + i] = 1
  for (let sweep = 0; sweep < 60; sweep++) {
    let rotated = false
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const pu = p * m
        const qu = q * m
        let alpha = 0
        let beta = 0
        let gamma = 0
        for (let i = 0; i < m; i++) {
          const x = U[pu + i]
          const y = U[qu + i]
          alpha += x * x
          beta += y * y
          gamma += x * y
        }
        if (gamma === 0 || Math.abs(gamma) <= 1e-15 * Math.sqrt(alpha * beta)) continue
        rotated = true
        const zeta = (beta - alpha) / (2 * gamma)
        const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta))
        const c = 1 / Math.sqrt(1 + t * t)
        const s = c * t
        for (let i = 0; i < m; i++) {
          const x = U[pu + i]
          const y = U[qu + i]
          U[pu + i] = c * x - s * y
          U[qu + i] = s * x + c * y
        }
        const pv = p * n
        const qv = q * n
        for (let i = 0; i < n; i++) {
          const x = V[pv + i]
          const y = V[qv + i]
          V[pv + i] = c * x - s * y
          V[qv + i] = s * x + c * y
        }
      }
    }
    if (!rotated) break
  }
  const norms = new Float64Array(n)
  for (let j = 0; j < n; j++) {
    let sum = 0
    for (let i = 0; i < m; i++) sum += U[j * m + i] * U[j * m + i]
    norms[j] = Math.sqrt(sum)
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((x, y) => norms[y] - norms[x])
  const u = new Float64Array(m * n)
  const s = new Float64Array(n)
  const vt = new Float64Array(n * n)
  for (let jj = 0; jj < n; jj++) {
    const j = order[jj]
    s[jj] = norms[j]
    if (norms[j] > 0) for (let i = 0; i < m; i++) u[i * n + jj] = U[j * m + i] / norms[j]
    for (let i = 0; i < n; i++) vt[jj * n + i] = V[j * n + i]
  }
  return { u, s, vt }
}

/** m >= n: eigenvectors of AtA give V; U S = A V. */
function svdViaGram(a: Float64Array, m: number, n: number): { u: Float64Array; s: Float64Array; vt: Float64Array } {
  const g = new Float64Array(n * n)
  for (let i = 0; i < m; i++) {
    const row = i * n
    for (let p = 0; p < n; p++) {
      const x = a[row + p]
      if (x === 0) continue
      const gp = p * n
      for (let q = p; q < n; q++) g[gp + q] += x * a[row + q]
    }
  }
  for (let p = 0; p < n; p++) for (let q = 0; q < p; q++) g[p * n + q] = g[q * n + p]
  const { vectors } = symmetricEigen(g, n)
  // AV (m x n), column norms are the singular values.
  const av = new Float64Array(m * n)
  for (let i = 0; i < m; i++) {
    const row = i * n
    for (let k = 0; k < n; k++) {
      const x = a[row + k]
      if (x === 0) continue
      const vk = k * n
      for (let j = 0; j < n; j++) av[row + j] += x * vectors[vk + j]
    }
  }
  const norms = new Float64Array(n)
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) norms[j] += av[i * n + j] * av[i * n + j]
  for (let j = 0; j < n; j++) norms[j] = Math.sqrt(norms[j])
  const order = Array.from({ length: n }, (_, i) => i).sort((x, y) => norms[y] - norms[x])
  const u = new Float64Array(m * n)
  const s = new Float64Array(n)
  const vt = new Float64Array(n * n)
  for (let jj = 0; jj < n; jj++) {
    const j = order[jj]
    s[jj] = norms[j]
    if (norms[j] > 0) for (let i = 0; i < m; i++) u[i * n + jj] = av[i * n + j] / norms[j]
    for (let i = 0; i < n; i++) vt[jj * n + i] = vectors[i * n + j]
  }
  return { u, s, vt }
}

const DROP = 1e-12

/**
 * Reduces the layer delta = up @ down (up out x rank, down rank x inFlat) to at most k terms with the best
 * approximation in the Frobenius norm, without forming out x inFlat. The new factors share the singular values
 * evenly (up' = U sqrt(S), down' = sqrt(S) Vt).
 */
export function shrinkLayer(
  up: Float64Array,
  down: Float64Array,
  out: number,
  inFlat: number,
  rank: number,
  k: number
): { up: Float64Array; down: Float64Array; k: number; energyKept: number } {
  if (up.length !== out * rank || down.length !== rank * inFlat) throw new Error('shrinkLayer: the factor sizes do not match.')
  if (k < 1) throw new Error('shrinkLayer: k must be at least 1.')

  // Gram matrices: Gb = up^t up (rank x rank), Ga = down down^t.
  const gb = new Float64Array(rank * rank)
  for (let o = 0; o < out; o++) {
    const row = o * rank
    for (let i = 0; i < rank; i++) {
      const x = up[row + i]
      if (x === 0) continue
      const gi = i * rank
      for (let j = i; j < rank; j++) gb[gi + j] += x * up[row + j]
    }
  }
  const ga = new Float64Array(rank * rank)
  for (let i = 0; i < rank; i++) {
    const ri = i * inFlat
    for (let j = i; j < rank; j++) {
      const rj = j * inFlat
      let sum = 0
      for (let c = 0; c < inFlat; c++) sum += down[ri + c] * down[rj + c]
      ga[i * rank + j] = sum
    }
  }
  for (let i = 0; i < rank; i++) {
    for (let j = 0; j < i; j++) {
      gb[i * rank + j] = gb[j * rank + i]
      ga[i * rank + j] = ga[j * rank + i]
    }
  }
  const eb = symmetricEigen(gb, rank)
  const ea = symmetricEigen(ga, rank)
  const rb = countAbove(eb.values, DROP)
  const ra = countAbove(ea.values, DROP)
  if (rb === 0 || ra === 0) {
    // A zero layer: keep a single zero term so the file still has a valid rank.
    return { up: new Float64Array(out), down: new Float64Array(inFlat), k: 1, energyKept: 1 }
  }

  // C = Lb^1/2 Vb^t Va La^1/2  (rb x ra); Vb, Va are the first rb / ra eigenvector columns.
  const vbtva = new Float64Array(rb * ra)
  for (let r = 0; r < rank; r++) {
    for (let i = 0; i < rb; i++) {
      const x = eb.vectors[r * rank + i]
      if (x === 0) continue
      const ci = i * ra
      const ar = r * rank
      for (let j = 0; j < ra; j++) vbtva[ci + j] += x * ea.vectors[ar + j]
    }
  }
  const sb = new Float64Array(rb)
  const sa = new Float64Array(ra)
  for (let i = 0; i < rb; i++) sb[i] = Math.sqrt(eb.values[i])
  for (let j = 0; j < ra; j++) sa[j] = Math.sqrt(ea.values[j])
  const core = new Float64Array(rb * ra)
  for (let i = 0; i < rb; i++) for (let j = 0; j < ra; j++) core[i * ra + j] = sb[i] * vbtva[i * ra + j] * sa[j]

  const svd = svdSmall(core, rb, ra)
  const r = svd.s.length
  let numRank = r
  while (numRank > 0 && svd.s[numRank - 1] <= DROP * svd.s[0]) numRank--
  if (numRank === 0) return { up: new Float64Array(out), down: new Float64Array(inFlat), k: 1, energyKept: 1 }
  const kk = Math.min(k, numRank)
  let total = 0
  let kept = 0
  for (let i = 0; i < numRank; i++) {
    const e = svd.s[i] * svd.s[i]
    total += e
    if (i < kk) kept += e
  }
  const energyKept = kk >= numRank || total === 0 ? 1 : kept / total

  // M1 = Vb Lb^-1/2 U_k S_k^1/2  (rank x kk), then up' = up M1.
  const ubs = new Float64Array(rb * kk) // Lb^-1/2 U_k S_k^1/2
  for (let i = 0; i < rb; i++) {
    for (let c = 0; c < kk; c++) ubs[i * kk + c] = (svd.u[i * r + c] * Math.sqrt(svd.s[c])) / sb[i]
  }
  const m1 = new Float64Array(rank * kk)
  for (let row = 0; row < rank; row++) {
    for (let i = 0; i < rb; i++) {
      const x = eb.vectors[row * rank + i]
      if (x === 0) continue
      for (let c = 0; c < kk; c++) m1[row * kk + c] += x * ubs[i * kk + c]
    }
  }
  const newUp = new Float64Array(out * kk)
  for (let o = 0; o < out; o++) {
    const row = o * rank
    for (let i = 0; i < rank; i++) {
      const x = up[row + i]
      if (x === 0) continue
      for (let c = 0; c < kk; c++) newUp[o * kk + c] += x * m1[i * kk + c]
    }
  }

  // M2 = S_k^1/2 Wk^t La^-1/2 Va^t  (kk x rank), then down' = M2 down.
  const m2 = new Float64Array(kk * rank)
  for (let c = 0; c < kk; c++) {
    const sq = Math.sqrt(svd.s[c])
    for (let j = 0; j < ra; j++) {
      const w = (svd.vt[c * ra + j] * sq) / sa[j]
      if (w === 0) continue
      for (let row = 0; row < rank; row++) m2[c * rank + row] += w * ea.vectors[row * rank + j]
    }
  }
  const newDown = new Float64Array(kk * inFlat)
  for (let c = 0; c < kk; c++) {
    const dst = c * inFlat
    for (let row = 0; row < rank; row++) {
      const x = m2[c * rank + row]
      if (x === 0) continue
      const src = row * inFlat
      for (let col = 0; col < inFlat; col++) newDown[dst + col] += x * down[src + col]
    }
  }
  return { up: newUp, down: newDown, k: kk, energyKept }
}

function countAbove(values: Float64Array, rel: number): number {
  const max = values.length > 0 ? values[0] : 0
  if (!(max > 0)) return 0
  let n = 0
  while (n < values.length && values[n] > rel * max) n++
  return n
}
