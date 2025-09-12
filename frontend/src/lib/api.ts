export const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export async function apiPreview(file: File) {
  const fd = new FormData()
  fd.append('file', file)
  const res = await fetch(`${API_BASE}/api/preview`, {
    method: 'POST',
    body: fd,
  })
  if (!res.ok) throw new Error('Preview failed')
  return res.json()
}

export async function apiKMeans(file: File, params: { k: number; max_iter?: number; tol?: number; columns?: string; random_state?: number }) {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('k', String(params.k))
  if (params.max_iter != null) fd.append('max_iter', String(params.max_iter))
  if (params.tol != null) fd.append('tol', String(params.tol))
  if (params.columns) fd.append('columns', params.columns)
  if (params.random_state != null) fd.append('random_state', String(params.random_state))
  const res = await fetch(`${API_BASE}/api/kmeans`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error('KMeans failed')
  return res.json()
}

export async function apiNaiveBayes(file: File, params: { target: string; variant: 'gaussian' | 'multinomial'; alpha?: number; feature_columns?: string }) {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('target', params.target)
  fd.append('variant', params.variant)
  if (params.alpha != null) fd.append('alpha', String(params.alpha))
  if (params.feature_columns) fd.append('feature_columns', params.feature_columns)
  const res = await fetch(`${API_BASE}/api/naive-bayes`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error('Naive Bayes failed')
  return res.json()
}

export async function apiDecisionTree(file: File, params: { target: string; max_depth?: number | null; min_samples_split?: number }) {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('target', params.target)
  if (params.max_depth != null) fd.append('max_depth', String(params.max_depth))
  if (params.min_samples_split != null) fd.append('min_samples_split', String(params.min_samples_split))
  const res = await fetch(`${API_BASE}/api/decision-tree`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error('Decision Tree failed')
  return res.json()
}

export async function apiReduct(file: File, params: { decision: string; conditional?: string }) {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('decision', params.decision)
  if (params.conditional) fd.append('conditional', params.conditional)
  const res = await fetch(`${API_BASE}/api/reduct`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error('Reduct failed')
  return res.json()
}

