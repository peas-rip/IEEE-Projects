// Thin API client. No state management library — the pages are small enough
// that a few fetch calls and direct DOM writes are clearer than React.

const API = '/api'

async function get(path) {
  const r = await fetch(API + path)
  if (!r.ok) {
    const body = await r.json().catch(() => ({}))
    throw new Error(body.detail || `${r.status} ${r.statusText}`)
  }
  return r.json()
}

async function post(path, body) {
  const r = await fetch(API + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  })
  if (!r.ok) {
    const b = await r.json().catch(() => ({}))
    throw new Error(b.detail || `${r.status} ${r.statusText}`)
  }
  return r.json()
}

async function upload(file) {
  const form = new FormData()
  form.append('file', file)
  const r = await fetch(API + '/predict', { method: 'POST', body: form })
  if (!r.ok) {
    const b = await r.json().catch(() => ({}))
    throw new Error(b.detail || `${r.status} ${r.statusText}`)
  }
  return r.json()
}

export const api = { get, post, upload }
