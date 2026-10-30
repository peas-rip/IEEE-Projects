import { api } from './api.js'
import { renderPredict } from './pages/predict.js'
import { renderPlayground } from './pages/playground.js'
import { renderResults } from './pages/results.js'
import { renderAbout } from './pages/about.js'

const view = document.getElementById('view')
const pill = document.getElementById('health-pill')

const routes = {
  predict: renderPredict,
  playground: renderPlayground,
  results: renderResults,
  about: renderAbout
}

function currentPage() {
  const hash = location.hash.replace(/^#\//, '')
  return routes[hash] ? hash : 'predict'
}

function setActiveNav() {
  document.querySelectorAll('.nav a').forEach(a => {
    a.classList.toggle('active', a.dataset.page === currentPage())
  })
}

async function render() {
  setActiveNav()
  view.innerHTML = '<p class="loading">Loading…</p>'
  try {
    await routes[currentPage()](view)
  } catch (err) {
    view.innerHTML = `<div class="error">
      <h2>Error</h2><p>${err.message}</p>
      <p class="hint">If the backend isn't running, start it with <code>run_app.bat</code>.</p>
    </div>`
  }
}

async function loadHealth() {
  try {
    const h = await api.get('/health')
    pill.textContent = h.ready ? 'Ready' : 'Not ready'
    pill.className = 'pill ' + (h.ready ? 'pill--ok' : 'pill--warn')
    pill.title = h.ready
      ? `Device: ${h.device}`
      : `Missing checkpoints: ${Object.entries(h.checkpoints).filter(([, v]) => !v.present).map(([k]) => k).join(', ')}`
  } catch {
    pill.textContent = 'Offline'
    pill.className = 'pill pill--err'
  }
}

window.addEventListener('hashchange', render)
render()
loadHealth()
setInterval(loadHealth, 30000)
