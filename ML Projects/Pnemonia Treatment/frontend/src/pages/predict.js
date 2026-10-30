import { api } from '../api.js'

let lastPrediction = null

export async function renderPredict(view) {
  view.innerHTML = `
    <h1>Predict</h1>
    <p class="lede">Upload a chest X-ray or pick a sample. The two-stage pipeline
    classifies it as Normal, Bacterial pneumonia, or Viral pneumonia.</p>

    <div class="banner">⚠️ Educational use only — not a medical device. Trained on
    paediatric patients (ages 1–5) from a single institution.</div>

    <div class="grid grid--2">
      <div class="card">
        <h3>Input</h3>
        <div class="dropzone" id="drop">
          <p>Drop an image here, or click to browse</p>
          <p class="hint">JPEG or PNG, any size — will be resized to 224×224</p>
          <input type="file" id="file" accept="image/*" />
        </div>
        <div id="preview" style="margin-top:1rem"></div>
        <div id="samples" style="margin-top:1rem"></div>
      </div>

      <div class="card">
        <h3>Result</h3>
        <div id="result"><p class="hint">No prediction yet.</p></div>
      </div>
    </div>

    <div class="card" id="members-card" style="display:none">
      <h3>Member contributions</h3>
      <p class="hint">Each model's P(pneumonia) before ensemble weighting. Click a
      weight preset to see how re-weighting changes the verdict.</p>
      <div id="members"></div>
      <div id="presets" style="margin-top:1rem"></div>
    </div>
  `

  const drop = view.querySelector('#drop')
  const fileInput = view.querySelector('#file')
  const preview = view.querySelector('#preview')
  const resultEl = view.querySelector('#result')

  drop.addEventListener('click', () => fileInput.click())
  fileInput.addEventListener('change', () => fileInput.files[0] && handleFile(fileInput.files[0]))

  ;['dragover', 'dragenter'].forEach(ev =>
    drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('drag') }))
  ;['dragleave', 'drop'].forEach(ev =>
    drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('drag') }))
  drop.addEventListener('drop', e => {
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  })

  await loadSamples(view.querySelector('#samples'))

  async function handleFile(file) {
    preview.innerHTML = `<img src="${URL.createObjectURL(file)}" style="max-width:100%;border-radius:8px" />`
    resultEl.innerHTML = '<p class="loading">Running…</p>'
    try {
      const pred = await api.upload(file)
      lastPrediction = pred
      showResult(resultEl, pred)
      showMembers(pred)
    } catch (err) {
      resultEl.innerHTML = `<div class="error"><p>${err.message}</p></div>`
    }
  }

  function showResult(el, pred) {
    const labelClass = `class-${pred.label}`
    const rows = ['NORMAL', 'BACTERIAL', 'VIRAL'].map(c => {
      const p = (pred.class_probs[c] || 0)
      return `<div class="prob-row ${labelClass}">
        <span class="prob-label">${c}</span>
        <div class="prob-track"><div class="prob-fill" style="width:${(p*100).toFixed(1)}%"></div></div>
        <span class="prob-value">${(p*100).toFixed(1)}%</span>
      </div>`
    }).join('')

    el.innerHTML = `
      <div style="font-size:1.3rem;font-weight:600;margin-bottom:.5rem">${pred.label}</div>
      <div class="hint" style="margin-bottom:1rem">
        Confidence: ${(pred.confidence*100).toFixed(1)}% ·
        ${pred.routed ? 'Routed to Stage 2' : 'Stopped at Stage 1'}
      </div>
      ${rows}
      <div style="margin-top:1rem;font-size:.85rem;color:var(--muted)">
        Stage 1 ensemble P(pneumonia): ${(pred.stage1_probs[1]*100).toFixed(1)}%
        ${pred.stage2_probs ? `<br>Stage 2 ensemble P(bacterial): ${(pred.stage2_probs[0]*100).toFixed(1)}%` : ''}
      </div>
    `
  }

  function showMembers(pred) {
    const card = document.getElementById('members-card')
    const el = document.getElementById('members')
    card.style.display = 'block'

    const s1Rows = Object.entries(pred.stage1_members).map(([name, p]) =>
      `<div class="prob-row"><span class="prob-label">${name}</span>
       <div class="prob-track"><div class="prob-fill" style="width:${(p*100).toFixed(1)}%"></div></div>
       <span class="prob-value">${(p*100).toFixed(1)}%</span></div>`
    ).join('')

    const s2Html = pred.stage2_members ? `<h3 style="margin-top:1rem">Stage 2 members</h3>` +
      Object.entries(pred.stage2_members).map(([name, p]) =>
        `<div class="prob-row"><span class="prob-label">${name}</span>
         <div class="prob-track"><div class="prob-fill" style="width:${(p*100).toFixed(1)}%"></div></div>
         <span class="prob-value">${(p*100).toFixed(1)}%</span></div>`
      ).join('') : ''

    el.innerHTML = `<h3>Stage 1 members — P(pneumonia)</h3>${s1Rows}${s2Html}`

    const presets = document.getElementById('presets')
    presets.innerHTML = `
      <button id="preset-paper" class="ghost">Paper weights (5/4/3/1)</button>
      <button id="preset-equal" class="ghost">Equal weights</button>
      <button id="preset-drop-inc" class="ghost">Drop Inception-v3</button>
      <span id="preset-result" style="margin-left:1rem"></span>
    `

    const paper = { vgg16: 5, inception_v3: 4, pneumonet_v2: 3, pneumonet_v1: 1 }
    const equal = { vgg16: 1, inception_v3: 1, pneumonet_v2: 1, pneumonet_v1: 1 }
    const dropInc = { vgg16: 5, pneumonet_v2: 3, pneumonet_v1: 1 }

    async function apply(weights) {
      const resultEl2 = document.getElementById('preset-result')
      resultEl2.textContent = 'Re-blending…'
      try {
        const reb = await api.post('/reblend', {
          stage1_members: Object.fromEntries(
            Object.entries(pred.stage1_members).map(([k, v]) => [k, [1-v, v]])
          ),
          stage2_members: pred.stage2_members
            ? Object.fromEntries(Object.entries(pred.stage2_members).map(([k, v]) => [k, [1-v, v]]))
            : null,
          stage1_weights: weights
        })
        resultEl2.innerHTML = `→ <strong>${reb.label}</strong> (${(reb.confidence*100).toFixed(1)}%)`
        showResult(document.getElementById('result'), reb)
      } catch (err) {
        resultEl2.textContent = err.message
      }
    }

    document.getElementById('preset-paper').onclick = () => apply(paper)
    document.getElementById('preset-equal').onclick = () => apply(equal)
    document.getElementById('preset-drop-inc').onclick = () => apply(dropInc)
  }

  async function loadSamples(el) {
    try {
      const samples = await api.get('/samples')
      if (!samples.length) { el.innerHTML = ''; return }
      el.innerHTML = `<h3>Or try a sample</h3><div class="gallery">${
        samples.slice(0, 18).map(s =>
          `<button data-url="${s.url}" title="True label: ${s.true_label}">
            <img src="${s.url}" alt="${s.name}" loading="lazy" />
            <div class="tag">${s.true_label}</div>
          </button>`
        ).join('')
      }</div>`

      el.querySelectorAll('button').forEach(btn => {
        btn.onclick = async () => {
          const resp = await fetch(btn.dataset.url)
          const blob = await resp.blob()
          const file = new File([blob], 'sample.jpeg', { type: 'image/jpeg' })
          handleFile(file)
        }
      })
    } catch {
      el.innerHTML = ''
    }
  }
}
