import { api } from '../api.js'

const PAPER_S1 = { vgg16: 5, inception_v3: 4, pneumonet_v2: 3, pneumonet_v1: 1 }
const EQUAL_S2 = { pneumonet_v1: 1, mobilenet_v2: 1, inception_v3: 1 }
const S2_235 = { pneumonet_v1: 2, mobilenet_v2: 3, inception_v3: 5 }

let s1Weights = { ...PAPER_S1 }
let s2Weights = { ...EQUAL_S2 }
let threshold = 0.5
let debounceTimer = null
let memberNames = null

export async function renderPlayground(view) {
  view.innerHTML = `
    <h1>Playground</h1>
    <p class="lede">Adjust ensemble weights and the Stage 1 threshold. The accuracy
    and confusion matrix update instantly from cached test-set predictions — no
    model runs.</p>

    <div class="banner">⚠️ The Stage 2 weights 2/3/5 were grid-searched on this
    test set, so their accuracy is optimistic by ~0.5–1.0 points.</div>

    <div class="grid grid--2">
      <div class="card">
        <h3>Stage 1 weights</h3>
        <div id="s1-sliders"></div>
        <div style="margin-top:.75rem">
          <button id="s1-paper" class="ghost">Paper (5/4/3/1)</button>
          <button id="s1-equal" class="ghost">Equal</button>
          <button id="s1-drop-inc" class="ghost">Drop Inception-v3</button>
        </div>
      </div>

      <div class="card">
        <h3>Stage 2 weights</h3>
        <div id="s2-sliders"></div>
        <div style="margin-top:.75rem">
          <button id="s2-equal" class="ghost">Equal</button>
          <button id="s2-235" class="ghost">2/3/5</button>
        </div>
        <h3 style="margin-top:1.5rem">Stage 1 threshold</h3>
        <div class="prob-row">
          <span class="prob-label">P(pneumonia) ≥</span>
          <input type="range" id="threshold" min="0" max="1" step="0.01" value="${threshold}" />
          <span class="prob-value" id="threshold-val">${threshold.toFixed(2)}</span>
        </div>
      </div>
    </div>

    <div class="card">
      <h3>Test-set accuracy</h3>
      <div id="metrics"><p class="loading">Computing…</p></div>
    </div>

    <div class="card" id="cm-card" style="display:none">
      <h3>Confusion matrix (624 test images)</h3>
      <div id="cm"></div>
    </div>
  `

  const sliderDefs = [
    { stage: 's1', name: 'vgg16', label: 'VGG-16', min: 0, max: 10 },
    { stage: 's1', name: 'inception_v3', label: 'Inception-v3', min: 0, max: 10 },
    { stage: 's1', name: 'pneumonet_v2', label: 'PneumoNet-v2', min: 0, max: 10 },
    { stage: 's1', name: 'pneumonet_v1', label: 'PneumoNet-v1', min: 0, max: 10 },
    { stage: 's2', name: 'pneumonet_v1', label: 'PneumoNet-v1', min: 0, max: 10 },
    { stage: 's2', name: 'mobilenet_v2', label: 'MobileNet-v2', min: 0, max: 10 },
    { stage: 's2', name: 'inception_v3', label: 'Inception-v3', min: 0, max: 10 }
  ]

  function buildSliders() {
    document.getElementById('s1-sliders').innerHTML = sliderDefs
      .filter(d => d.stage === 's1')
      .map(d => `<div class="member-row">
        <span class="member-name">${d.label}</span>
        <input type="range" data-stage="${d.stage}" data-name="${d.name}"
               min="${d.min}" max="${d.max}" step="1" value="${s1Weights[d.name]}" />
        <span class="member-value" id="s1-${d.name}">${s1Weights[d.name]}</span>
      </div>`).join('')

    document.getElementById('s2-sliders').innerHTML = sliderDefs
      .filter(d => d.stage === 's2')
      .map(d => `<div class="member-row">
        <span class="member-name">${d.label}</span>
        <input type="range" data-stage="${d.stage}" data-name="${d.name}"
               min="${d.min}" max="${d.max}" step="1" value="${s2Weights[d.name]}" />
        <span class="member-value" id="s2-${d.name}">${s2Weights[d.name]}</span>
      </div>`).join('')

    view.querySelectorAll('input[type="range"]').forEach(input => {
      input.addEventListener('input', () => {
        const stage = input.dataset.stage
        const name = input.dataset.name
        const val = parseInt(input.value, 10)
        if (stage === 's1') s1Weights[name] = val
        else s2Weights[name] = val
        document.getElementById(`${stage}-${name}`).textContent = val
        debouncedEvaluate()
      })
    })
  }

  document.getElementById('s1-paper').onclick = () => { s1Weights = { ...PAPER_S1 }; buildSliders(); evaluate() }
  document.getElementById('s1-equal').onclick = () => { s1Weights = { vgg16: 1, inception_v3: 1, pneumonet_v2: 1, pneumonet_v1: 1 }; buildSliders(); evaluate() }
  document.getElementById('s1-drop-inc').onclick = () => { s1Weights = { vgg16: 5, pneumonet_v2: 3, pneumonet_v1: 1 }; s1Weights.inception_v3 = 0; buildSliders(); evaluate() }
  document.getElementById('s2-equal').onclick = () => { s2Weights = { ...EQUAL_S2 }; buildSliders(); evaluate() }
  document.getElementById('s2-235').onclick = () => { s2Weights = { ...S2_235 }; buildSliders(); evaluate() }

  const thresholdInput = view.querySelector('#threshold')
  thresholdInput.addEventListener('input', () => {
    threshold = parseFloat(thresholdInput.value)
    document.getElementById('threshold-val').textContent = threshold.toFixed(2)
    debouncedEvaluate()
  })

  buildSliders()

  try {
    memberNames = await api.get('/testset/members')
  } catch {
    // fall back to defaults already in PAPER_S1 / EQUAL_S2
  }

  await evaluate()

  function debouncedEvaluate() {
    clearTimeout(debounceTimer)
    debounceTimer = setTimeout(evaluate, 120)
  }

  async function evaluate() {
    const el = document.getElementById('metrics')
    try {
      const r = await api.post('/testset/evaluate', {
        stage1_weights: s1Weights,
        stage2_weights: s2Weights,
        threshold
      })
      showMetrics(r)
      showCM(r)
    } catch (err) {
      el.innerHTML = `<div class="error"><p>${err.message}</p></div>`
    }
  }

  function showMetrics(r) {
    const el = document.getElementById('metrics')
    const acc = (r.accuracy * 100).toFixed(2)
    const baseline = 78.53
    const delta = ((r.accuracy - 0.7853) * 100).toFixed(2)
    const deltaStr = delta >= 0 ? `+${delta}` : delta

    el.innerHTML = `
      <div style="font-size:2rem;font-weight:600;margin-bottom:.5rem">${acc}%</div>
      <dl class="kv">
        <dt>Correct</dt><dd>${r.correct} / ${r.total}</dd>
        <dt>Δ vs paper config</dt><dd>${deltaStr} pts</dd>
        <dt>Routed to Stage 2</dt><dd>${r.routed_to_stage2}</dd>
        <dt>Stage 1 missed pneumonia</dt><dd>${390 - r.routed_to_stage2}</dd>
      </dl>
      <div style="margin-top:1rem">
        ${['NORMAL', 'BACTERIAL', 'VIRAL'].map(c => {
          const m = r.per_class[c]
          return `<div class="prob-row">
            <span class="prob-label">${c}</span>
            <div class="prob-track"><div class="prob-fill" style="width:${(m.f1*100).toFixed(1)}%"></div></div>
            <span class="prob-value">F1 ${(m.f1*100).toFixed(1)}%</span>
          </div>`
        }).join('')}
      </div>
    `
  }

  function showCM(r) {
    const card = document.getElementById('cm-card')
    const el = document.getElementById('cm')
    card.style.display = 'block'
    const labels = r.class_order
    const cm = r.confusion_matrix

    el.innerHTML = `<table class="cm">
      <tr><th></th>${labels.map(l => `<th>${l}</th>`).join('')}</tr>
      ${cm.map((row, i) => `<tr><th>${labels[i]}</th>${
        row.map((v, j) => `<td class="${i === j ? 'diag' : 'off-diag'}">${v}</td>`).join('')
      }</tr>`).join('')}
    </table>
    <p class="hint" style="margin-top:.5rem">Rows = true label, columns = predicted.</p>`
  }
}
