import { api } from '../api.js'

export async function renderResults(view) {
  view.innerHTML = `
    <h1>Results</h1>
    <p class="lede">What this reproduction actually achieved, measured on the
    official Kermany test split (624 images). The paper reports 98.03% for its
    detection ensemble; this project gets 86.06% and the gap is explained below.</p>

    <div class="banner">⚠️ These are the numbers from the Colab run. See
    <a href="#/about">About</a> for why they differ from the paper's.</div>

    <div id="content"><p class="loading">Loading results…</p></div>
  `

  let results
  try {
    results = await api.get('/results')
  } catch (err) {
    view.querySelector('#content').innerHTML =
      `<div class="error"><p>${err.message}</p></div>`
    return
  }

  const el = view.querySelector('#content')

  const headline = [
    ['Stage 1 ensemble', results.stage1.ensemble.accuracy, results.stage1.ensemble.auc_ovr],
    ['Stage 2 ensemble', results.stage2.ensemble.accuracy, results.stage2.ensemble.auc_ovr],
    ['Full pipeline', results.pipeline.accuracy, results.pipeline.auc_ovr],
    ['Drop Inception-v3 (Stage 1)', results.ablations.results[1].accuracy, null]
  ]

  const memberRows = [
    ...results.stage1.members.map(m => ({ ...m, stage: 'Stage 1' })),
    ...results.stage2.members.map(m => ({ ...m, stage: 'Stage 2' }))
  ]

  const ablations = results.ablations.results

  el.innerHTML = `
    <div class="card">
      <h3>Headline figures</h3>
      <table>
        <tr><th>Metric</th><th class="num">Accuracy</th><th class="num">AUC</th></tr>
        ${headline.map(([name, acc, auc]) => `<tr>
          <td>${name}</td>
          <td class="num"><strong>${(acc*100).toFixed(2)}%</strong></td>
          <td class="num">${auc ? auc.toFixed(4) : '—'}</td>
        </tr>`).join('')}
      </table>
    </div>

    <div class="card">
      <h3>Per-member accuracy</h3>
      <table>
        <tr><th>Stage</th><th>Model</th><th>Family</th><th class="num">Accuracy</th><th class="num">AUC</th></tr>
        ${memberRows.map(m => `<tr>
          <td>${m.stage}</td>
          <td>${m.name}</td>
          <td class="hint">${m.family.split(',')[0]}</td>
          <td class="num">${(m.accuracy*100).toFixed(2)}%</td>
          <td class="num">${m.auc_ovr.toFixed(4)}</td>
        </tr>`).join('')}
      </table>
    </div>

    <div class="card">
      <h3>Ablations (no retraining)</h3>
      <table>
        <tr><th>Configuration</th><th class="num">Accuracy</th><th class="num">Δ</th><th class="num">Images fixed</th></tr>
        ${ablations.map(a => `<tr>
          <td>${a.description}</td>
          <td class="num"><strong>${(a.accuracy*100).toFixed(2)}%</strong></td>
          <td class="num">${a.delta_vs_baseline ? '+' + (a.delta_vs_baseline*100).toFixed(2) : '—'}</td>
          <td class="num">${a.images_fixed || '—'}</td>
        </tr>`).join('')}
      </table>
    </div>

    <div class="card">
      <h3>Error decomposition</h3>
      <p class="hint">${results.pipeline.error_analysis.observed_pattern}</p>
      <table>
        <tr><th>Cause</th><th class="num">Images</th><th class="num">Share</th></tr>
        <tr><td>Healthy lungs flagged as pneumonia</td><td class="num">${results.pipeline.error_analysis.stage1_false_positives_normal_called_pneumonia}</td><td class="num">61.2%</td></tr>
        <tr><td>Pneumonia missed by Stage 1</td><td class="num">${results.pipeline.error_analysis.stage1_false_negatives_pneumonia_missed}</td><td class="num">3.7%</td></tr>
        <tr><td>Subtype confusion on correctly detected pneumonia</td><td class="num">47</td><td class="num">35.1%</td></tr>
      </table>
    </div>

    <div class="card">
      <h3>Deviations from the paper</h3>
      <table>
        <tr><th>Item</th><th>Label</th><th>Impact</th></tr>
        ${results.deviations_from_paper.map(d => `<tr>
          <td>${d.item}</td>
          <td><span class="badge ${d.label === 'DIFFERENT PROTOCOL' ? 'badge--warn' : ''}">${d.label}</span></td>
          <td class="hint">${d.impact}</td>
        </tr>`).join('')}
      </table>
    </div>

    <div class="card">
      <h3>Figures</h3>
      <div class="grid grid--2" style="margin-top:1rem">
        <div>
          <img src="/api/figures/fig_pipeline_confusion.png" alt="Pipeline confusion matrix" style="width:100%;border:1px solid var(--line);border-radius:8px" />
          <p class="hint" style="text-align:center">Pipeline confusion matrix</p>
        </div>
        <div>
          <img src="/api/figures/fig_ablations.png" alt="Ablation comparison" style="width:100%;border:1px solid var(--line);border-radius:8px" />
          <p class="hint" style="text-align:center">Ablation comparison</p>
        </div>
      </div>
    </div>
  `
}
