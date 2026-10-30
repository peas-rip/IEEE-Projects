export async function renderAbout(view) {
  view.innerHTML = `
    <h1>About</h1>
    <p class="lede">This is an educational reimplementation of the two-stage
    pneumonia ensemble from Godbole et al. (2024). It is not a clinical tool.</p>

    <div class="banner">⚠️ <strong>Educational use only.</strong> Not a medical
    device. Trained on paediatric patients (ages 1–5) from a single institution.
    Any other pathology, adult patient, or out-of-distribution image will be
    misclassified. Probabilities are uncalibrated classifier outputs, not
    clinical risks.</div>

    <div class="card">
      <h3>What this project does</h3>
      <p>Two weighted ensembles arranged as an inference cascade:</p>
      <ol>
        <li><strong>Stage 1 (detection)</strong> — Normal vs Pneumonia. Four models:
        VGG-16, Inception-v3, PneumoNet-v1, PneumoNet-v2. Paper weights: 5/4/3/1.</li>
        <li><strong>Stage 2 (classification)</strong> — Bacterial vs Viral, applied only
        if Stage 1 flags pneumonia. Three models: Inception-v3 (Xception substitute),
        MobileNet-v2, PneumoNet-v1. Equal weights.</li>
      </ol>
      <p class="hint">Input: 224×224 grayscale chest X-ray, replicated to 3 channels.</p>
    </div>

    <div class="card">
      <h3>Source paper</h3>
      <p>Godbole, S., Kattukaran, A., Savla, S., Pradhan, V., Kanani, P., & Patil, D.
      (2024). "Enhancing Paediatric Pneumonia Detection and Classification Using
      Customized CNNs and Transfer Learning Based Ensemble Models."
      <em>Int. Res. J. Multidisc. Technovation</em>, 6(6), 38–53.
      <a href="https://doi.org/10.54392/irjtmt2463" target="_blank">doi:10.54392/irjtmt2463</a></p>
      <p class="hint">Paper licence: CC BY 4.0. This project is an independent
      reimplementation, not affiliated with or endorsed by the authors.</p>
    </div>

    <div class="card">
      <h3>Why the numbers differ from the paper</h3>
      <table>
        <tr><th>Factor</th><th>Impact</th></tr>
        <tr><td>Evaluation protocol</td><td class="hint">Paper: random 80:20 split over all images. This project: official patient-disjoint Kermany test split.</td></tr>
        <tr><td>Training budget</td><td class="hint">Paper: unknown epochs. This project: 15 epochs, flat lr 1e-4.</td></tr>
        <tr><td>Inception-v3 collapse</td><td class="hint">Trained at 224px (native 299px). NORMAL recall 0.056. Cost: −4.48 pts end-to-end.</td></tr>
        <tr><td>Xception substitution</td><td class="hint">torchvision has no Xception; Inception-v3 substitute is 15 pts weaker.</td></tr>
        <tr><td>Class weights + threshold</td><td class="hint">Inverse-frequency weights + fixed 0.5 threshold. Explains AUC 0.95 vs accuracy 0.79 gap.</td></tr>
      </table>
    </div>

    <div class="card">
      <h3>Deviations from the paper (labelled)</h3>
      <p class="hint">Every design decision where the paper is silent is tagged
      in the codebase as <code>PAPER-STATED</code>, <code>INFERRED</code>,
      <code>DEFAULT</code>, or <code>UNKNOWN</code>. See
      <code>research/assumptions-and-open-questions.md</code> for the full ledger.</p>
      <table>
        <tr><th>Item</th><th>Label</th></tr>
        <tr><td>Ensemble aggregation (weighted mean of softmax probs)</td><td><span class="badge">INFERRED</span></td></tr>
        <tr><td>Xception → Inception-v3</td><td><span class="badge">INFERRED</span></td></tr>
        <tr><td>Input channels (grayscale → 3ch)</td><td><span class="badge">DEFAULT</span></td></tr>
        <tr><td>Batch size / epochs / seeds</td><td><span class="badge">DEFAULT</span></td></tr>
        <tr><td>AdaptiveAvgPool2d((7,7))</td><td><span class="badge">DEFAULT</span></td></tr>
        <tr><td>Stage 2 ensemble weights (paper doesn't publish)</td><td><span class="badge">UNKNOWN</span></td></tr>
      </table>
    </div>

    <div class="card">
      <h3>Licence</h3>
      <p>Code: MIT. Paper: CC BY 4.0. Dataset: Kermany chest X-ray corpus — see
      publisher terms. ImageNet-pretrained backbones: see torchvision weight
      licences.</p>
      <p class="hint">Educational use only. Not a medical device. Not for clinical,
      commercial, or patient-facing use without independent regulatory review.</p>
    </div>
  `
}
