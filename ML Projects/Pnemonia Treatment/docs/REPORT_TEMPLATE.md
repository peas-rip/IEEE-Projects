# Report Template

An IEEE-style scaffold for a short technical report or student project write-up.
The measured values from this project are **pre-filled**; everything you must
write yourself is in `[SQUARE BRACKETS]`.

Keep the section structure — it is the one your examiner expects. Delete
comment lines (`<!-- ... -->`) before submitting.

**Before you submit:** every number in this template comes from
`results/reproduction_run.json`. If you change the method, re-run the
evaluation and update both. Do not carry forward a number you did not measure.

---

> **Title**
> [A concise, specific title. Avoid restating the paper's title. Good example:]
> *An Educational Reimplementation of a Two-Stage Paediatric Pneumonia
> Ensemble: Reproduction, Failure Analysis, and a Zero-Cost Ablation*
>
> **Author(s)**
> [Your name], [Department], [Institution]
>
> **Abstract**
> [4–6 sentences in this order:]
> (1) what was reproduced and from which source paper;
> (2) the data and — importantly — the split used;
> (3) the headline result, **86.06%** accuracy and **0.9634** AUC for the
> Stage 1 detection ensemble, and **78.53%** end-to-end for the 3-class
> pipeline;
> (4) the main finding of your analysis: that a single collapsed ensemble
> member (Inception-v3, NORMAL recall 0.056) costs **4.48** percentage points
> end to end, and that removing it raises performance to **83.01%** with no
> retraining;
> (5) one sentence on why your numbers differ from the paper's 98.03% and what
> that implies for the evaluation protocol.
>
> **Index Terms**
> [Choose 4–6:] Chest X-ray, ensemble methods, transfer learning, paediatric
> pneumonia, reproducibility, evaluation protocol

---

## I. Introduction

[Para 1 — context. Paediatric pneumonia burden, why automated screening from
chest X-rays is an active area, why sensitivity matters more than specificity
in a screening setting.]

[Para 2 — the source. Introduce Godbole et al. (2024) and its contribution: two
weighted ensembles arranged as a cascade, plus two custom CNNs. State the two
reported figures — 98.03% and 91.93% — and note that the paper does not report
an end-to-end 3-class number.]

[Para 3 — the gap. State plainly that reproducing a published result requires
more than reimplementing the architecture: the split, the training budget and
the augmentation policy all move the numbers. Several of the paper's
hyper-parameters are not published at all.]

[Para 4 — your contribution. This work: (a) a faithful reimplementation of the
published architecture and weights; (b) evaluation on the official,
patient-disjoint test split; (c) per-member failure analysis; (d) an ablation
that quantifies the cost of one collapsed ensemble member; (e) an error
decomposition of the full pipeline.]

> **Contributions list** (write as bullet points)
> - [ ] A faithful reimplementation of the two-stage ensemble with the paper's
>   published Stage 1 weights (5/4/3/1).
> - [ ] Evaluation on the official Kermany test split — a harder, leakage-free
>   protocol than the paper's random 80:20 split.
> - [ ] Identification and quantification of a collapsed Stage 1 member and its
>   end-to-end cost (−4.48 points).
> - [ ] A decomposition of the 134 pipeline errors by architectural cause.
> - [ ] [Your own additional result.]

---

## II. Related Work

[Brief. 2–3 paragraphs. Cover: (1) CNN-based pneumonia classification on the
Kermany dataset generally; (2) ensemble methods in medical imaging; (3) the
reproducibility literature on evaluation protocols — the argument that
in-distribution random splits systematically overstate generalisation.
**Cite at least three sources you actually read.**]

---

## III. Method

### A. Dataset

| Property | Value |
|---|---|
| Source | Kermany paediatric chest X-ray corpus |
| Subjects | Paediatric patients, ages 1–5 |
| Total images | 5,856 |
| Classes | Normal 1,583 · Bacterial 2,766 · Viral 1,514 |
| Official split | 5,216 train / 624 test (+ a 16-image val folder, unused) |
| Used here | 4,172 train / 1,044 validation / 624 test (Stage 1); 3,100 / 775 / 390 (Stage 2) |
| Input | 224×224×3 (grayscale replicated to RGB) |
| Augmentation | On-the-fly random flip, rotation, scaling, brightness/contrast |

[State explicitly that the validation split is drawn from the training folder at
runtime (80:20, seed 42), and that it is stratified **by image**, not by patient.]

### B. Architecture

[Describe the two custom CNNs as implemented, and note that Inception-v3
substitutes for Xception because torchvision ships no Xception. The
`PAPER-STATED` / `INFERRED` / `DEFAULT` labels in
`research/assumptions-and-open-questions.md` are the right way to keep your
decisions auditable — quote the ones that matter most.]

| Component | Choice | Status |
|---|---|---|
| PneumoNet-v1 / v2 | Custom CNN, 3×3 first conv, 2-unit softmax head | PAPER-STATED (conv kernel inferred) |
| Stage 1 backbones | VGG-16, Inception-v3, PneumoNet-v1, PneumoNet-v2 | PAPER-STATED |
| Stage 2 backbones | Inception-v3 (Xception substitute), MobileNet-v2, PneumoNet-v1 | INFERRED |
| Aggregation | Weighted average of softmax probabilities | INFERRED (no equation in paper) |
| Loss | Cross-entropy with inverse-frequency class weights | INFERRED |
| Optimiser | SGD, lr 1e-4, momentum 0.9, 15 epochs | DEFAULT (schedule unstated in paper) |

### C. Ensemble weights

Stage 1 uses the weights published by the paper:

| Model | Weight |
|---|---:|
| VGG-16 | 5 |
| Inception-v3 | 4 |
| PneumoNet-v2 | 3 |
| PneumoNet-v1 | 1 |

Stage 2 weights were not published. We grid-search integers 1–6 and report
1/1/1. [If you ran Lab 5, report your validation-selected weights here and say
so.]

### D. Inference cascade

[Describe: image → Stage 1 → if P(pneumonia) ≥ 0.5, route to Stage 2 →
Normal / Bacterial / Viral. Note that Stage 1 errors propagate irrecoverably.]

---

## IV. Experimental Setup

| Item | Value |
|---|---|
| Hardware | [GPU used — e.g. NVIDIA T4, 16 GB] |
| Software | Python [version], PyTorch [version], torchvision [version] |
| Checkpoints | 7 models, 15 epochs each, final epoch saved |
| Test set | `data/real_3class/test` (624 images) and `data/real_stage2/test` (390) |
| Metrics | Accuracy, precision, recall, F1, one-vs-rest AUC, macro AP |
| Statistical treatment | [State what you did: confidence intervals, paired tests, or none] |

---

## V. Results

### A. Per-member performance

![Member accuracy](figures/fig_member_accuracy.png)

| Stage | Model | Weight | Accuracy | AUC | Paper |
|---|---|---:|---:|---:|---:|
| 1 | PneumoNet-v1 | 1 | 0.8285 | 0.9456 | 0.962 |
| 1 | PneumoNet-v2 | 3 | 0.8510 | 0.9298 | 0.968 |
| 1 | VGG-16 | 5 | 0.9119 | 0.9690 | — |
| 1 | Inception-v3 | 4 | 0.6458 | 0.7969 | — |
| 2 | PneumoNet-v1 | 1 | 0.8205 | 0.9396 | 0.873 |
| 2 | MobileNet-v2 | 1 | 0.8513 | 0.9370 | — |
| 2 | Inception-v3 (Xception substitute) | 1 | 0.7026 | 0.7845 | — |

[Comment on the Inception-v3 row. Its NORMAL recall is 0.056 — say what that
means. Do not omit the row.]

### B. Ensemble performance

| Ensemble | Accuracy | AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Stage 1 (5/4/3/1) | 0.8606 | 0.9634 | 0.8244 | 0.9872 | 0.8985 |
| Stage 2 (1/1/1, all 390 pneumonia) | 0.8949 | 0.9390 | [measure] | [measure] | [measure] |
| Stage 2 (routed 385 only) | 0.8779 | — | 0.9324 / 0.8037 | 0.8661 / 0.8973 | 0.8980 / 0.8479 |

![Stage 1 confusion matrix](figures/fig_stage1_confusion.png)

### C. End-to-end pipeline

| Metric | Value |
|---|---:|
| Accuracy | **0.7853** (490 / 624) |
| AUC (OvR) | 0.9503 |
| AP (macro) | 0.9135 |

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| NORMAL | 0.9682 | 0.6496 | 0.7775 |
| BACTERIAL | 0.9079 | 0.8554 | 0.8809 |
| VIRAL | 0.5481 | 0.8851 | 0.6770 |

![Pipeline confusion matrix](figures/fig_pipeline_confusion.png)

### D. Error decomposition

| Cause | Images | Share |
|---|---:|---:|
| Healthy lungs flagged as pneumonia | 82 | 61.2% |
| Pneumonia missed by Stage 1 | 5 | 3.7% |
| Subtype confusion on correctly detected pneumonia | 47 | 35.1% |
| **Total errors** | **134** | |

### E. Ablations

![Ablations](figures/fig_ablations.png)

| Configuration | Accuracy | Δ | Net images fixed |
|---|---:|---:|---:|
| Baseline (paper weights) | 0.7853 | — | — |
| Drop Inception-v3 from Stage 1 | **0.8301** | +4.48 | 28 |
| Stage 2 weights 2/3/5 | 0.7981 | +1.28 | 8 |
| Both | 0.8429 | +5.76 | 36 |

[Interpret, do not restate. The two repairs are exactly additive, so they act on
disjoint error sets — say so and explain why that is informative. Flag that the
2/3/5 weights are test-fitted and therefore optimistic.]

---

## VI. Discussion

[This is the section that earns marks. Write 600–900 words against these
prompts.]

1. **Accuracy is lower than the paper's; AUC is not.** Stage 1 AUC 0.9634
   against the paper's 0.97, but accuracy 86.06% against 98.03%. Explain what
   that combination implies about ranking quality versus decision rule.
2. **Protocol versus capability.** The paper's split is a stratified random
   20% of the same corpus it trained on; yours is the official patient-disjoint
   test folder. Estimate how many points of the gap you attribute to protocol,
   and state your uncertainty.
3. **Training budget.** The two custom CNNs are the largest deficit (−13.3 and
   −11.7 points) and are the only models trained from scratch under a 15-epoch,
   flat-learning-rate budget that the paper never specifies.
4. **Substitution.** Xception was replaced by Inception-v3. What can and cannot
   be concluded without an Xception implementation?
5. **The cascade is expensive.** If Stage 1 routed perfectly and Stage 2 kept
   its standalone accuracy, the pipeline would score about 93.4%. Discuss when
   a cascade architecture is worth its cost.
6. [Your own observation — one thing the numbers show that you did not expect.]

---

## VII. Threats to Validity

*Maintain this list. Mark each threat as mitigated, unmitigated, or accepted,
and say what you would do about it if you had more time.*

- **Single test set, n = 624.** Binomial standard error ≈ 1.6 points at
  p ≈ 0.785, so a 95% interval is roughly ±3.2 points. Differences smaller than
  ~3 points are not meaningful.
- **Test-fitted Stage 2 weights.** Inflates 0.8949, 0.7981 and 0.8429.
- **Train/validation split is by image, not by patient.** Validation metrics
  used for model selection are optimistic; test metrics are unaffected.
- **No calibration.** Class-weighted training plus a fixed 0.5 threshold
  explains the AUC/accuracy gap; probabilities are not risks.
- **Substituted backbone.** Xception is absent; its contribution cannot be
  isolated.
- **Aggregate-only artifacts.** Per-image predictions were not retained in the
  recorded run, so ROC curves and subgroup analyses require re-running.
- **Single institution, ages 1–5.** No external validation; any pathology other
  than bacterial or viral pneumonia is a false positive by construction.
- [Add anything your own changes introduced.]

---

## VIII. Conclusion

[Para 1 — what was achieved: a faithful reimplementation, evaluated honestly,
reproducing the architecture rather than the headline number.]

[Para 2 — the main finding: one collapsed ensemble member cost 4.48 points
end to end, invisible in the per-model table without inspecting per-class
recall; removing it reaches 83.01% with no retraining.]

[Para 3 — the lesson: reported accuracy depends on the evaluation protocol at
least as much as on the architecture, and reproductions should report the
protocol as carefully as the model.]

[Para 4 — limitations in one sentence, then future work: threshold selection on
validation, calibration, patient-level splitting, and testing whether Inception-v3
recovers at its native 299×299 resolution.]

---

## References

[1] S. Godbole, A. Kattukaran, S. Savla, V. Pradhan, P. Kanani, and D. Patil,
"Enhancing Paediatric Pneumonia Detection and Classification Using Customized
CNNs and Transfer Learning Based Ensemble Models," *Int. Res. J. Multidisc.
Technovation*, vol. 6, no. 6, pp. 38–53, 2024, doi: 10.54392/irjtmt2463.

[2] M. Kermany, K. Zhang, and M. Goldbaum, "Labeled Optical Coherence
Tomography (OCT) and Chest X-Ray Images for Classification," *Mendeley Data*,
V2, 2018, doi: 10.17632/rscbjw95tg.2.

[3] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet Classification
with Deep Convolutional Neural Networks," *Commun. ACM*, vol. 60, no. 6,
pp. 84–90, 2017.

[4] F. Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions,"
in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 1800–1807.

[5] K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for
Large-Scale Image Recognition," in *Proc. Int. Conf. Learn. Represent. (ICLR)*,
2015.

[6] [Your additional sources — add what you cite in Related Work.]

---

## Submission checklist

- [ ] Every number in the report traces to `results/reproduction_run.json` or to
      a run of your own that you can reproduce.
- [ ] The paper's 98.03% is never presented as your result.
- [ ] The split you evaluated on is stated in the Experimental Setup section.
- [ ] The Inception-v3 collapse is reported, not omitted.
- [ ] Figure captions are self-contained (readable without the body text).
- [ ] Threats to validity includes at least one thing that limits *your* work,
      not just the paper's.
- [ ] References are complete and consistent in format.
- [ ] You have cited every external idea, dataset and figure.
