# Research Paper Notes

## Paper Identification
- **Title:** Enhancing Paediatric Pneumonia Detection and Classification Using Customized CNNs and Transfer Learning Based Ensemble Models
- **Authors:** Shubham Godbole, Adit Kattukaran, Saurin Savla, Vedant Pradhan, Pratik Kanani, Deepali Patil
- **Venue:** Int. Res. J. Multidisc. Technovation, 6(6) (2024) 38–53
- **DOI:** https://doi.org/10.54392/irjtmt2463
- **Published:** 30-10-2024 (Received: 18-07-2024; Revised: 19-10-2024; Accepted: 26-10-2024)
- **License:** Creative Commons Attribution 4.0 International License (CC BY 4.0)
- **Source:** `Paper/View of Enhancing Paediatric Pneumonia Detection and Classification Using Customized CNNs and Transfer Learning Based Ensemble Models.pdf`
- **Extraction:** `research/extracted-paper.md` + `research/pages/page_*.png` (16 rendered page images; PDF was image-based, no text layer)

## Plain-Language Summary
This paper proposes two weighted ensemble models for paediatric pneumonia using chest X-rays:
1. A **pneumonia detection** ensemble (Normal vs. Pneumonia) combining VGG-16, Inception-v3, and two custom CNNs (PneumoNet-v1, PneumoNet-v2), achieving **98.03%** accuracy.
2. A **pneumonia classification** ensemble (Bacterial vs. Viral) combining Xception, MobileNet-v2, and PneumoNet-v1, achieving **91.93%** accuracy.
Custom CNN architectures PneumoNet-v1 and PneumoNet-v2 are designed to extract relevant features from paediatric chest X-rays. All models are trained on the Guangzhou Women and Children's Medical Center (Kermany) dataset with an 80:20 train/test split and images resized to 224×224.

## Problem and Motivation
- Pneumonia is a leading cause of mortality in children under five years of age (PAPER-STATED, Abstract, Introduction).
- Accurate diagnosis is paramount; chest X-rays (CXR) are the primary tool for detection.
- Existing research mainly focuses on binary Normal vs. Pneumonia classification; there is a need to further classify Pneumonia into Viral or Bacterial for better diagnostics using a weighted ensemble model (PAPER-STATED, Section 1 Introduction).
- The paper aims to (1) achieve high classification accuracy, (2) differentiate bacterial vs. viral pneumonia, and (3) explore transfer-learning-based ensemble models and custom CNN designs (PAPER-STATED, Section 3.1).

## Claimed Contributions
1. Proposed two weighted ensemble models: one for Normal vs. Pneumonia detection, another for Bacterial vs. Viral classification (PAPER-STATED, Abstract & Section 3.1).
2. Development of two custom CNN architectures: PneumoNet-v1 and PneumoNet-v2 (PAPER-STATED, Abstract & Section 3.3/3.4).
3. Ensemble of VGG-16, Inception-v3, PneumoNet-v1, PneumoNet-v2 for detection (98.03% accuracy) and Xception, MobileNet-v2, PneumoNet-v1 for classification (91.93% accuracy) (PAPER-STATED, Abstract & Section 4).
4. Data augmentation to increase training samples for bacterial (2,766 → 5,781) and viral (1,514 → 3,149) pneumonia (PAPER-STATED, Section 3.2).

## Section-by-Section Technical Summary
| Section | Content | Evidence Label |
|---------|---------|----------------|
| Abstract | Problem, models, datasets, key results (96.2% detection individual PneumoNet-v1, 98.3% & 96.8% individual accuracies, 98.03% detection ensemble, 91.93% classification ensemble) | PAPER-STATED |
| 1. Introduction | Pneumonia background, CXR, deep learning, contributions | PAPER-STATED |
| 2. Related Works | Survey of prior pneumonia detection works | PAPER-STATED |
| 2.2 Table 1 | Tabular summary of existing methods, gaps, results | PAPER-STATED |
| 3. Methodology | Proposed research: two ensembles | PAPER-STATED |
| 3.1 Proposed Research | Ensemble 1: VGG-16, Inception-v3, PneumoNet-v1, PneumoNet-v2 (detection). Ensemble 2: Xception, MobileNet-v2, PneumoNet-v1 (classification) | PAPER-STATED |
| 3.2 Data Preprocessing | Kermany dataset, 5,856 CXR images, 224×224, 80:20 split, augmentation | PAPER-STATED |
| 3.3 PneumoNet-v1 Architecture | Custom CNN details, input 224×224×3, conv layers, batch norm, SGD lr=1e-4 | PAPER-STATED |
| 3.4 PneumoNet-v2 Architecture | Custom CNN details, input 224×224×3, lower learning rate | PAPER-STATED |
| 3.5 Transfer Learning | VGG-16, VGG-19, Inception-v3, Xception, ResNet-50, ResNet-101, DenseNet-121, DenseNet-169, MobileNet-v2 | PAPER-STATED |
| 3.6 Transfer Learning (Bacterial vs. Viral) | Xception, DenseNet-121, DenseNet-169, VGG-16, VGG-19, ResNet-50, ResNet-101, MobileNet-v2, Inception-v3 | PAPER-STATED |
| 3.6.1–3.6.3 | Architecture descriptions: Xception, MobileNet-v2, DenseNet-121 | PAPER-STATED |
| 3.7 Ensemble Models | Model combinations, optimal weights via grid search/genetic algorithm | PAPER-STATED |
| Table 2 | Normal vs. Pneumonia per-model metrics | PAPER-STATED |
| Table 3 | Bacterial vs. Viral per-model metrics | PAPER-STATED |
| Table 4 | Top performing ensemble combinations | PAPER-STATED |
| 4. Results | Evaluation results, AUC, confusion matrices (Figures 5–10) | PAPER-STATED |
| 4.1 Detection (Normal vs. Pneumonia) | Detection ensemble accuracy 98.03%, AUC 0.97 | PAPER-STATED |
| 4.2 Classification (Bacterial vs. Viral) | Classification ensemble accuracy 91.93%, AUC 0.91 | PAPER-STATED |
| 5. Conclusion | Summary, future scope | PAPER-STATED |
| 6. Future Scope | Dataset limitations, GANs, multi-modal integration | PAPER-STATED |

## End-to-End Data Flow
1. Input: Chest X-ray image (anterior-posterior), resized to 224×224×3.
2. Preprocessing: normalization, data augmentation (flipping, scaling, brightness/contrast, shear).
3. Split: 80% training, 20% testing (stratified by class).
4. First ensemble model (detection): four base models (VGG-16, Inception-v3, PneumoNet-v1, PneumoNet-v2) each output a probability for Normal vs. Pneumonia. Weighted average of predictions → final detection label.
   - If "No pneumonia" → End (Normal).
   - If "Pneumonia detected" → proceed to second ensemble.
5. Second ensemble model (classification): three base models (Xception, MobileNet-v2, PneumoNet-v1) each output a probability for Bacterial vs. Viral. Weighted average → final bacterial/viral label.
6. Output: Normal / Pneumonia-Bacterial / Pneumonia-Viral.

## Equation / Notation Glossary
| Symbol | Meaning | Source |
|--------|---------|--------|
| Eq. (1) | SGD update: θ_{t+1} = θ_t − η∇_θJ(θ_t; x^(i), y^(i)) | PAPER-STATED, Section 3 |
| Eq. (2) | Mini-batch GD update: θ_{t+1} = θ_t − η (1/B) Σ ∇_θJ(...) | PAPER-STATED |
| Eq. (3) | Momentum velocity: v_{t+1} = γv_t + η∇_θJ(θ_t) | PAPER-STATED |
| Eq. (4) | Momentum param update: θ_{t+1} = θ_t − v_{t+1} | PAPER-STATED |
| Eq. (5) | AUC: ∫₀¹ TPR d(FPR) | PAPER-STATED, Section 4.1 |
| Eq. (6) | AP: Σₙ (Rₙ − Rₙ₋₁) Pₙ | PAPER-STATED, Section 4.1 |
| η | Learning rate (initialized at 10⁻⁴) | PAPER-STATED |
| γ | Momentum coefficient | PAPER-STATED |

## Datasets and Evaluation
- **Dataset:** Guangzhou Women and Children's Medical Center (Kermany Dataset) (PAPER-STATED).
- **Total:** 5,856 paediatric chest X-rays (ages 1–5 years) (PAPER-STATED).
- **Classes:** Normal (1,583), Pneumonia (4,280) → Bacterial (2,766), Viral (1,514) (PAPER-STATED).
- **Split:** 80:20 train/test (PAPER-STATED).
- **Augmented counts:** Bacterial → 5,781; Viral → 3,149 (PAPER-STATED).
- **Image size:** 224×224 pixels (standard ImageNet input size) (PAPER-STATED).
- **Metrics:** Accuracy, Precision, Recall, F1-Score, Validation Accuracy, Test Accuracy, AUC, AP (PAPER-STATED, Tables 2–4).
- **Reported results:** Detection ensemble 98.03% (PneumoNet-v1 individual 96.2%, PneumoNet-v2 96.8%); Classification ensemble 91.93% (PneumoNet-v1 87.3% bacterial/viral) (PAPER-STATED, Abstract & Section 4).

## Limitations and Threats to Validity
- Ensemble models increase computational inefficiency (PAPER-STATED, Conclusion).
- Dataset limitation: only viral or bacterial cases (no broader pneumonia types mentioned) (PAPER-STATED, Future Scope).
- Exact ensemble combination weights not fully disclosed for all combinations (INFERRED/UNKNOWN).
- Combination rule (weighted average vs. genetic algorithm stacking) not specified in pseudocode (UNKNOWN).
- No public dataset download link / credentials provided; data available from corresponding author upon reasonable request (PAPER-STATED, Data Availability).

## Source References
- Abstract: page 1 (problem, models, results).
- Section 1 Introduction: page 1 (motivation, contributions).
- Section 3.1 Proposed Research: page 4 (ensemble model definitions).
- Section 3.2 Data Preprocessing: page 4 (dataset, split, image size, augmentation).
- Section 3.3 PneumoNet-v1: pages 4–5 (architecture).
- Section 3.4 PneumoNet-v2: page 5 (architecture).
- Section 3.5 Transfer Learning: pages 5–6 (baseline models).
- Section 3.7 Ensemble Models: page 6 (combinations, weights).
- Table 2: page 7 (Normal vs. Pneumonia metrics).
- Table 3: page 8 (Bacterial vs. Viral metrics).
- Table 4: page 9 (top ensemble combinations).
- Section 4 Results: pages 9–10 (evaluation, AUC, confusion matrices).
- Section 5 Conclusion: page 12 (limitations, future scope).
- Data Availability / Funding / Competing Interests: page 14.

## Unresolved Ambiguities
1. Exact optimal weights for each ensemble combination are only partially listed (e.g., Inception-v3:4, VGG-16:5, PneumoNet-v2:3, PneumoNet-v1:1 → 98.03%); other combos lack published weights (UNKNOWN).
2. The precise ensemble aggregation formula (weighted average of probabilities vs. meta-learner stacking) is described narratively but not given as an equation (UNKNOWN).
3. No link to official code repository or pretrained weights is provided in the paper (UNKNOWN).
4. Exact train/validation/test sub-split within the 80% training portion is not specified (UNKNOWN).
5. Batch size and number of epochs are not stated (UNKNOWN).
