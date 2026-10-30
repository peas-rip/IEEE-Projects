# Lab Exercises

Ten labs built on the shipped code and checkpoints. They are ordered by
difficulty; **Labs 1–6 need no GPU and no retraining**, Lab 7–10 optionally do.

Each lab states an objective, the exact commands or code to write, and the
result you should be able to explain. Labs 1–6 are also the fastest way to make
sure your installation and data are correct.

> **Academic integrity.** This is a reference implementation with published
> numbers. Using it as a learning resource is normal engineering practice;
> submitting it as coursework that is entirely your own work is not. Follow your
> institution's policy, and cite what you reuse. Every lab below asks you to
> *change* something and *explain* the result, which is the part that has to be
> yours.

---

## Level 1 — Reproduce and reason (no code changes)

### Lab 1 — Reproduce the run end to end

**Objective.** Confirm the shipped checkpoints and your environment produce the
published numbers.

**Method.** Follow `docs/SETUP.md` Sections 2–6, then check your output against
the known-good table in `RESULTS.md` Section 11.

**What to write up.**
1. The four headline numbers you obtained and whether they matched.
2. Why it matters that `--ablate` reproduces `0.7853` for the baseline before
   it reports any variant.
3. One sentence on why `--allow-missing` results must never be reported as
   ensemble results.

**Time:** 30–60 min.

---

### Lab 2 — A collapsed ensemble member

**Objective.** Show that a single bad model can drag an entire ensemble below
the accuracy of its best member, and quantify the damage.

**Method.**

```bash
python scripts/evaluate_full_pipeline.py --ablate
```

Then open `results/reproduction_run.json` and read the per-member Stage 1 rows.

**Analysis prompts.**
1. Inception-v3 scores 0.6458 accuracy with **NORMAL recall 0.056**. State
   what that recall number means in clinical terms.
2. The paper gives Inception-v3 a weight of 4 out of 13 total votes. The best
   member, VGG-16, has 0.9119 accuracy and a weight of 5. Explain why the
   ensemble cannot "outvote" a member whose outputs are consistently wrong in
   the same direction.
3. Dropping it raises end-to-end accuracy by **+4.48 points** (28 of 624
   images). Explain why the gain is larger than the ~26-point gap between the
   two members' individual accuracies would suggest, or argue that it is
   consistent — your reasoning, with the numbers.
4. VGG-16 alone (0.9119) beats the Stage 1 ensemble (0.8606). Under what
   conditions is ensembling *expected* to help, and which of those conditions
   is violated here?

**Deliverable:** 400–600 words plus a table of per-member weights and accuracies.

---

### Lab 3 — Sensitivity versus specificity in a cascade

**Objective.** See how a high-sensitivity first stage produces a specific
downstream error signature.

**Method.** Read the confusion matrices in `docs/figures/fig_pipeline_confusion.png`
and `fig_stage1_confusion.png`. Reconstruct the 134 pipeline errors by cause
(`RESULTS.md` Section 6) yourself from the matrices — do not just copy the
table.

**Analysis prompts.**
1. Stage 1 has recall 0.9872 and precision 0.8244. Compute the false-positive
   rate on healthy lungs.
2. Of the 82 false-positive normals, 76 end up labelled VIRAL. Explain
   mechanically why Viral precision is 0.548 while its recall is 0.885.
3. If Stage 1 routed every image correctly and Stage 2 kept its standalone
   accuracy of 0.8949, what would the pipeline accuracy be? Show the arithmetic.
4. A clinician wants a screen that misses as little pneumonia as possible. Is
   the current operating point appropriate? What single change would you make,
   and what would it cost?

**Deliverable:** 500 words, with the error-decomposition table reproduced.

---

## Level 2 — Small code changes, no GPU

### Lab 4 — Weighted ensembles without retraining

**Objective.** Learn how ensemble weights behave by changing them and
re-scoring.

**Method.** Add a `--weights` flag to `scripts/evaluate_stage1_ensemble.py` that
accepts a comma-separated list in the same order as `members`, or write a small
script that loads the members, caches their probabilities once, then scores many
weight vectors cheaply (the expensive part is the forward pass — do it once).

Suggested experiments:
* Equal weights 1/1/1/1 versus the paper's 5/4/3/1.
* VGG-16 only (equivalent to a weight vector with the other three at 0).
* Sweep the Inception weight from 0 to 6 while holding the others at the paper's
  ratio, and plot accuracy against that weight.

**Expected result.** Accuracy should peak at weight 0 for Inception-v3 and
decrease monotonically as its weight grows.

**Deliverable:** the plot, plus a statement of the weight that maximises
accuracy and a discussion of what that says about weight search in general.

---

### Lab 5 — Data leakage in weight tuning

**Objective.** Feel the difference between fitting on validation and fitting on
test.

**Method.** `evaluate_stage2_ensemble.py` grid-searches weights over 1–6 and
reports accuracy **on the same test set** (0.8949). Restructure it so the
weights are selected on the validation split and then applied unchanged to the
test set, and compare:

| Config | Accuracy |
|---|---:|
| Grid-searched **on test** (shipped) | 0.8949 |
| Grid-searched **on validation** (your run) | *measure it* |
| Equal weights 1/1/1 (no fitting at all) | *measure it* |

You will need per-image probabilities on both splits — Lab 6 builds the tool.

**Analysis prompts.**
1. Quantify the optimism: `test-tuned − validation-tuned`.
2. The ablation "Stage 2 weights 2/3/5" gains +1.28 points but is test-fitted.
   Estimate how much of that gain you expect to survive validation fitting, and
   justify the estimate.
3. State the general rule you would apply before shipping any tunable
   hyper-parameter in a pipeline like this one.

**Deliverable:** table of the three accuracies plus a 300-word discussion.

---

### Lab 6 — Per-image prediction dump (implementation)

**Objective.** Turn aggregate metrics into an inspectable artifact.

**Method.** Write `scripts/dump_predictions.py` that loads a set of members,
runs them over a loader once, and writes a CSV with one row per image:

```
image,true_label,p_normal,p_pneumonia            # stage 1
image,true_label,p_bacterial,p_viral             # stage 2
image,true_label,predicted,p_normal,p_bacterial,p_viral   # pipeline
```

Reuse `pneumonia.evaluate.predict_probs` for the forward pass and
`pneumonia.checkpoints.checkpoint_path` for the paths. Do not duplicate the
model-loading code from the evaluation scripts.

**What to build on top of it.**
* Rank the 624 test images by `p_normal` and list the 20 with the highest
  predicted pneumonia probability among the true NORMAL class.
* Compute the calibration error: group predictions into 10 probability bins and
  compare mean predicted probability with observed frequency.
* Pick a decision threshold on the validation split that maximises balanced
  accuracy, then report the pipeline accuracy at that threshold instead of 0.5.

**Deliverable:** the CSV, the calibration table, and the threshold you chose and
why. This lab directly attacks the AUC 0.95 / accuracy 0.785 gap.

---

### Lab 7 — The cascade cost

**Objective.** Isolate how much accuracy the two-stage architecture itself
costs.

**Method.** Train (or reuse) a single 3-class model on `data/real_3class/train`
— `configs/default.yaml` with `num_classes: 3` and three class folders is enough.
Compare it on the same 624 test images against the pipeline.

**Analysis prompts.**
1. Report both accuracies and both AUCs on the *same* test set.
2. Argue for or against: "a cascade is only worth it when the first stage is
   more accurate than it is here."
3. Note that a 3-class model cannot be selectively reweighted at inference
   time, while the cascade can (Lab 4). What does that buy you?

**Requires:** GPU (optional). **Time:** 1–2 h including training.

---

## Level 3 — Optional GPU work

### Lab 8 — Fix the collapsed member

**Objective.** Test whether Inception-v3's collapse was a training
configuration bug rather than an architecture problem.

**Method.** Inception-v3 is designed for 299×299 input and was trained here at
224×224 with the same learning rate as the pretrained backbone. Change one
variable at a time:
1. Train at 299×299 (`data.image_size: 299` in `configs/stage1_inception.yaml`).
2. Or use a discriminative learning rate (lower for the backbone than the head).
3. Or both.

Evaluate with `evaluate_stage1_ensemble.py` and re-run the ablation.

**Expected result.** If NORMAL recall recovers from 0.056 to a useful range,
the collapse was configuration-induced, and the 4-model ensemble should
reclaim the +4.48 points. If it does not, the 3-model variant is the better
design and that is a defensible result either way.

**Deliverable:** per-member table before/after, the ablation rerun, and your
conclusion. Note that `evaluate_stage1_ensemble.py` scores a fixed 224 px
loader, so either change the loader or note the input-size mismatch explicitly.

---

### Lab 9 — Reproduce the paper's protocol

**Objective.** Determine how much of the 12-point Stage 1 gap is the test
protocol rather than the models.

**Method.** The paper used a stratified 80:20 random split over all 5,856
images and reported 98.03%. Build that split, train **one** model twice (cheapest
option: PneumoNet-v2), and evaluate each on **both** test sets:

| Trained on | Evaluated on | Accuracy |
|---|---|---|
| Paper protocol | Paper's random holdout | *measure* |
| Paper protocol | Official 624-image test folder | *measure* |
| Official split | Official 624-image test folder | 0.8510 (shipped) |

**What it settles.** The first row tells you what the paper's number is worth;
the second tells you whether the protocol or the training is the dominant
cause. This is the single most informative experiment in the whole project.

**Requires:** GPU. **Time:** ~1 h.

---

### Lab 10 — Make the pipeline usable on one image

**Objective.** Ship a single command that takes a JPEG and returns Normal /
Bacterial / Viral with probabilities from the full ensemble.

**Method.** `pneumonia infer` currently scores exactly one model. Extend it, or
write `scripts/predict_one.py`, that:
1. Loads the Stage 1 members and blends with the paper weights 5/4/3/1.
2. If P(max ≥ threshold) indicates pneumonia, loads the Stage 2 members and
   blends them.
3. Prints the three-class probabilities and the route taken.
4. Accepts `--stage1-weights`, `--stage2-weights` and `--threshold` so the
   Lab 6 threshold can be used.

**Deliverable:** the command, a demo on three images (one per class), and a
timing comparison between loading all seven models once versus per request.
Explain why this is not a deployable service and what would have to change
(ONNX/TorchScript export, batching, a model server, calibration) — and state
that it would not be a medical device regardless.

**Requires:** GPU optional. **Time:** 2–3 h.

---

## Suggested report topics

If you are writing up one of these for a course, the strongest framing is:

| Topic | Why it reads well |
|---|---|
| Lab 2 + Lab 4 | One failure, quantified, with a control experiment |
| Lab 6 | Turns a benchmark into a diagnostic tool |
| Lab 8 | A hypothesis-driven fix with a falsifiable outcome |
| Lab 9 | Directly addresses an external claim; good literature-adjacent work |

All of them are honest about what was measured. That is the difference between
a report that survives a viva and one that does not.
