# Assumptions and Open Questions

## Assumptions (engineering defaults, labeled INFERRED unless paper-stated)

1. **Ensemble aggregation is a weighted average of class probabilities.**  
   The paper describes "integration of the output provided by three models" and "calculating the optimal weights" but does not give an explicit aggregation equation. Default: weighted average of softmax probabilities. Label: INFERRED.

2. **Transfer-learning models are fine-tuned (not frozen feature extractors) on the Kermany dataset.**  
   Paper states all models were trained on the Kermany dataset. Default: standard fine-tuning with a new classification head. Label: INFERRED.

3. **PneumoNet-v1/v2 output head is a 2-unit Dense layer with Softmax and Categorical Crossentropy loss.**  
   Paper states "final layer is a dense layer with 2 units... Softmax activation... Categorical Crossentropy". Label: PAPER-STATED.

4. **Input images are converted to 3-channel RGB (224x224x3).**  
   Paper specifies input size 224x224x3. Chest X-rays are typically grayscale; default is to replicate the single channel to 3 channels to match ImageNet-pretrained backbones. Label: INFERRED.

5. **Binary detection (Normal vs. Pneumonia) and binary classification (Bacterial vs. Viral) are treated as 2-class problems with softmax over 2 units.**  
   Label: INFERRED.

6. **Deterministic seeding for reproducibility.**  
   Paper does not specify seeds. Default: set Python/numpy/torch seeds to a fixed value (e.g., 42) where supported. Label: DEFAULT.

## Open Questions (UNKNOWN / needs resolution)

1. **Exact optimal ensemble weights for every combination.**  
   Paper lists weights only for the best detection combo (Inception-v3: 4, VGG-16: 5, PneumoNet-v2: 3, PneumoNet-v1: 1 → 98.03%) and notes weight ranges 1–15 / 1–20 were searched. Weights for the classification ensemble and other combinations are not published.  
   Decision needed: use published weights where available, otherwise perform a small grid search on validation data to estimate weights, clearly labeled as an INFERRED/DEFAULT choice.

2. **Precise ensemble combination rule.**  
   Narrative mentions averaging probability values and a genetic algorithm for weight optimization, but no pseudocode or closed-form formula.  
   Decision needed: implement weighted average of probabilities; document as INFERRED.

3. **Official author code repository.**  
   The paper does not provide a code URL. Need to search whether an official repo exists; if absent, this is a clean reimplementation. Label: UNKNOWN.

4. **Pretrained weights.**  
   No pretrained weights referenced. Transfer-learning backbones will use ImageNet pretrained weights from torchvision (standard). Label: DEFAULT (engineering choice, not paper-stated).

5. **Batch size and number of epochs.**  
   Not stated in the paper. Defaults: batch_size=32, epochs=10 (smoke test); configurable via config file. Label: DEFAULT.

6. **Train/validation sub-split.**  
   Paper specifies 80:20 train/test but no validation split within training. Default: hold out 10% of training as validation for weight selection / early stopping. Label: DEFAULT.

7. **Dataset access.**  
   Data available from the corresponding author upon reasonable request (paper Data Availability section). No public download link. The project will include instructions to obtain the dataset and a synthetic fixture for smoke testing. Label: UNKNOWN (real data not obtainable without author contact).

8. **Grayscale vs. RGB input for custom CNNs.**  
   Paper says input size 224x224x3 but chest X-rays are grayscale. Default: convert grayscale to 3 identical channels. Label: INFERRED.

## Decisions Log (to be filled during implementation)

| # | Decision | Rationale | Status |
|---|----------|-----------|--------|
| 1 | Framework: PyTorch | Available in environment; widely used; torchvision provides transfer-learning backbones | Proposed |
| 2 | Ensemble aggregation: weighted average of softmax probabilities | Matches paper narrative; simple and interpretable | Proposed |
| 3 | Grayscale -> 3-channel replication | Required for ImageNet-pretrained backbones and 224x224x3 input spec | Proposed |
| 4 | batch_size=32, epochs=10 (configurable) | Paper silent; engineering defaults | Proposed |
| 5 | Real dataset optional; synthetic fixture for smoke tests | Real data gated behind author request | Proposed |
