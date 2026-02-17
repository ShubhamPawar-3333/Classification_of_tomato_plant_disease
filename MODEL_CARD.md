---
language: en
license: mit
tags:
  - image-classification
  - tensorflow
  - keras
  - efficientnet
  - agriculture
  - plant-disease
  - gradcam
  - rag
datasets:
  - PlantVillage
metrics:
  - accuracy
  - f1
  - precision
  - recall
---

# 🍅 Agricultural Disease Diagnosis and Advisory System — Tomato Crops

## Model Description

Fine-tuned **EfficientNetB0** for classifying 10 tomato leaf diseases from the
PlantVillage dataset. Integrated into a full advisory system with GradCAM++
explainability, severity estimation, and RAG-powered treatment advice.

| Detail | Value |
|--------|-------|
| Architecture | EfficientNetB0 + custom classification head |
| Framework | TensorFlow / Keras 2.15+ |
| Parameters | ~4.3M (backbone) + ~2.6K (head) |
| Input size | 224 × 224 × 3 |
| Output | 10-class softmax |
| Training | Two-phase transfer learning |

## Intended Use

- Assist farmers and agronomists in early disease identification
- **Not** a replacement for professional plant pathology diagnosis
- Intended for tomato crops only

## Training Data

**PlantVillage** — 16,011 tomato leaf images across 10 classes:

| # | Class | Samples |
|---|-------|---------|
| 1 | Bacterial Spot | 2,127 |
| 2 | Early Blight | 1,000 |
| 3 | Late Blight | 1,909 |
| 4 | Leaf Mold | 952 |
| 5 | Septoria Leaf Spot | 1,771 |
| 6 | Spider Mites (Two-spotted) | 1,676 |
| 7 | Target Spot | 1,404 |
| 8 | Yellow Leaf Curl Virus | 3,209 |
| 9 | Mosaic Virus | 373 |
| 10 | Healthy | 1,591 |

**Preprocessing:** `tf.keras.applications.efficientnet.preprocess_input`
(scales to [-1, 1] range)

**Augmentation:** rotation (±25°), width/height shift (20%), shear (15%),
zoom (20%), horizontal flip, brightness (±15%)

## Training Procedure

### Phase 1 — Head Only (5 epochs)
- Frozen backbone, train classification head only
- LR: 3×10⁻⁴, optimizer: Adam
- Purpose: warm up the randomly initialized head

### Phase 2 — Full Fine-Tuning (25 epochs)
- Unfreeze all layers
- LR: 1×10⁻⁴ with cosine decay
- ReduceLROnPlateau (patience=3, factor=0.5)
- EarlyStopping (patience=5, restore best weights)

## Evaluation Results

| Metric | Score |
|--------|-------|
| **Accuracy** | **99.03%** |
| F1 (weighted) | 99.03% |
| Precision (weighted) | 99.04% |
| Recall (weighted) | 99.03% |

Evaluated on a held-out 20% test split with no data leakage.

## Explainability

**GradCAM++** heatmaps highlight the regions the model focuses on:
- Second-order gradients for better localization
- Automatic fallback to last Conv2D layer if configured layer not found
- Heatmap overlay at 60% opacity for visual clarity

## Severity Estimation

Derived from GradCAM++ activation maps:
- **Healthy:** <5% affected area
- **Mild:** 5–20%
- **Moderate:** 20–50%
- **Severe:** >50%

## Limitations

- Trained only on PlantVillage lab images (white background) — field images
  with complex backgrounds may reduce accuracy
- Single-disease-per-image assumption — co-infections not handled
- Does not detect nutritional deficiencies or abiotic stress
- 10 classes only — other tomato diseases not covered

## Ethical Considerations

- **Not a replacement** for professional diagnosis — always recommend expert
  consultation for severe cases
- Confidence thresholding (≤70% triggers "uncertain" warning)
- Treatment suggestions sourced from published agricultural research

## How to Use

```python
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("artifacts/training/model.keras")
img = Image.open("leaf.jpg").resize((224, 224))
x = tf.keras.applications.efficientnet.preprocess_input(
    np.array(img)[np.newaxis, ...]
)
predictions = model.predict(x)
```

## Citation

```bibtex
@article{hughes2015plantvillage,
  title={An open access repository of images on plant health to enable
         the development of mobile disease diagnostics},
  author={Hughes, David and Salathé, Marcel},
  journal={arXiv preprint arXiv:1511.08060},
  year={2015}
}
```
