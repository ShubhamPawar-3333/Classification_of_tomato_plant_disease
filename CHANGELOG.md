# CHANGELOG

## v3.0 — Training Pipeline Refactoring (2026-02-13)

### 🔴 Critical Fix: Preprocessing Mismatch
- **Before:** `ImageDataGenerator(rescale=1.0/255)` — scales pixels to [0, 1]
- **After:** `ImageDataGenerator(preprocessing_function=preprocess_input)` — EfficientNet's built-in preprocessing
- **Impact:** EfficientNet expects raw [0, 255] pixels (it normalizes internally). Using `rescale=1/255` made all pretrained features output near-zero → 18% accuracy. Fix restores expected 95%+ accuracy.
- **Files:** `model_training.py`, `model_evaluation.py`, `colab_training.ipynb`

### 🟡 Fix: Validation Generator Augmentation Bug
- **Before:** Same `ImageDataGenerator` (with rotation, zoom, shifts) used for both training and validation via `subset='training'`/`subset='validation'`
- **After:** Separate `val_datagen` with `preprocessing_function` only — no augmentation
- **Impact:** Validation metrics are now deterministic and unbiased
- **Files:** `model_training.py`

### 🟡 Fix: Dynamic EfficientNet Selection
- **Before:** `EfficientNetB4` hardcoded everywhere, but `IMAGE_SIZE=224` (optimal for B0)
- **After:** Automatic selection via `EFFICIENTNET_MAP`:
  - 224 → EfficientNetB0
  - 240 → EfficientNetB1
  - 260 → EfficientNetB2
  - 300 → EfficientNetB3
  - 380 → EfficientNetB4
- **Impact:** IMAGE_SIZE and backbone are always matched. Change `IMAGE_SIZE` in `params.yaml` and the correct backbone is automatically selected.
- **Files:** `prepare_base_model.py`, `stage_03_model_training.py`, `stage_04_model_evaluation.py`

### 🟡 Fix: Learning Rate Stabilization
- **Before:** Phase 1 LR = 0.001 (too aggressive for EfficientNet frozen features)
- **After:** Phase 1 LR = 3e-4, Phase 2 LR = 1e-4
- **Impact:** Prevents gradient explosions in early training; smoother convergence
- **Files:** `model_training.py`, `params.yaml`

### ✅ Enhancement: Pre-Training Sanity Checks
- Added assertions before training:
  - `IMAGE_SIZE` must be a valid EfficientNet input size
  - `NUM_CLASSES` from generator must match model output shape
- Added logging: selected backbone, image size, preprocessing function
- **Files:** `model_training.py`, `model_evaluation.py`

### ✅ Enhancement: Evaluation Pipeline Consistency
- `model_evaluation.py` now uses `preprocess_input` (matches training)
- Added `val_gen.reset()` before `predict()` for deterministic results
- Added sanity check logs: `class_indices`, `model.output_shape`
- Class names derived from generator instead of config hardcoding
- **Files:** `model_evaluation.py`, `stage_04_model_evaluation.py`

### ✅ Enhancement: Classification Head Simplification
- **Before:** GAP → BatchNorm → Dropout(0.3) → Dense(256) → BatchNorm → Dropout(0.2) → Dense(10)
- **After:** GAP → Dropout(0.3) → Dense(256) → Dense(10)
- **Impact:** Excessive regularization was choking gradient flow with frozen backbone
- **Files:** `prepare_base_model.py`

---

### Files Modified
| File | Changes |
|---|---|
| `components/prepare_base_model.py` | Dynamic EfficientNet selection, simplified head |
| `components/model_training.py` | preprocess_input, separate val gen, LR=3e-4, sanity checks |
| `components/model_evaluation.py` | preprocess_input, gen reset, sanity logs |
| `pipeline/stage_03_model_training.py` | Dynamic backbone MLflow logging |
| `pipeline/stage_04_model_evaluation.py` | Dynamic backbone MLflow logging |
| `config/config.yaml` | Model name → "EfficientNet" (dynamic) |
| `params.yaml` | EPOCHS → 30 |
| `research/colab_training.ipynb` | All fixes applied (v3) |
