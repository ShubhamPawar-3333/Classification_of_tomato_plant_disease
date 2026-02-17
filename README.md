# 🍅 Tomato Disease Advisory System

> **AI-powered tomato leaf disease diagnosis** with GradCAM++ explainability, severity estimation, and RAG-powered treatment advisories.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-green.svg)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What It Does

Upload a tomato leaf photo → get:

1. **Disease Classification** — EfficientNetB0 identifies 1 of 10 diseases (99.03% accuracy)
2. **GradCAM++ Heatmap** — visual explanation of _where_ the model sees disease
3. **Severity Estimation** — mild / moderate / severe based on affected leaf area
4. **AI Treatment Advisory** — RAG-powered advice using Groq LLM + agricultural knowledge base

---

## 🏗️ Architecture

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  Tomato Leaf │────▶│ EfficientNetB0│────▶│ Classification│
│   Image      │     │  (224×224)    │     │  (10 classes) │
└──────────────┘     └───────┬───────┘     └──────┬───────┘
                             │                     │
                    ┌────────▼────────┐    ┌───────▼───────┐
                    │   GradCAM++     │    │   Severity    │
                    │   Heatmap       │    │   Estimation  │
                    └────────┬────────┘    └───────┬───────┘
                             │                     │
                    ┌────────▼─────────────────────▼───────┐
                    │           FAISS Vector Store          │
                    │    (20 knowledge docs embedded)       │
                    └────────────────┬─────────────────────┘
                                     │
                    ┌────────────────▼─────────────────────┐
                    │         Groq LLM (Llama 3.3 70B)     │
                    │     Treatment Advisory Generation     │
                    └──────────────────────────────────────┘
```

---

## 🔬 Tech Stack

| Layer | Technology |
|-------|-----------|
| **CV Model** | EfficientNetB0 (two-phase transfer learning) |
| **Explainability** | GradCAM++ with second-order gradients |
| **Severity** | Heatmap activation analysis |
| **Vector DB** | FAISS (all-MiniLM-L6-v2 embeddings) |
| **LLM** | Groq (Llama 3.3 70B Versatile) |
| **Pipeline** | DVC (5 reproducible stages) |
| **Experiment Tracking** | MLflow |
| **UI** | Gradio |
| **CI/CD** | GitHub Actions → HuggingFace Spaces |
| **Containerization** | Docker |

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **99.03%** |
| F1 (weighted) | 99.03% |
| Precision (weighted) | 99.04% |
| Recall (weighted) | 99.03% |

**10 Classes:** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/ShubhamPawar-3333/Classification_of_tomato_plant_disease.git
cd Classification_of_tomato_plant_disease
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -r requirements.txt
pip install -e .
```

### 2. Set Environment Variables

```bash
# Required for AI advisory (get free key at https://console.groq.com/)
set GROQ_API_KEY=your_key_here
```

### 3. Run the App

```bash
python app.py
# Open http://localhost:7860
```

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## 📁 Project Structure

```
├── app.py                          # Gradio web application
├── main.py                         # Full pipeline runner
├── Dockerfile                      # HuggingFace Spaces deployment
├── dvc.yaml                        # Pipeline definition (5 stages)
├── config/config.yaml              # Paths & model configuration
├── params.yaml                     # Hyperparameters
│
├── src/tomato_disease_advisor/
│   ├── components/
│   │   ├── data_ingestion.py       # PlantVillage dataset download
│   │   ├── prepare_base_model.py   # EfficientNet setup + custom head
│   │   ├── model_training.py       # Two-phase transfer learning
│   │   ├── model_evaluation.py     # Metrics + confusion matrix
│   │   ├── explainer.py            # GradCAM++ heatmap generation
│   │   └── severity.py             # Disease severity estimation
│   ├── rag/
│   │   ├── store.py                # FAISS index builder
│   │   ├── retriever.py            # Semantic knowledge retrieval
│   │   └── advisor.py              # Groq LLM treatment advisor
│   ├── pipeline/                   # DVC pipeline stages (01–05)
│   ├── config/configuration.py     # Typed config management
│   ├── entity/config_entity.py     # Dataclass definitions
│   └── feedback/collector.py       # User feedback JSONL logger
│
├── knowledge/
│   ├── diseases/                   # 10 disease info markdown files
│   └── treatments/                 # 10 treatment guide markdown files
│
├── artifacts/                      # DVC-tracked outputs
│   ├── training/model.keras        # Trained model (~41 MB)
│   ├── vectorstore/                # FAISS index + metadata
│   └── evaluations/                # Confusion matrix, training curves
│
├── tests/                          # pytest suite (60 tests)
├── .github/workflows/              # CI/CD (sync-to-hf, model upload)
├── MODEL_CARD.md                   # HuggingFace model card
└── docs/                           # Architecture documentation
```

---

## 🔄 DVC Pipeline

```bash
dvc repro  # Run all stages
```

| Stage | Command | Outputs |
|-------|---------|---------|
| 1. Data Ingestion | Download PlantVillage (16K images) | `artifacts/data_ingestion/dataset` |
| 2. Prepare Base Model | EfficientNetB0 + classification head | `artifacts/prepare_base_model/` |
| 3. Model Training | Two-phase transfer learning (30 epochs) | `artifacts/training/model.keras` |
| 4. Model Evaluation | Test metrics + confusion matrix | `scores.json` |
| 5. Build Vectorstore | FAISS index from knowledge base | `artifacts/vectorstore/` |

See [docs/PIPELINE_FLOW.md](docs/PIPELINE_FLOW.md) for detailed pipeline documentation.

---

## 🧠 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **EfficientNetB0** over ResNet/VGG | Best accuracy-per-FLOP; compound scaling; only 4.3M params |
| **Two-phase training** | Phase 1 warms head (5 epochs, high LR) → Phase 2 fine-tunes all layers (25 epochs, low LR) |
| **GradCAM++** over GradCAM | Second-order gradients → better localization for small lesions |
| **FAISS** over ChromaDB | Faster search, no server dependency, single-file persistence |
| **Groq** over OpenAI | Free tier, ultra-fast inference (~200ms), Llama 3.3 70B |
| **Gradio** over Flask | Built-in image upload, ML-first components, HF Spaces native |

See [docs/PROJECT_FLOW.md](docs/PROJECT_FLOW.md) for full architecture documentation.

---

## 🐳 Docker

```bash
docker build -t tomato-app .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key tomato-app
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **PlantVillage Dataset** — Hughes & Salathé (2015)
- **EfficientNet** — Tan & Le (2019)
- **GradCAM++** — Chattopadhyay et al. (2018)
- **Groq** — Free LLM inference API
