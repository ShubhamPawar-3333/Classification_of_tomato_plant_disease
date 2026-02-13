"""
Tomato Disease Advisory System — Gradio Web Application

Upload a tomato leaf photo → get disease diagnosis, GradCAM++ heatmap,
severity estimation, and AI-powered treatment advisory.

Run: python app.py
"""
import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

import gradio as gr

# ── Project imports ──────────────────────────────────────────────
from tomato_disease_advisor.config import ConfigurationManager
from tomato_disease_advisor.components.explainer import GradCAMExplainer
from tomato_disease_advisor.components.severity import SeverityEstimator
from tomato_disease_advisor.rag.retriever import KnowledgeRetriever
from tomato_disease_advisor.rag.advisor import TreatmentAdvisor
from tomato_disease_advisor.feedback.collector import FeedbackCollector


# ── Global state (loaded once) ──────────────────────────────────
MODEL = None
CONFIG = None
EXPLAINER = None
SEVERITY_ESTIMATOR = None
RETRIEVER = None
ADVISOR = None
FEEDBACK = FeedbackCollector()

# Store last prediction for feedback
LAST_PREDICTION = {}

# ── Colour palette ──────────────────────────────────────────────
SEVERITY_COLORS = {
    "healthy": "#22c55e",   # green
    "mild": "#eab308",      # yellow
    "moderate": "#f97316",  # orange
    "severe": "#ef4444",    # red
}

CONFIDENCE_COLORS = {
    "confident": "#22c55e",
    "warning": "#f97316",
    "abstain": "#ef4444",
}


# ═══════════════════════════════════════════════════════════════
#  INITIALIZATION
# ═══════════════════════════════════════════════════════════════

def load_system():
    """Load model and all components (called once at startup)."""
    global MODEL, CONFIG, EXPLAINER, SEVERITY_ESTIMATOR, RETRIEVER, ADVISOR

    print("\n🍅 Loading Tomato Disease Advisory System...")

    config_mgr = ConfigurationManager()

    # Load model
    training_cfg = config_mgr.config.training
    model_path = training_cfg.trained_model_path
    print(f"   Loading model: {model_path}")
    MODEL = tf.keras.models.load_model(str(model_path))

    # Store full config
    CONFIG = {
        "class_names": list(config_mgr.config.class_names),
        "image_size": config_mgr.params.IMAGE_SIZE,
    }

    # GradCAM++ explainer
    gradcam_cfg = config_mgr.get_gradcam_config()
    EXPLAINER = GradCAMExplainer(gradcam_cfg)

    # Severity estimator
    severity_cfg = config_mgr.get_severity_config()
    SEVERITY_ESTIMATOR = SeverityEstimator(severity_cfg)

    # Confidence thresholds
    confidence_cfg = config_mgr.get_confidence_config()
    CONFIG["abstention_threshold"] = confidence_cfg.abstention_threshold
    CONFIG["warning_threshold"] = confidence_cfg.warning_threshold

    # Knowledge retriever (only if index exists)
    vectorstore_cfg = config_mgr.get_vectorstore_config()
    index_dir = Path(vectorstore_cfg.index_path)
    if (index_dir / "index.faiss").exists():
        RETRIEVER = KnowledgeRetriever(
            index_path=str(index_dir),
            embedding_model=vectorstore_cfg.embedding_model,
        )
        print("   ✅ Knowledge retriever loaded")
    else:
        RETRIEVER = None
        print("   ⚠️  No FAISS index found — advisory will use fallback mode")

    # LLM advisor
    rag_cfg = config_mgr.get_rag_config()
    ADVISOR = TreatmentAdvisor(rag_cfg)

    print("   ✅ System ready!\n")


# ═══════════════════════════════════════════════════════════════
#  CORE PREDICTION
# ═══════════════════════════════════════════════════════════════

def predict(image):
    """
    Main prediction function called by Gradio.

    Args:
        image: PIL Image or numpy array from Gradio

    Returns:
        Tuple of outputs for each Gradio component
    """
    global LAST_PREDICTION

    if image is None:
        return (
            "⚠️ Please upload an image.",
            None, None, None,
            "Upload an image to get started.",
        )

    # ── Preprocess ──
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    img_size = CONFIG["image_size"]

    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image).convert("RGB")
    else:
        pil_img = image.convert("RGB")

    pil_img = pil_img.resize((img_size, img_size))
    original_array = np.array(pil_img)
    input_array = preprocess_input(original_array.copy().astype(np.float32))
    input_batch = np.expand_dims(input_array, axis=0)

    # ── Classify ──
    predictions = MODEL.predict(input_batch, verbose=0)
    class_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_idx])
    class_name = CONFIG["class_names"][class_idx]

    # Pretty class name
    display_name = class_name.replace("_", " ").replace("Tomato ", "").strip()

    # ── Confidence assessment ──
    if confidence < CONFIG["abstention_threshold"]:
        conf_level = "abstain"
        conf_msg = "🚫 Very low confidence — please consult an expert"
    elif confidence < CONFIG["warning_threshold"]:
        conf_level = "warning"
        conf_msg = "⚠️ Moderate confidence — consider a second opinion"
    else:
        conf_level = "confident"
        conf_msg = "✅ High confidence"

    # ── GradCAM++ ──
    heatmap = EXPLAINER.generate_heatmap(MODEL, input_batch, class_idx)
    overlay = EXPLAINER.overlay_heatmap(original_array, heatmap)

    # ── Severity ──
    severity = SEVERITY_ESTIMATOR.estimate_severity(heatmap, class_name)
    sev_level = severity["severity_level"]
    sev_pct = severity["affected_area_pct"]
    sev_color = SEVERITY_COLORS.get(sev_level, "#888")

    # ── All probabilities (top 5) ──
    top_indices = np.argsort(predictions[0])[::-1][:5]
    prob_dict = {
        CONFIG["class_names"][i].replace("_", " "): float(predictions[0][i])
        for i in top_indices
    }

    # ── Build classification result markdown ──
    conf_color = CONFIDENCE_COLORS.get(conf_level, "#888")
    result_md = f"""
## 🔬 Diagnosis: {display_name}

| | |
|---|---|
| **Confidence** | <span style="color:{conf_color}; font-weight:bold">{confidence:.1%}</span> {conf_msg} |
| **Severity** | <span style="color:{sev_color}; font-weight:bold">{sev_level.upper()}</span> ({sev_pct:.1f}% affected area) |
| **Description** | {severity["description"]} |
"""

    # ── Advisory ──
    advisory_md = _generate_advisory(
        class_name, confidence, sev_level, sev_pct
    )

    # ── Store for feedback ──
    LAST_PREDICTION = {
        "class_name": class_name,
        "confidence": confidence,
        "severity_level": sev_level,
    }

    return (
        result_md,          # classification result
        overlay,            # GradCAM overlay image
        prob_dict,          # probability label
        advisory_md,        # treatment advisory
    )


def _generate_advisory(
    class_name: str,
    confidence: float,
    severity_level: str,
    affected_area: float,
) -> str:
    """Generate treatment advisory using RAG or fallback."""

    if RETRIEVER is None:
        # Fallback: read treatment file directly
        return _fallback_advisory(class_name, severity_level)

    try:
        # Retrieve relevant knowledge
        knowledge = RETRIEVER.retrieve_for_disease(class_name, top_k=3)

        # Generate LLM advisory
        result = ADVISOR.generate_advisory(
            disease_name=class_name,
            confidence=confidence,
            severity_level=severity_level,
            affected_area=affected_area,
            disease_chunks=knowledge.get("disease_info", []),
            treatment_chunks=knowledge.get("treatment_info", []),
        )

        return result.get("advisory", _fallback_advisory(class_name, severity_level))

    except Exception as e:
        print(f"[App] Advisory error: {e}")
        return _fallback_advisory(class_name, severity_level)


def _fallback_advisory(class_name: str, severity_level: str) -> str:
    """Read treatment markdown directly as fallback."""
    # Map class name to treatment file
    disease_key = class_name.lower()
    for prefix in ["tomato_", "tomato__"]:
        if disease_key.startswith(prefix):
            disease_key = disease_key[len(prefix):]
            break

    key_map = {
        "spider_mites_two_spotted_spider_mite": "spider_mites",
        "spider_mites_two_s": "spider_mites",
        "tomato_yellowleaf__curl_virus": "yellow_leaf_curl_virus",
        "yellowleaf__curl_virus": "yellow_leaf_curl_virus",
        "tomato_mosaic_virus": "mosaic_virus",
        "target_spot": "target_spot",
    }
    disease_key = key_map.get(disease_key, disease_key)

    treatment_path = Path("knowledge/treatments") / f"{disease_key}.md"
    if treatment_path.exists():
        content = treatment_path.read_text(encoding="utf-8")
        return f"*📋 Knowledge base advisory (AI advisor unavailable):*\n\n{content}"

    return (
        f"*No specific treatment information available for {class_name}. "
        f"Please consult your local agricultural extension officer.*"
    )


# ═══════════════════════════════════════════════════════════════
#  FEEDBACK
# ═══════════════════════════════════════════════════════════════

def submit_feedback(is_correct: bool, comment: str) -> str:
    """Submit user feedback on the prediction."""
    if not LAST_PREDICTION:
        return "⚠️ No prediction to provide feedback on."

    FEEDBACK.save_feedback(
        image_name="uploaded_image",
        predicted_class=LAST_PREDICTION.get("class_name", "unknown"),
        confidence=LAST_PREDICTION.get("confidence", 0),
        severity_level=LAST_PREDICTION.get("severity_level", "unknown"),
        is_correct=is_correct,
        user_comment=comment,
    )

    emoji = "✅" if is_correct else "❌"
    return f"{emoji} Thank you for your feedback! This helps improve the model."


# ═══════════════════════════════════════════════════════════════
#  GRADIO INTERFACE
# ═══════════════════════════════════════════════════════════════

def build_app() -> gr.Blocks:
    """Build the Gradio interface."""

    custom_css = """
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #16a34a 0%, #166534 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2em;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1em;
        margin-top: 0;
    }
    .severity-box {
        border-radius: 12px;
        padding: 16px;
    }
    .feedback-section {
        border-top: 1px solid #e5e7eb;
        padding-top: 16px;
        margin-top: 16px;
    }
    """

    with gr.Blocks(
        title="🍅 Tomato Disease Advisor",
        theme=gr.themes.Soft(
            primary_hue="green",
            secondary_hue="orange",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=custom_css,
    ) as app:

        # ── Header ──
        gr.HTML("""
            <div style="text-align:center; padding: 20px 0 10px 0;">
                <h1 class="main-title">🍅 Tomato Disease Advisory System</h1>
                <p class="subtitle">
                    Upload a tomato leaf photo for instant AI-powered disease diagnosis,
                    visual explanation, severity assessment, and treatment advice.
                </p>
            </div>
        """)

        with gr.Row():
            # ── Left Column: Input ──
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="📸 Upload Leaf Image",
                    type="pil",
                    height=350,
                )
                analyze_btn = gr.Button(
                    "🔍 Analyze Disease",
                    variant="primary",
                    size="lg",
                )
                gr.Examples(
                    examples=[],  # User can add sample images here
                    inputs=image_input,
                    label="Sample Images",
                )

            # ── Right Column: Results ──
            with gr.Column(scale=1):
                result_output = gr.Markdown(
                    value="*Upload an image and click 'Analyze Disease' to begin.*",
                    label="Diagnosis Result",
                )
                prob_output = gr.Label(
                    label="📊 Class Probabilities (Top 5)",
                    num_top_classes=5,
                )

        # ── Visual Explanation ──
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔥 GradCAM++ Explanation")
                gradcam_output = gr.Image(
                    label="Disease Heatmap Overlay",
                    type="numpy",
                    height=350,
                )
                gr.Markdown(
                    "*Highlighted regions show where the model detected "
                    "disease symptoms.*",
                    elem_classes=["subtitle"],
                )

            with gr.Column():
                gr.Markdown("### 💊 Treatment Advisory")
                advisory_output = gr.Markdown(
                    value="*Treatment advice will appear here after analysis.*",
                )

        # ── Feedback Section ──
        with gr.Accordion("📝 Was this diagnosis helpful?", open=False):
            with gr.Row():
                correct_btn = gr.Button("👍 Correct", variant="primary")
                incorrect_btn = gr.Button("👎 Incorrect", variant="secondary")
            feedback_comment = gr.Textbox(
                label="Optional comment",
                placeholder="Any additional feedback...",
                lines=2,
            )
            feedback_status = gr.Markdown("")

        # ── Footer ──
        gr.HTML("""
            <div style="text-align:center; padding:20px; color:#9ca3af; font-size:0.85em;">
                Built with EfficientNet (99% accuracy) · GradCAM++ · FAISS + SciBERT · Groq LLM
                <br>
                <a href="https://github.com/ShubhamPawar-3333/Classification_of_tomato_plant_disease"
                   target="_blank" style="color:#16a34a;">GitHub Repository</a>
            </div>
        """)

        # ── Event handlers ──
        analyze_btn.click(
            fn=predict,
            inputs=[image_input],
            outputs=[
                result_output,
                gradcam_output,
                prob_output,
                advisory_output,
            ],
        )

        correct_btn.click(
            fn=lambda c: submit_feedback(True, c),
            inputs=[feedback_comment],
            outputs=[feedback_status],
        )
        incorrect_btn.click(
            fn=lambda c: submit_feedback(False, c),
            inputs=[feedback_comment],
            outputs=[feedback_status],
        )

    return app


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    load_system()
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
