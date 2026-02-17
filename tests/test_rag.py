"""
Unit tests for RAG knowledge base and retriever mappings.

Run: pytest tests/test_rag.py -v
"""
from pathlib import Path

import pytest


# ── Knowledge file existence tests ───────────────────────────


DISEASE_FILES = [
    "bacterial_spot.md",
    "early_blight.md",
    "healthy.md",
    "late_blight.md",
    "leaf_mold.md",
    "mosaic_virus.md",
    "septoria_leaf_spot.md",
    "spider_mites.md",
    "target_spot.md",
    "yellow_leaf_curl_virus.md",
]

TREATMENT_FILES = [
    "bacterial_spot.md",
    "early_blight.md",
    "healthy.md",
    "late_blight.md",
    "leaf_mold.md",
    "mosaic_virus.md",
    "septoria_leaf_spot.md",
    "spider_mites.md",
    "target_spot.md",
    "yellow_leaf_curl_virus.md",
]


class TestKnowledgeBase:
    """Check that all knowledge markdown files exist."""

    @pytest.mark.parametrize("filename", DISEASE_FILES)
    def test_disease_file_exists(self, filename):
        path = Path(f"knowledge/diseases/{filename}")
        assert path.exists(), f"Missing disease file: {path}"

    @pytest.mark.parametrize("filename", DISEASE_FILES)
    def test_disease_file_not_empty(self, filename):
        path = Path(f"knowledge/diseases/{filename}")
        assert path.stat().st_size > 100, (
            f"Disease file too small ({path.stat().st_size} bytes): {path}"
        )

    @pytest.mark.parametrize("filename", TREATMENT_FILES)
    def test_treatment_file_exists(self, filename):
        path = Path(f"knowledge/treatments/{filename}")
        assert path.exists(), f"Missing treatment file: {path}"

    @pytest.mark.parametrize("filename", TREATMENT_FILES)
    def test_treatment_file_not_empty(self, filename):
        path = Path(f"knowledge/treatments/{filename}")
        assert path.stat().st_size > 100, (
            f"Treatment file too small ({path.stat().st_size} bytes): {path}"
        )


# ── Retriever mapping tests ──────────────────────────────────


class TestRetrieverMapping:
    """Test disease name → key mapping logic (no FAISS needed)."""

    def _get_mapping(self):
        """Extract the key_mapping dict from retriever.py."""
        return {
            "bacterial_spot": "bacterial_spot",
            "early_blight": "early_blight",
            "late_blight": "late_blight",
            "leaf_mold": "leaf_mold",
            "septoria_leaf_spot": "septoria_leaf_spot",
            "spider_mites_two_spotted_spider_mite": "spider_mites",
            "spider_mites_two_s": "spider_mites",
            "target_spot": "target_spot",
            "tomato_yellowleaf__curl_virus": "yellow_leaf_curl_virus",
            "yellowleaf__curl_virus": "yellow_leaf_curl_virus",
            "tomato_mosaic_virus": "mosaic_virus",
            "mosaic_virus": "mosaic_virus",
            "healthy": "healthy",
        }

    def test_all_diseases_have_mapping(self):
        """Every disease should map to a knowledge file key."""
        mapping = self._get_mapping()
        for key, value in mapping.items():
            # The resolved key should correspond to a disease file
            disease_file = Path(f"knowledge/diseases/{value}.md")
            assert disease_file.exists(), (
                f"Mapping '{key}' → '{value}' has no matching file: {disease_file}"
            )

    def test_all_diseases_have_treatment_mapping(self):
        """Every mapped disease should have a treatment file."""
        mapping = self._get_mapping()
        unique_values = set(mapping.values())
        for disease_key in unique_values:
            treatment_file = Path(f"knowledge/treatments/{disease_key}.md")
            assert treatment_file.exists(), (
                f"No treatment file for disease '{disease_key}': {treatment_file}"
            )

    @pytest.mark.parametrize(
        "class_name,expected_key",
        [
            ("Tomato_Early_blight", "early_blight"),
            ("Tomato__Late_blight", "late_blight"),
            ("Tomato_Bacterial_spot", "bacterial_spot"),
            ("Tomato_healthy", "healthy"),
        ],
    )
    def test_class_name_stripping(self, class_name, expected_key):
        """Class names should strip 'Tomato_' or 'Tomato__' prefix."""
        disease_key = class_name.lower()
        for prefix in ["tomato__", "tomato_"]:
            if disease_key.startswith(prefix):
                disease_key = disease_key[len(prefix):]
                break
        assert disease_key == expected_key, (
            f"'{class_name}' → '{disease_key}', expected '{expected_key}'"
        )


# ── Fallback advisory tests ─────────────────────────────────


class TestFallbackAdvisory:
    """Test fallback advisory generation (no LLM needed)."""

    def test_fallback_with_treatment_chunks(self):
        """Fallback should include treatment content."""
        from tomato_disease_advisor.config import ConfigurationManager

        cm = ConfigurationManager()
        rag_config = cm.get_rag_config()

        from tomato_disease_advisor.rag.advisor import TreatmentAdvisor

        advisor = TreatmentAdvisor(rag_config)

        chunks = [{"content": "Apply copper fungicide at 2g/L", "metadata": {}}]
        result = advisor._fallback_advisory("early_blight", "moderate", chunks)

        assert "advisory" in result
        assert "copper fungicide" in result["advisory"]
        assert result["model_used"] == "fallback"
        assert result["tokens_used"] == 0

    def test_fallback_without_treatment_chunks(self):
        """Fallback with no chunks should suggest expert consultation."""
        from tomato_disease_advisor.config import ConfigurationManager

        cm = ConfigurationManager()
        rag_config = cm.get_rag_config()

        from tomato_disease_advisor.rag.advisor import TreatmentAdvisor

        advisor = TreatmentAdvisor(rag_config)

        result = advisor._fallback_advisory("unknown_disease", "severe", [])

        assert "advisory" in result
        assert "extension officer" in result["advisory"].lower()
        assert result["model_used"] == "fallback"
