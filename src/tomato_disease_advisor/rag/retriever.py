"""
Knowledge Retriever

Loads a pre-built FAISS index and retrieves the most relevant knowledge
chunks for a given disease query. Supports filtering by disease name
and knowledge category (diseases / treatments).

Used by: advisor.py and prediction.py
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


class KnowledgeRetriever:
    """
    Retrieves relevant knowledge from FAISS vector store.

    Supports:
      - Semantic search (embed query → find nearest vectors)
      - Metadata filtering (by disease_name, category)
      - Score thresholding
    """

    def __init__(
        self,
        index_path: str,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize KnowledgeRetriever.

        Args:
            index_path: Path to the FAISS index directory
            embedding_model: Override embedding model name. If None,
                             uses the model stored in store_config.json.
        """
        self.index_path = Path(index_path)
        self._model = None
        self._index = None
        self._metadata = None

        # Load store config
        config_path = self.index_path / "store_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                self._store_config = json.load(f)
        else:
            self._store_config = {}

        self._embedding_model_name = (
            embedding_model
            or self._store_config.get("embedding_model", "all-MiniLM-L6-v2")
        )

    def _load_index(self):
        """Lazy-load FAISS index and metadata."""
        if self._index is not None:
            return

        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu required: pip install faiss-cpu")

        # Load FAISS index
        faiss_path = self.index_path / "index.faiss"
        if not faiss_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {faiss_path}. "
                f"Run stage_05_build_vectorstore first."
            )

        self._index = faiss.read_index(str(faiss_path))
        print(f"[Retriever] Loaded FAISS index: {self._index.ntotal} vectors")

        # Load metadata
        meta_path = self.index_path / "metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        print(f"[Retriever] Loaded metadata: {len(self._metadata)} chunks")

    def _get_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._embedding_model_name)
            print(f"[Retriever] Loaded embedding model: "
                  f"{self._embedding_model_name}")
        return self._model

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        disease_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict]:
        """
        Retrieve relevant knowledge chunks for a query.

        Args:
            query: Search query (e.g., disease name + "treatment")
            top_k: Number of results to return
            disease_filter: Filter by disease_name (e.g., "early_blight")
            category_filter: Filter by category ("diseases" or "treatments")
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of dicts with: content, metadata, score
        """
        self._load_index()
        model = self._get_model()

        # Generate query embedding
        query_embedding = model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        # Search with more results than needed for post-filtering
        search_k = min(top_k * 3, self._index.ntotal)
        scores, indices = self._index.search(query_embedding, search_k)

        # Filter and format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue

            entry = self._metadata[idx]
            meta = entry.get("metadata", {})

            # Apply filters
            if disease_filter and meta.get("disease_name") != disease_filter:
                continue
            if category_filter and meta.get("category") != category_filter:
                continue
            if score < score_threshold:
                continue

            results.append({
                "content": entry["content"],
                "metadata": meta,
                "score": float(score),
            })

            if len(results) >= top_k:
                break

        return results

    def retrieve_for_disease(
        self,
        disease_name: str,
        top_k: int = 5,
    ) -> Dict[str, List[Dict]]:
        """
        Retrieve both disease info and treatment info for a specific disease.

        This is the primary method used by the prediction pipeline.

        Args:
            disease_name: Disease key (e.g., "early_blight", "bacterial_spot")
            top_k: Number of chunks per category

        Returns:
            dict with 'disease_info' and 'treatment_info' lists
        """
        # Convert class name format to disease key
        # e.g., "Tomato_Early_blight" → "early_blight"
        disease_key = disease_name.lower()
        for prefix in ["tomato_", "tomato__"]:
            if disease_key.startswith(prefix):
                disease_key = disease_key[len(prefix):]
                break

        # Map class names to knowledge file stems
        key_mapping = {
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

        # Try exact match first, then fuzzy
        resolved_key = key_mapping.get(disease_key, disease_key)

        # Semantic search as additional retrieval
        query = f"tomato {disease_name} disease symptoms treatment"

        # Get disease info
        disease_info = self.retrieve(
            query=query,
            top_k=top_k,
            disease_filter=resolved_key,
            category_filter="diseases",
        )

        # If no exact match results, try semantic search without filter
        if not disease_info:
            disease_info = self.retrieve(
                query=query,
                top_k=top_k,
                category_filter="diseases",
            )

        # Get treatment info
        treatment_info = self.retrieve(
            query=query,
            top_k=top_k,
            disease_filter=resolved_key,
            category_filter="treatments",
        )

        if not treatment_info:
            treatment_info = self.retrieve(
                query=query,
                top_k=top_k,
                category_filter="treatments",
            )

        return {
            "disease_info": disease_info,
            "treatment_info": treatment_info,
            "disease_key": resolved_key,
        }
