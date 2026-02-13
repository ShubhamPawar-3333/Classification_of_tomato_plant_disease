"""
FAISS Vector Store Builder

Loads disease and treatment knowledge markdown files, chunks them,
generates embeddings using SciBERT (or any HuggingFace model), and
builds a FAISS index for fast similarity search.

Used by: stage_05_build_vectorstore.py (DVC pipeline)
Used at inference by: retriever.py
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

from tomato_disease_advisor.entity import VectorStoreConfig


class KnowledgeDocument:
    """Represents a chunk of knowledge with metadata."""

    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata

    def __repr__(self):
        return f"KnowledgeDocument(source={self.metadata.get('source', '?')}, len={len(self.content)})"


class VectorStoreBuilder:
    """
    Builds a FAISS vector store from knowledge markdown files.

    Workflow:
      1. Load all .md files from knowledge/diseases/ and knowledge/treatments/
      2. Split into chunks (with overlap)
      3. Generate embeddings using sentence-transformers
      4. Build FAISS index
      5. Save index + metadata to disk
    """

    def __init__(self, config: VectorStoreConfig):
        """
        Initialize VectorStoreBuilder.

        Args:
            config: VectorStoreConfig with paths and embedding settings
        """
        self.config = config

    def _load_markdown_files(self) -> List[Dict[str, str]]:
        """
        Load all markdown files from the knowledge directory.

        Returns:
            List of dicts with 'content', 'source', and 'category' keys
        """
        knowledge_dir = Path(self.config.knowledge_dir)
        documents = []

        for subdir in ["diseases", "treatments"]:
            dir_path = knowledge_dir / subdir
            if not dir_path.exists():
                print(f"[VectorStore] Warning: {dir_path} does not exist")
                continue

            for md_file in sorted(dir_path.glob("*.md")):
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                if content:
                    documents.append({
                        "content": content,
                        "source": str(md_file.name),
                        "category": subdir,  # "diseases" or "treatments"
                        "disease_name": md_file.stem,  # e.g., "early_blight"
                    })
                    print(f"[VectorStore] Loaded: {subdir}/{md_file.name} "
                          f"({len(content)} chars)")

        print(f"[VectorStore] Total documents loaded: {len(documents)}")
        return documents

    def _chunk_text(
        self, text: str, chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        """
        Split text into overlapping chunks.

        Uses a simple character-based chunking with awareness of markdown
        headers (##) — tries to split at section boundaries when possible.

        Args:
            text: Full document text
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between consecutive chunks

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        # Try to split at markdown headers first
        sections = []
        current_section = ""

        for line in text.split("\n"):
            if line.startswith("## ") and current_section:
                sections.append(current_section.strip())
                current_section = line + "\n"
            else:
                current_section += line + "\n"

        if current_section.strip():
            sections.append(current_section.strip())

        # Now chunk sections that are too large
        for section in sections:
            if len(section) <= chunk_size:
                chunks.append(section)
            else:
                # Further split large sections by overlap
                start = 0
                while start < len(section):
                    end = start + chunk_size
                    chunk = section[start:end]

                    # Try to break at a newline
                    if end < len(section):
                        last_newline = chunk.rfind("\n")
                        if last_newline > chunk_size * 0.5:
                            chunk = chunk[:last_newline]
                            end = start + last_newline

                    chunks.append(chunk.strip())
                    start = end - chunk_overlap

        return [c for c in chunks if c.strip()]

    def _create_documents(
        self, raw_docs: List[Dict[str, str]]
    ) -> List[KnowledgeDocument]:
        """
        Chunk raw documents into KnowledgeDocument objects.

        Args:
            raw_docs: List of raw document dicts

        Returns:
            List of KnowledgeDocument with chunked content and metadata
        """
        all_chunks = []

        for doc in raw_docs:
            chunks = self._chunk_text(
                doc["content"],
                self.config.chunk_size,
                self.config.chunk_overlap,
            )

            for i, chunk in enumerate(chunks):
                knowledge_doc = KnowledgeDocument(
                    content=chunk,
                    metadata={
                        "source": doc["source"],
                        "category": doc["category"],
                        "disease_name": doc["disease_name"],
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                )
                all_chunks.append(knowledge_doc)

        print(f"[VectorStore] Total chunks created: {len(all_chunks)}")
        return all_chunks

    def build(self) -> str:
        """
        Build the FAISS vector store from knowledge files.

        Returns:
            str: Path to the saved FAISS index directory
        """
        from sentence_transformers import SentenceTransformer
        import numpy as np

        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required. Install with: pip install faiss-cpu"
            )

        # Step 1: Load documents
        print("[VectorStore] Step 1: Loading knowledge files...")
        raw_docs = self._load_markdown_files()

        if not raw_docs:
            raise ValueError(
                f"No markdown files found in {self.config.knowledge_dir}. "
                f"Ensure knowledge/diseases/ and knowledge/treatments/ "
                f"contain .md files."
            )

        # Step 2: Chunk documents
        print("[VectorStore] Step 2: Chunking documents...")
        documents = self._create_documents(raw_docs)

        # Step 3: Generate embeddings
        print(f"[VectorStore] Step 3: Generating embeddings with "
              f"{self.config.embedding_model}...")
        model = SentenceTransformer(self.config.embedding_model)

        texts = [doc.content for doc in documents]
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=16,
            normalize_embeddings=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        print(f"[VectorStore] Embedding shape: {embeddings.shape}")

        # Step 4: Build FAISS index
        print("[VectorStore] Step 4: Building FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product (cosine sim for normalized vectors)
        index.add(embeddings)

        print(f"[VectorStore] Index built with {index.ntotal} vectors "
              f"(dim={dimension})")

        # Step 5: Save index and metadata
        index_path = Path(self.config.index_path)
        index_path.mkdir(parents=True, exist_ok=True)

        faiss_path = index_path / "index.faiss"
        faiss.write_index(index, str(faiss_path))

        # Save document metadata and contents
        metadata = []
        for doc in documents:
            metadata.append({
                "content": doc.content,
                "metadata": doc.metadata,
            })

        meta_path = index_path / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Save config info for retriever
        config_info = {
            "embedding_model": self.config.embedding_model,
            "dimension": dimension,
            "num_vectors": index.ntotal,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
        }
        config_path = index_path / "store_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_info, f, indent=2)

        print(f"[VectorStore] ✅ Index saved to: {index_path}")
        print(f"[VectorStore]    - index.faiss ({faiss_path.stat().st_size / 1024:.1f} KB)")
        print(f"[VectorStore]    - metadata.json ({len(metadata)} chunks)")
        print(f"[VectorStore]    - store_config.json")

        return str(index_path)
