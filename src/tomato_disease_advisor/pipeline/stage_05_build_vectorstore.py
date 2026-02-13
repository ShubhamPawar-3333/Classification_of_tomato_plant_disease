"""
Stage 05: Build Vector Store

DVC pipeline stage that builds the FAISS vector store from
knowledge markdown files (diseases + treatments).

Usage:
    python -m tomato_disease_advisor.pipeline.stage_05_build_vectorstore
"""
from tomato_disease_advisor.config import ConfigurationManager
from tomato_disease_advisor.rag import VectorStoreBuilder


STAGE_NAME = "Build Vector Store"


def main():
    """Build the FAISS vector store from knowledge files."""
    print(f"\n{'='*60}")
    print(f">>> Stage: {STAGE_NAME}")
    print(f"{'='*60}\n")

    try:
        # Get configuration
        config = ConfigurationManager()
        vectorstore_config = config.get_vectorstore_config()

        # Build vector store
        builder = VectorStoreBuilder(config=vectorstore_config)
        index_path = builder.build()

        print(f"\n{'='*60}")
        print(f">>> Stage {STAGE_NAME} completed successfully!")
        print(f">>> Index saved to: {index_path}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n>>> Stage {STAGE_NAME} FAILED!")
        print(f">>> Error: {e}")
        raise


if __name__ == "__main__":
    main()
