"""
Upload trained model to HuggingFace Hub.

Usage:
    python scripts/upload_model_to_hf.py \
        --model-path artifacts/training/model.keras \
        --repo-id ShubhamPawar-3333/tomato-disease-efficientnet \
        --token $HF_TOKEN
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Upload model to HuggingFace Hub"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/training/model.keras",
        help="Path to the trained model file",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="AIenthusSP/agricultural-disease-advisory-system",
        help="HuggingFace Hub repo ID (user/repo)",
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="HuggingFace API token",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: Model file not found at {model_path}")
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: pip install huggingface_hub")
        return 1

    api = HfApi(token=args.token)

    # Create repo if it doesn't exist
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        exist_ok=True,
    )

    # Upload model file
    print(f"Uploading {model_path} to {args.repo_id}...")
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo=model_path.name,
        repo_id=args.repo_id,
        repo_type="model",
    )

    # Upload MODEL_CARD.md as the repo README
    card_path = Path("MODEL_CARD.md")
    if card_path.exists():
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
        )
        print("Uploaded MODEL_CARD.md as README.md")

    print(f"✅ Model uploaded to https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
