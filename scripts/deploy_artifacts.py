"""
Deploy artifacts to HuggingFace Space — uploads only changed files.

Compares SHA-256 hashes against last deployment to avoid redundant uploads.

Usage:
    python scripts/deploy_artifacts.py                    # Upload changed artifacts
    python scripts/deploy_artifacts.py --force             # Force upload all
    python scripts/deploy_artifacts.py --dry-run           # Show what would upload
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


SPACE_ID = "AIenthusSP/agricultural-disease-advisory-system"
HASH_FILE = Path(".deploy_hashes.json")

# Artifacts to track: (local_path, path_in_space, description)
ARTIFACTS = [
    ("artifacts/training/model.keras", "artifacts/training/model.keras", "Trained model"),
    ("artifacts/vectorstore", "artifacts/vectorstore", "FAISS vectorstore"),
]


def sha256_file(filepath: Path) -> str:
    """Compute SHA-256 of a single file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_directory(dirpath: Path) -> str:
    """Compute combined SHA-256 of all files in a directory (sorted)."""
    h = hashlib.sha256()
    for filepath in sorted(dirpath.rglob("*")):
        if filepath.is_file():
            h.update(str(filepath.relative_to(dirpath)).encode())
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
    return h.hexdigest()


def compute_hash(path: Path) -> str:
    """Compute hash of file or directory."""
    if path.is_file():
        return sha256_file(path)
    elif path.is_dir():
        return sha256_directory(path)
    return ""


def load_hashes() -> dict:
    """Load previous deployment hashes."""
    if HASH_FILE.exists():
        with open(HASH_FILE, "r") as f:
            return json.load(f)
    return {}


def save_hashes(hashes: dict):
    """Save current deployment hashes."""
    with open(HASH_FILE, "w") as f:
        json.dump(hashes, f, indent=2)


def upload_to_space(local_path: str, space_path: str) -> bool:
    """Upload file/directory to HuggingFace Space using CLI."""
    cmd = [
        "huggingface-cli", "upload",
        SPACE_ID,
        local_path,
        space_path,
        "--repo-type", "space",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ❌ Upload failed: {result.stderr.strip()}")
        return False
    print(f"  ✅ Upload successful")
    return True


def main():
    parser = argparse.ArgumentParser(description="Deploy changed artifacts to HF Space")
    parser.add_argument("--force", action="store_true", help="Force upload all artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    args = parser.parse_args()

    # Verify HF CLI is available
    try:
        subprocess.run(
            ["huggingface-cli", "--version"],
            capture_output=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ huggingface-cli not found. Run: pip install huggingface_hub[cli]")
        return 1

    previous_hashes = load_hashes()
    current_hashes = {}
    to_upload = []

    print(f"\n📦 Checking artifacts for changes...\n")

    for local_path, space_path, description in ARTIFACTS:
        path = Path(local_path)

        if not path.exists():
            print(f"  ⚠️  {description}: {local_path} not found — skipping")
            continue

        current_hash = compute_hash(path)
        current_hashes[local_path] = current_hash
        previous_hash = previous_hashes.get(local_path, "")

        if args.force or current_hash != previous_hash:
            status = "FORCE" if args.force else "CHANGED"
            print(f"  🔄 {description}: {status}")
            to_upload.append((local_path, space_path, description))
        else:
            print(f"  ✅ {description}: unchanged — skipping")

    if not to_upload:
        print(f"\n✅ All artifacts are up-to-date. Nothing to deploy.\n")
        return 0

    print(f"\n{'📋 Would upload' if args.dry_run else '🚀 Uploading'} "
          f"{len(to_upload)} artifact(s):\n")

    if args.dry_run:
        for local_path, space_path, description in to_upload:
            print(f"  • {description}: {local_path} → {space_path}")
        print(f"\nRun without --dry-run to upload.")
        return 0

    success_count = 0
    for local_path, space_path, description in to_upload:
        print(f"\n📤 Uploading {description}...")
        if upload_to_space(local_path, space_path):
            success_count += 1

    # Save hashes only for successful uploads
    if success_count == len(to_upload):
        save_hashes(current_hashes)
        print(f"\n✅ All {success_count} artifact(s) deployed successfully!\n")
    else:
        print(f"\n⚠️  {success_count}/{len(to_upload)} uploaded. "
              f"Hashes NOT saved — retry failed uploads.\n")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
