"""
indexer.py — Fashion Image Indexer using SigLIP

Processes all images in the dataset directory, extracts normalized
embeddings using SigLIP (google/siglip-base-patch16-224), and saves
the index as a .pt file for fast retrieval.
"""

import os
import sys
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel


# ── Configuration ──────────────────────────────────────────────────────
MODEL_NAME = "google/siglip-base-patch16-224"
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")
INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.pt")
BATCH_SIZE = 32


def load_model(device: str):
    """Load SigLIP model and processor."""
    print(f"[indexer] Loading SigLIP model: {MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"[indexer] Model loaded on {device}")
    return model, processor


def gather_image_paths(image_dir: str) -> list[str]:
    """Collect all .jpg / .jpeg / .png image paths from the directory."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = []
    for fname in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_exts:
            paths.append(os.path.join(image_dir, fname))
    print(f"[indexer] Found {len(paths)} images in {image_dir}")
    return paths


@torch.no_grad()
def extract_embeddings(
    image_paths: list[str],
    model,
    processor,
    device: str,
    batch_size: int = BATCH_SIZE,
) -> torch.Tensor:
    """Extract L2-normalized SigLIP image embeddings in batches."""
    all_embeddings = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Indexing"):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"[indexer] Warning: skipping {p} — {e}")
                continue

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.get_image_features(**inputs)          # (B, D)
        embeddings = outputs / outputs.norm(dim=-1, keepdim=True)  # L2 norm
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def build_index(image_dir: str = IMAGE_DIR, index_path: str = INDEX_PATH):
    """Main entry point: build and save the image index."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model(device)

    image_paths = gather_image_paths(image_dir)
    if not image_paths:
        print("[indexer] No images found. Exiting.")
        sys.exit(1)

    embeddings = extract_embeddings(image_paths, model, processor, device)
    print(f"[indexer] Embeddings shape: {embeddings.shape}")

    # Save index: embeddings tensor + ordered list of file paths
    index = {
        "embeddings": embeddings,
        "paths": image_paths,
    }
    torch.save(index, index_path)
    print(f"[indexer] Index saved to {index_path}")
    return index


if __name__ == "__main__":
    build_index()
