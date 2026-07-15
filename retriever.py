"""
retriever.py — Two-Stage Fashion Image Retriever

Stage 1 (Recall):  SigLIP text encoder → cosine similarity against index.
Stage 2 (Re-rank): BLIP Image-Text Matching cross-encoder for compositional
                    understanding and fine-grained attribute matching.
"""

import os
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModel,
    BlipProcessor,
    BlipForImageTextRetrieval,
)


# ── Configuration ──────────────────────────────────────────────────────
SIGLIP_MODEL = "google/siglip-base-patch16-224"
BLIP_ITM_MODEL = "Salesforce/blip-itm-base-coco"
INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.pt")
TOP_N_RECALL = 20          # Candidates from Stage 1
TOP_K_FINAL  = 5           # Final results after re-ranking


class FashionRetriever:
    """Two-stage multimodal retriever: SigLIP recall + BLIP ITM re-ranking."""

    def __init__(
        self,
        index_path: str = INDEX_PATH,
        device: str | None = None,
        top_n: int = TOP_N_RECALL,
        top_k: int = TOP_K_FINAL,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_n = top_n
        self.top_k = top_k

        # ── Load index ─────────────────────────────────────────────────
        print(f"[retriever] Loading index from {index_path}")
        index = torch.load(index_path, map_location="cpu", weights_only=False)
        self.embeddings = index["embeddings"].to(self.device)  # (N, D)
        self.paths = index["paths"]                            # list[str]

        # ── Stage 1 model: SigLIP ──────────────────────────────────────
        print(f"[retriever] Loading SigLIP: {SIGLIP_MODEL}")
        self.siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
        self.siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL).to(self.device)
        self.siglip_model.eval()

        # ── Stage 2 model: BLIP ITM ───────────────────────────────────
        print(f"[retriever] Loading BLIP ITM: {BLIP_ITM_MODEL}")
        self.blip_processor = BlipProcessor.from_pretrained(BLIP_ITM_MODEL)
        self.blip_model = BlipForImageTextRetrieval.from_pretrained(
            BLIP_ITM_MODEL
        ).to(self.device)
        self.blip_model.eval()

        print("[retriever] Ready.")

    # ── Stage 1: Fast Recall via SigLIP ─────────────────────────────
    @torch.no_grad()
    def _recall(self, query: str, top_n: int | None = None) -> list[tuple[str, float]]:
        """Encode the query with SigLIP and return top_n candidates by cosine sim."""
        top_n = top_n or self.top_n

        inputs = self.siglip_processor(
            text=[query], return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        text_emb = self.siglip_model.get_text_features(**inputs)      # (1, D)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        sims = (self.embeddings @ text_emb.T).squeeze(-1)             # (N,)
        topk = sims.topk(min(top_n, len(self.paths)))

        candidates = []
        for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
            candidates.append((self.paths[idx], score))
        return candidates

    # ── Stage 2: BLIP ITM Re-ranking ────────────────────────────────
    @torch.no_grad()
    def _rerank(
        self, query: str, candidates: list[tuple[str, float]], top_k: int | None = None
    ) -> list[tuple[str, float, float]]:
        """Re-rank candidates using BLIP Image-Text Matching scores."""
        top_k = top_k or self.top_k
        scored = []

        for path, recall_score in candidates:
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue

            inputs = self.blip_processor(
                images=img, text=query, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.blip_model(**inputs)
            # ITM head outputs logits: [not-match, match]
            itm_score = outputs.itm_score.softmax(dim=-1)[0, 1].item()
            scored.append((path, recall_score, itm_score))

        # Sort by ITM score (descending)
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    # ── Public API ──────────────────────────────────────────────────
    def search(
        self,
        query: str,
        top_n: int | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Run the full two-stage retrieval pipeline.

        Returns a list of dicts with keys:
            path, filename, recall_score, itm_score, rank
        """
        candidates = self._recall(query, top_n)
        reranked = self._rerank(query, candidates, top_k)

        results = []
        for rank, (path, recall_score, itm_score) in enumerate(reranked, 1):
            results.append(
                {
                    "rank": rank,
                    "path": path,
                    "filename": os.path.basename(path),
                    "recall_score": round(recall_score, 4),
                    "itm_score": round(itm_score, 4),
                }
            )
        return results
