# Multimodal Fashion & Context Retrieval — Technical Report

## 1. Problem Statement

Given a diverse dataset of fashion images, build an intelligent search engine that can retrieve specific images based on complex natural language queries describing clothing types, colours, environments, and compositional attributes. The system must go beyond vanilla CLIP to handle fine-grained, multi-attribute queries such as distinguishing *"a red shirt with blue pants"* from *"a blue shirt with red pants"*.

---

## 2. Approaches Considered

### 2.1 Vanilla CLIP (Baseline — Rejected)
**How it works:** CLIP uses a dual-encoder architecture — a text encoder and an image encoder trained contrastively to project images and text into a shared embedding space. Retrieval is performed via cosine similarity.

**Strengths:**
- Zero-shot capability; no task-specific training needed.
- Fast inference; text and image embeddings are computed independently.

**Weaknesses:**
- *Bag-of-words compositionality:* CLIP's contrastive loss doesn't learn fine-grained attribute binding. It struggles with queries like "red shirt with blue pants" vs. "blue shirt with red pants" because the same words appear in both, and the embeddings become very similar.
- Limited understanding of spatial relationships and attribute-object bindings.

### 2.2 VLMs (Vision-Language Models like LLaVA, GPT-4V)
**How it works:** Use a large VLM to generate detailed captions for all images, then perform text-to-text retrieval (e.g., using BM25 or a text embedding model).

**Strengths:**
- Rich, structured captions can capture nuanced attributes.
- Supports very complex, open-ended queries.

**Weaknesses:**
- Extremely compute-intensive for captioning the entire dataset.
- Adds a lossy information bottleneck (the caption may miss details).
- Requires a powerful GPU or API credits.

### 2.3 Two-Stage Retrieval: SigLIP + BLIP ITM (Chosen Approach ✅)
**How it works:** A fast dual-encoder (SigLIP) retrieves a shortlist of candidates, then a cross-encoder (BLIP ITM) re-ranks them with deep image-text attention.

**Why SigLIP over CLIP?** SigLIP replaces CLIP's softmax-based contrastive loss with a pairwise sigmoid loss. This gives it a stronger baseline for fine-grained image-text alignment, particularly on fashion and attribute-rich content.

**Why BLIP ITM for re-ranking?** Cross-encoders compute joint image-text representations (via cross-attention), allowing the model to explicitly reason about whether a specific attribute mentioned in the text is present in the image. This directly addresses the compositionality limitation.

---

## 3. Chosen Architecture — Deep Dive

### 3.1 Stage 1: Fast Recall (SigLIP)

```
Query: "A woman in athletic wear jogging in an urban park"
         │
         ▼
  ┌─────────────────┐
  │  SigLIP Text     │
  │  Encoder          │ ──→ Text Embedding (1, 768)
  └─────────────────┘
         │
         ▼  Cosine Similarity
  ┌─────────────────┐
  │  Pre-computed    │
  │  Image Index     │ ──→ Top-20 Candidates
  │  (N, 768)        │
  └─────────────────┘
```

- **Model:** `google/siglip-base-patch16-224`
- **Index:** All images are pre-encoded and L2-normalized. Stored as a PyTorch tensor for vectorized cosine similarity.
- **Speed:** The entire recall stage runs in milliseconds, even on CPU.

### 3.2 Stage 2: Cross-Encoder Re-ranking (BLIP ITM)

```
  Top-20 Candidates
         │
         ▼  (For each candidate)
  ┌──────────────────────────────┐
  │  BLIP ITM Cross-Encoder     │
  │  Image + Text → Cross-Attn  │ ──→ Match Score ∈ [0, 1]
  └──────────────────────────────┘
         │
         ▼  Sort by Score
  Final Top-5 Results
```

- **Model:** `Salesforce/blip-itm-base-coco`
- **Key Insight:** Unlike dual-encoders, the cross-encoder processes the image and text *jointly*. It attends to specific image regions conditioned on the text, enabling it to verify attribute-object bindings (e.g., "red" → shirt, "blue" → pants).
- **Cost:** Runs 20 forward passes (one per candidate), but each pass is fast on modern hardware.

### 3.3 Why This Architecture Works for Fashion

| Challenge | How Our System Handles It |
|---|---|
| **Attribute specificity** ("red floral dress") | SigLIP captures colour + pattern; BLIP ITM verifies binding |
| **Compositionality** ("red shirt blue pants" vs reverse) | Cross-attention in BLIP ITM resolves attribute-object pairs |
| **Context awareness** ("sunny garden", "urban park") | SigLIP encodes scene-level semantics in the embedding |
| **Multi-person scenes** ("couple in matching outfits") | BLIP ITM attends to multiple subjects jointly |

---

## 4. Implementation Details

| Component | Details |
|---|---|
| **Language** | Python 3.12 |
| **Framework** | PyTorch 2.4 + HuggingFace Transformers 4.45 |
| **Stage 1 Model** | `google/siglip-base-patch16-224` (400M params) |
| **Stage 2 Model** | `Salesforce/blip-itm-base-coco` (~250M params) |
| **Vector Store** | In-memory PyTorch `.pt` file (embeddings + paths) |
| **Top-N Recall** | 20 candidates |
| **Top-K Final** | 5 results |

### File Structure
```
fashion_&_context_retrival/
├── test/                  # Image dataset (~500+ images)
├── indexer.py             # Build SigLIP image index
├── retriever.py           # Two-stage retrieval engine
├── search.py              # Interactive CLI search
├── eval.py                # Run evaluation queries
├── index.pt               # Pre-computed image embeddings
├── eval_results/          # Evaluation output images + JSON
└── report.md              # This document
```

---

## 5. Evaluation Queries

The system was evaluated on 5 queries designed to test different retrieval challenges:

1. **"A person wearing a red floral dress standing in a sunny garden"** — Tests colour + pattern + scene.
2. **"A man in a navy blue suit with a striped tie at a formal event"** — Tests fine-grained fashion attributes.
3. **"A woman in athletic wear jogging in an urban park"** — Tests action + context.
4. **"A child wearing a yellow raincoat splashing in puddles"** — Tests subject + action + weather context.
5. **"A couple dressed in matching white outfits on a beach at sunset"** — Tests multi-person + scene.

Results (with top-5 retrieved images per query) are available in the `eval_results/` directory.

---

## 6. Future Work

### 6.1 Location-Aware Retrieval
Integrate a geolocation/scene classification model (e.g., PlacesNet) to add location tags (beach, city, park, indoor) as metadata filters. This would allow pre-filtering before embedding search, dramatically improving precision for location-specific queries.

### 6.2 Weather/Lighting Conditions
Train a lightweight classifier on weather attributes (sunny, rainy, overcast, golden hour) and add these as structured metadata. This helps queries like "a rainy day outfit" or "sunset beach scene" that rely on atmospheric context.

### 6.3 Fine-tuned Fashion Embeddings
Fine-tune SigLIP on a fashion-specific dataset (e.g., DeepFashion2 or FashionIQ) using hard-negative mining to improve attribute binding accuracy. This would boost Stage 1 recall quality.

### 6.4 Hybrid Retrieval with Generated Captions
Use a VLM to pre-generate structured captions for each image (color, garment type, scene, activity). Store these alongside embeddings and add BM25-based text filtering as a parallel retrieval path. Fuse scores from visual retrieval and text retrieval for a hybrid approach.

### 6.5 User Feedback Loop
Implement a relevance feedback mechanism where users mark results as relevant/irrelevant, and the system re-ranks using the feedback signal (Rocchio-style query expansion in embedding space).

---

## 7. Conclusion

The two-stage architecture (SigLIP recall + BLIP ITM re-ranking) provides a practical and effective solution for fashion image retrieval that explicitly addresses the compositionality challenge of vanilla CLIP. The approach is scalable (SigLIP indexing is fast), accurate (cross-encoder re-ranking catches attribute mismatches), and extensible (metadata filters can be added for location/weather).
