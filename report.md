# Multimodal Fashion & Context Retrieval Report

## 1. Approaches

To build an intelligent fashion search engine, several architectures can be considered, each offering distinct tradeoffs between accuracy, scalability, and compositionality.

### Approach A: Dual-Encoder Architecture (e.g., CLIP / SigLIP)
*   **Mechanism:** Projects text descriptions and images into a shared semantic embedding space. Retrieval is performed using fast vector similarity search (e.g., Cosine Similarity).
*   **Tradeoffs:**
    *   *Pros:* Extremely fast inference and highly scalable. Can run nearest neighbor queries in milliseconds on millions of images.
    *   *Cons:* Struggles with compositionality and attribute-object binding (the "bag-of-words" problem). For example, it often cannot distinguish between "a red shirt and blue pants" and "a blue shirt and red pants."
*   **Best Use Cases:** Initial candidate retrieval stages or large-scale web searches.

### Approach B: Cross-Encoder Architecture (e.g., Image-Text Matching / ITM models)
*   **Mechanism:** Processes the image and query text jointly through a transformer model, employing cross-attention layers to calculate an alignment score.
*   **Tradeoffs:**
    *   *Pros:* Highly precise. Effectively handles compositionality, fine-grained details, and attribute-object associations because it reasons over joint features.
    *   *Cons:* Extremely slow and computationally expensive. It requires running the heavy transformer forward pass for every query-image pair, making it unscalable for direct search over large databases.
*   **Best Use Cases:** Re-ranking a small list of candidate images retrieved by a faster method.

### Approach C: Two-Stage Hybrid Pipeline (SigLIP Recall + BLIP ITM Re-ranking)
*   **Mechanism:** A combination of the above. Stage 1 utilizes a fast dual-encoder to extract a shortlist of candidates (e.g., top 20). Stage 2 uses a cross-encoder to re-rank the shortlist.
*   **Tradeoffs:**
    *   *Pros:* Balances speed and precision. Captures complex semantic context and compositionality while remaining scalable to larger databases.
    *   *Cons:* Requires hosting two models and adds a minor latency overhead for the re-ranking stage.
*   **Best Use Cases:** Multi-attribute, fine-grained queries requiring semantic accuracy and context awareness.

---

## 2. Short Write-up on Chosen Approach

The chosen approach is the **Two-Stage Hybrid Pipeline** combining **SigLIP** (for recall) and **BLIP ITM** (for re-ranking).

### Why SigLIP (Stage 1 Recall)
We utilize `google/siglip-base-patch16-224` to compute embeddings for all database images. Unlike standard CLIP, which uses a softmax normalization loss, SigLIP utilizes a sigmoid loss that optimizes pairwise image-text alignment. This gives SigLIP a stronger baseline for fine-grained zero-shot recognition, capturing style, context, and basic color schemas efficiently.

### Why BLIP ITM (Stage 2 Re-ranking)
To solve the compositionality issue (e.g., distinguishing "red shirt with blue pants"), we use `Salesforce/blip-itm-base-coco` as a cross-encoder. 
*   **Feature Interaction:** It uses cross-attention layers to let the query text tokens directly interact with the visual patches of the retrieved images.
*   **Attribute Binding:** The model verifies if the specific colors and clothing styles match the correct subjects by computing an explicit Image-Text Matching (ITM) confidence score. The system sorts the SigLIP candidates by this ITM score to deliver the final results.

---

## 3. Codebase (GitHub) Link

The complete implementation pipelines can be accessed in the repository:
*   **Codebase Link:** [https://github.com/MithileshGandhe/Multimodel_fashion__context_retrieval](https://github.com/MithileshGandhe/Multimodel_fashion__context_retrieval)

### Key Files in Repository:
*   [indexer.py](file:///c:/Users/Lenovo/Desktop/fashion_&_context_retrival/indexer.py): Handles batch feature extraction and L2-normalized image embedding indexing.
*   [retriever.py](file:///c:/Users/Lenovo/Desktop/fashion_&_context_retrival/retriever.py): Implementation of the Two-Stage Retriever class (Recall + ITM Re-ranker).
*   [search.py](file:///c:/Users/Lenovo/Desktop/fashion_&_context_retrival/search.py): Command-line search interface supporting interactive and single-query search modes.
*   [eval.py](file:///c:/Users/Lenovo/Desktop/fashion_&_context_retrival/eval.py): Evaluation script that validates the pipeline using the 5 evaluation queries.

---

## 4. Approaches for future work

### How to extend this solution for adding locations and weather
To incorporate location (e.g., cities, landmarks, home/office settings) and weather context:
1.  **Metadata Pre-filtering:** Build a lightweight classifier (e.g., a ResNet fine-tuned on Places365 for locations and weather attributes) to pre-tag images with structured metadata (e.g., `location: office`, `weather: rainy`).
2.  **Query Parsing (NER):** Parse queries using Named Entity Recognition (NER) to detect location or weather keywords.
3.  **Structured SQL/Vector Search:** Filter out images that do not match the parsed metadata before performing the vector search. This prevents visual features from getting confused by unrelated contextual queries.

### How to improve precision
1.  **Fine-tuning on Fashion Datasets:** Contrastively fine-tune the SigLIP vision-text encoders on fashion datasets (e.g., DeepFashion2, Fashionpedia, or FashionIQ) using hard-negative mining (specifically pairing "blue shirt, red pants" text with "red shirt, blue pants" images as negative pairs) to teach the dual-encoder strict compositionality.
2.  **VLM Dense Captioning:** Use a Vision-Language Model (VLM) such as LLaVA or Qwen-VL to auto-generate highly detailed, structured text descriptions of the clothes (colors, fabrics, patterns) and contexts. Combine these captions with a text embedding model (e.g., SentenceTransformers) to enable hybrid search (combining dense visual search with text-based BM25).
