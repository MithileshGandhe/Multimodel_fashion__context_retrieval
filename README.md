# Multimodal Fashion & Context Retrieval Engine

An intelligent zero-shot search engine capable of retrieving specific fashion images from a database based on complex natural language descriptions (clothing style, color combination, scene context, weather, and vibe). 

This project implements a **Two-Stage Retrieval Pipeline** specifically designed to handle fine-grained compositionality (e.g., distinguishing *"red shirt with blue pants"* from *"blue shirt with red pants"*) which standard vision-language models like vanilla CLIP often struggle with.

---

## 🚀 Architecture Overview

```
                        User Query
                            │
                            ▼
           ┌─────────────────────────────────┐
           │      Stage 1: Fast Recall       │
           │  (SigLIP Vector Cosine Search)  │
           └─────────────────────────────────┘
                            │
                            ▼ (Top 20 Candidates)
           ┌─────────────────────────────────┐
           │     Stage 2: Deep Re-ranking    │
           │     (BLIP Cross-Encoder ITM)    │
           └─────────────────────────────────┘
                            │
                            ▼ (Top 5 Final Results)
                      Output Images
```

1.  **Stage 1: Fast Recall (SigLIP)**
    *   **Model:** `google/siglip-base-patch16-224` (Sigmoid Vision-Language model).
    *   **Mechanism:** Generates dense vector representations (embeddings) for all images in the database. When a query is entered, it calculates cosine similarity to fetch the top 20 candidate images.
    *   **Performance:** Completed in milliseconds.

2.  **Stage 2: Cross-Encoder Re-ranking (BLIP ITM)**
    *   **Model:** `Salesforce/blip-itm-base-coco` (Image-Text Matching head).
    *   **Mechanism:** Performs joint visual-textual cross-attention. Conditioned on the text query, the model explicitly checks for attribute-object binding (e.g., mapping color descriptions to correct clothing garments).
    *   **Performance:** Reranks the top candidates to produce a highly accurate final Top-5 result list.

---

## 📂 Project Structure

```
fashion_&_context_retrival/
├── test/                  # Database folder containing raw images
├── eval_results/          # Evaluation results output folder
│   ├── query_1/           # Ranked images retrieved for Query 1
│   ├── ...
│   └── evaluation_report.json # Detailed ITM & Recall score report
├── indexer.py             # Feature extraction and indexing pipeline
├── retriever.py           # Core two-stage retrieval engine
├── search.py              # Interactive CLI search utility
├── eval.py                # Evaluation runner for test queries
├── index.pt               # Pre-computed image index embeddings
├── report.md              # Project technical report (Markdown)
├── report.docx            # Project technical report (Word Document)
├── requirements.txt       # Project python dependencies
└── README.md              # This file
```

---

## 🛠️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MithileshGandhe/Multimodel_fashion__context_retrieval.git
    cd Multimodel_fashion__context_retrieval
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    
    # On Windows:
    .\venv\Scripts\activate
    
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 💻 How to Use

### 1. Interactive Search CLI
Run the interactive CLI loop to input custom search queries on the fly:
```bash
python search.py
```
Type your query and press `Enter`. Type `quit` or `exit` to exit the interface.

### 2. Single-shot Query
Query directly from the terminal:
```bash
python search.py "a person in a bright yellow raincoat"
```

### 3. Run the Evaluation Suite
Evaluate the retriever pipeline against the five core assignment benchmarks:
```bash
python eval.py
```
This runs the evaluation and saves a detailed JSON report and ranked images to `eval_results/`.

### 4. Re-Index Images (Optional)
If you update the images inside the `test/` folder, regenerate the visual index by running:
```bash
python indexer.py
```

---

## 📊 Evaluation Benchmark

The retriever has been tested against the following assignment queries:
1.  *“A person wearing a red floral dress standing in a sunny garden”*
2.  *“A man in a navy blue suit with a striped tie at a formal event”*
3.  *“A woman in athletic wear jogging in an urban park”*
4.  *“A child wearing a yellow raincoat splashing in puddles”*
5.  *“A couple dressed in matching white outfits on a beach at sunset”*

Detailed evaluation reports, including similarity metrics, are stored inside the `eval_results/` folder.
