"""
ColPali Indexing Skeleton — PDF/Image → MongoDB Atlas Vector Search
===================================================================
Starter code for Theme 3 (Adaptive Retrieval) hackathon ideas.

What this gives you:
- ColPali v1.2 page-level VLM embeddings (no OCR needed)
- PDF → page images → embeddings → MongoDB Atlas Vector Search
- Query: text or image query → nearest page matches
- Works on scanned PDFs, medical images, satellite maps, engineering schematics

Paper anchor: ColPali (arXiv:2407.01449)

Install:
    pip install colpali-engine pdf2image Pillow pymongo torch transformers
    # Also: poppler-utils (Linux/Mac) or poppler (Windows via conda)
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Generator

import torch
from PIL import Image
from pymongo import MongoClient

# ColPali — install: pip install colpali-engine
# from colpali_engine.models import ColPali, ColPaliProcessor

MONGODB_URI = os.environ["MONGODB_URI"]
DB_NAME = os.environ.get("MONGODB_DB", "colpali_demo")

mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[DB_NAME]
pages_col = db["page_embeddings"]

# Create Atlas Vector Search index (run once in Atlas UI or via API):
# {
#   "mappings": {
#     "dynamic": true,
#     "fields": {
#       "embedding": [{
#         "type": "knnVector",
#         "dimensions": 128,
#         "similarity": "cosine"
#       }]
#     }
#   }
# }
VECTOR_INDEX_NAME = "colpali_vector_index"


# ---------------------------------------------------------------------------
# ColPali model loader
# ---------------------------------------------------------------------------

def load_colpali_model():
    """
    Load ColPali v1.2. Requires GPU for reasonable speed (CPU works for demo).
    Falls back to a mock for environments without colpali-engine installed.
    """
    try:
        from colpali_engine.models import ColPali, ColPaliProcessor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ColPali.from_pretrained(
            "vidore/colpali-v1.2",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )
        processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
        model.eval()
        print(f"ColPali loaded on {device}")
        return model, processor
    except ImportError:
        print("colpali-engine not installed — using mock embeddings for demo")
        return None, None


# ---------------------------------------------------------------------------
# PDF → page images
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: str) -> Generator[tuple[int, Image.Image], None, None]:
    """Convert each page of a PDF to a PIL Image."""
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(pdf_path, dpi=150)
        for i, page in enumerate(pages):
            yield i, page
    except ImportError:
        print("pdf2image not installed — using single dummy image")
        dummy = Image.new("RGB", (800, 1000), color=(255, 255, 255))
        yield 0, dummy


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_image(model, processor, image: Image.Image) -> list[float]:
    """Compute ColPali page-level embedding for a single image."""
    if model is None:
        # Mock: return random 128-dim vector for demo without GPU
        import random
        return [random.gauss(0, 1) for _ in range(128)]

    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze()
    return embedding.cpu().float().tolist()


def embed_query_text(model, processor, query: str) -> list[float]:
    """Compute ColPali text query embedding."""
    if model is None:
        import random
        return [random.gauss(0, 1) for _ in range(128)]

    inputs = processor(text=query, return_tensors="pt").to(model.device)
    with torch.no_grad():
        embedding = model.get_text_features(**inputs).squeeze()
    return embedding.cpu().float().tolist()


# ---------------------------------------------------------------------------
# Index pipeline
# ---------------------------------------------------------------------------

def index_pdf(pdf_path: str, doc_id: str, model, processor, metadata: dict = None) -> int:
    """
    Index all pages of a PDF into MongoDB.
    Returns the number of pages indexed.
    """
    metadata = metadata or {}
    indexed = 0

    for page_num, page_image in pdf_to_images(pdf_path):
        embedding = embed_image(model, processor, page_image)

        # Store as PNG bytes in MongoDB (GridFS better for large docs)
        img_buffer = io.BytesIO()
        page_image.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()

        pages_col.update_one(
            {"doc_id": doc_id, "page_num": page_num},
            {"$set": {
                "doc_id": doc_id,
                "page_num": page_num,
                "embedding": embedding,
                "image_bytes": img_bytes,  # For demo; use GridFS in production
                "file_name": Path(pdf_path).name,
                "metadata": metadata,
            }},
            upsert=True
        )
        indexed += 1
        print(f"  Indexed page {page_num + 1} of {doc_id}")

    return indexed


def index_image_file(image_path: str, doc_id: str, model, processor, metadata: dict = None) -> None:
    """Index a single image (satellite map, X-ray, schematic) into MongoDB."""
    metadata = metadata or {}
    image = Image.open(image_path).convert("RGB")
    embedding = embed_image(model, processor, image)

    pages_col.update_one(
        {"doc_id": doc_id, "page_num": 0},
        {"$set": {
            "doc_id": doc_id,
            "page_num": 0,
            "embedding": embedding,
            "file_name": Path(image_path).name,
            "metadata": metadata,
        }},
        upsert=True
    )
    print(f"Indexed image: {image_path}")


# ---------------------------------------------------------------------------
# Query pipeline
# ---------------------------------------------------------------------------

def query_by_text(query_text: str, model, processor, top_k: int = 5) -> list[dict]:
    """Find the most relevant pages for a text query."""
    query_embedding = embed_query_text(model, processor, query_text)
    return _vector_search(query_embedding, top_k)


def query_by_image(image_path: str, model, processor, top_k: int = 5) -> list[dict]:
    """Find visually similar pages to a query image."""
    image = Image.open(image_path).convert("RGB")
    query_embedding = embed_image(model, processor, image)
    return _vector_search(query_embedding, top_k)


def _vector_search(query_embedding: list[float], top_k: int) -> list[dict]:
    """Run Atlas Vector Search on pre-indexed page embeddings."""
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": top_k * 10,
                "limit": top_k
            }
        },
        {
            "$project": {
                "doc_id": 1,
                "page_num": 1,
                "file_name": 1,
                "metadata": 1,
                "score": {"$meta": "vectorSearchScore"},
                "image_bytes": 0  # Exclude large bytes from result
            }
        }
    ]
    return list(pages_col.aggregate(pipeline))


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model, processor = load_colpali_model()

    # Index a sample PDF (create a dummy one if none exists)
    sample_pdf = "sample_document.pdf"
    if not Path(sample_pdf).exists():
        print(f"No {sample_pdf} found — creating a dummy PNG to index instead")
        dummy_img = Image.new("RGB", (800, 600), color=(200, 220, 255))
        dummy_img.save("dummy_page.png")
        index_image_file("dummy_page.png", doc_id="demo_doc_001", model=model, processor=processor,
                         metadata={"domain": "demo", "title": "Dummy Document"})
    else:
        n = index_pdf(sample_pdf, doc_id="demo_doc_001", model=model, processor=processor,
                      metadata={"domain": "demo"})
        print(f"Indexed {n} pages from {sample_pdf}")

    # Query
    results = query_by_text("financial results Q4 2025", model, processor, top_k=3)
    print("\nTop matches:")
    for r in results:
        print(f"  doc={r['doc_id']} page={r['page_num']} score={r.get('score', 'N/A'):.4f}")
