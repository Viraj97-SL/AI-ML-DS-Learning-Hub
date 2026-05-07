<div align="center">

# 📄 Paper Digest: ColPali — Efficient Document Retrieval with Vision Language Models

**arXiv:2407.01449 · July 2024**

</div>

---

## One-Line Summary
Index PDF pages as visual embeddings using a VLM — no OCR, no layout extraction, no text parsing — and retrieve them with text queries that match charts, tables, figures, and scanned documents.

## Problem It Solves
Standard RAG pipelines assume text: extract text from PDF → chunk → embed → store. This fails for scanned documents, tables, charts, engineering schematics, medical images, and any document where the layout is meaningful. ColPali skips text entirely: it treats each PDF page as an image and produces a rich embedding that captures both visual and textual content.

## Key Contributions

| Contribution | Detail |
|--------------|--------|
| **PaliGemma backbone** | Uses Google's PaliGemma VLM to produce page-level embeddings |
| **Late interaction** | Multi-vector representation (one vector per image patch) enables fine-grained matching |
| **No preprocessing** | Eliminates OCR, layout detection, table parsing pipelines |
| **Text-to-page retrieval** | Text query matched against visual page embeddings |
| **ViDoRe benchmark** | New benchmark for visually-rich document retrieval — ColPali sets SOTA |

## How It Works

```
PDF Page (image) → PaliGemma VLM → 128 patch embeddings (one per image region)
Text Query → Query encoder → Query embedding
Match: MaxSim(query_embedding, patch_embeddings) → relevance score
```

## MongoDB Atlas Integration

```python
from colpali_engine.models import ColPali, ColPaliProcessor

model = ColPali.from_pretrained("vidore/colpali-v1.2")

def index_page(page_image, doc_id: str, page_num: int):
    embedding = model.encode_images([page_image])[0]  # 128-dim vector
    db.pages.insert_one({
        "doc_id": doc_id,
        "page_num": page_num,
        "embedding": embedding.tolist(),
        "image_s3_path": f"s3://bucket/{doc_id}/{page_num}.png"
    })

# Atlas Vector Search index: 128 dimensions, cosine similarity
def query(text: str, top_k: int = 5) -> list:
    query_embedding = model.encode_queries([text])[0].tolist()
    return list(db.pages.aggregate([{
        "$vectorSearch": {
            "index": "colpali_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": top_k * 10,
            "limit": top_k
        }
    }]))
```

## When to Use ColPali vs Standard RAG

| Situation | Use |
|-----------|-----|
| Clean digital PDFs with extractable text | Standard RAG (faster, cheaper) |
| Scanned PDFs, handwritten documents | **ColPali** |
| Documents with important tables/charts | **ColPali** |
| Medical images, X-rays, pathology slides | **ColPali** |
| Satellite/aerial imagery | **ColPali** |
| Engineering schematics, P&IDs | **ColPali** |
| Mixed documents (some pages text, some charts) | **ColPali** (handles both) |

## Benchmark Results (ViDoRe)
- ColPali: 81.3 nDCG@5 (vs 52.4 for best text-based baseline)
- 36% improvement over OCR + embedding pipeline on visually-rich documents

## Relevant Hackathon Ideas
Theme 3: #79 ColPriorArt, #83 SatelliteMind, #88 ManuscriptVision, #96 RadioAtlas, Blueprint 01 Viral Autopsy, Blueprint 08 Exodus Mapper

---

*See also: [Search-R1](search_r1.md) · [HippoRAG](hipporag.md)*
