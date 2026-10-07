"""Index a folder of PDFs into Milvus for the agentic RAG layer.

The same collection is read by voxclinbench_app.py (RAG pages) and by the
Milvus RAG tools in agents/langflow/clinical_reasoning_multiagent.json, which
filter on paper_type (Pharmacology, Clinical Guidelines, Rehabilitation,
Comorbidity).

    export MILVUS_URI=http://localhost:19530 GEMINI_API_KEY=...
    python scripts/index_papers.py --pdf_dir papers/pharma --paper_type Pharmacology
    python scripts/index_papers.py --pdf_dir papers/guidelines --paper_type "Clinical Guidelines"

Env: MILVUS_URI, MILVUS_TOKEN (optional), MILVUS_COLLECTION, GEMINI_API_KEY,
EMBED_DEVICE (optional: cpu / cuda / mps).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tcdann"))
import indexer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf_dir", required=True, help="Folder of PDFs (searched recursively)")
    ap.add_argument("--paper_type", default="Other", help="Category label stored with every chunk")
    ap.add_argument("--collection", default=indexer.DEFAULT_COLLECTION_NAME)
    ap.add_argument("--index_type", default="HNSW", choices=indexer.INDEX_TYPES)
    ap.add_argument("--rebuild", action="store_true", help="Drop and recreate the collection first")
    args = ap.parse_args()

    pdfs = sorted(Path(args.pdf_dir).rglob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found under {args.pdf_dir}")

    idx = indexer.PDFIndexer(index_type=args.index_type, collection_name=args.collection,
                             drop_old_collection=args.rebuild)
    all_chunks = []
    for pdf in pdfs:
        chunks = idx.embed_chunks(idx.extract_and_chunk(str(pdf), paper_type=args.paper_type))
        all_chunks.extend(chunks)
        print(f"{pdf.name}: {len(chunks)} chunks")
    if not all_chunks:
        sys.exit("No text extracted from the PDFs.")

    if args.rebuild or idx._collection.num_entities == 0:
        idx.build_index(all_chunks)            # creates the ANN index and loads the collection
    else:
        idx._chunks = getattr(idx, "_chunks", [])
        idx.add_to_index(all_chunks)           # append to an existing, already indexed collection
    print(f"Indexed {len(all_chunks)} chunks from {len(pdfs)} PDFs into "
          f"'{args.collection}' ({args.paper_type}).")


if __name__ == "__main__":
    main()
