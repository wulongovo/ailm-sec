#!/usr/bin/env python3
"""CLI tool to ingest data into the knowledge base."""

import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.embeddings import EmbeddingService
from core.rag import KnowledgeBase
from core.ingestion import DocumentIngestor

def main():
    ap = argparse.ArgumentParser(description="Ingest data into AILLM-SEC knowledge base")
    ap.add_argument("--cve", help="Path to CVE JSONL file")
    ap.add_argument("--qa", help="Path to Q&A JSONL file")
    ap.add_argument("--markdown", help="Path to markdown file")
    ap.add_argument("--dir", help="Ingest all files in a directory")
    args = ap.parse_args()

    embedding = EmbeddingService()
    kb = KnowledgeBase(embedding)
    ingestor = DocumentIngestor(kb)

    total = 0
    if args.cve:
        total += ingestor.ingest_cve_jsonl(args.cve)
    if args.qa:
        total += ingestor.ingest_qa_jsonl(args.qa)
    if args.markdown:
        total += ingestor.ingest_markdown(args.markdown)
    if args.dir:
        for f in os.listdir(args.dir):
            fp = os.path.join(args.dir, f)
            if f.endswith(".jsonl"):
                if "cve" in f.lower():
                    total += ingestor.ingest_cve_jsonl(fp)
                else:
                    total += ingestor.ingest_qa_jsonl(fp)
            elif f.endswith((".md", ".markdown")):
                total += ingestor.ingest_markdown(fp)

    print(f"\nTotal ingested: {total} documents")
    print(f"Knowledge base stats: {kb.get_stats()}")

if __name__ == "__main__":
    main()
