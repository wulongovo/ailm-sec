"""Document Ingestion - Load and process security documents."""

import json
import os
from lib.logger import log


class DocumentIngestor:
    """Ingest security documents into the knowledge base."""

    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def ingest_cve_jsonl(self, filepath):
        """Ingest CVE data from JSONL file."""
        if not os.path.exists(filepath):
            log.warning(f"File not found: {filepath}")
            return 0

        texts = []
        metadatas = []
        count = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    text = self._format_cve(item)
                    texts.append(text)
                    metadatas.append({
                        "source": "cve",
                        "id": item.get("id", ""),
                        "severity": item.get("severity", ""),
                        "category": item.get("category", ""),
                    })
                    count += 1
                except json.JSONDecodeError:
                    continue

        if texts:
            self.kb.add_documents(texts, metadatas)
        log.info(f"Ingested {count} CVE entries from {filepath}")
        return count

    def ingest_qa_jsonl(self, filepath):
        """Ingest Q&A training data into knowledge base."""
        if not os.path.exists(filepath):
            log.warning(f"File not found: {filepath}")
            return 0

        texts = []
        metadatas = []
        count = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    q = item.get("question", "")
                    a = item.get("answer", "")
                    if q and a:
                        text = f"Q: {q}\n\nA: {a}"
                        texts.append(text)
                        metadatas.append({
                            "source": "qa",
                            "category": item.get("category", ""),
                            "difficulty": item.get("difficulty", ""),
                        })
                        count += 1
                except json.JSONDecodeError:
                    continue

        if texts:
            self.kb.add_documents(texts, metadatas)
        log.info(f"Ingested {count} Q&A entries from {filepath}")
        return count

    def ingest_markdown(self, filepath):
        """Ingest a markdown file."""
        if not os.path.exists(filepath):
            return 0

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Split by headers
        sections = []
        current = ""
        for line in content.split("\n"):
            if line.startswith("#") and current:
                sections.append(current.strip())
                current = line + "\n"
            else:
                current += line + "\n"
        if current.strip():
            sections.append(current.strip())

        if sections:
            self.kb.add_documents(sections, [{"source": "markdown", "file": filepath}] * len(sections))
        log.info(f"Ingested {len(sections)} sections from {filepath}")
        return len(sections)

    def _format_cve(self, item):
        """Format CVE item into searchable text."""
        parts = []
        if item.get("id"):
            parts.append(f"CVE ID: {item['id']}")
        if item.get("description"):
            parts.append(f"Description: {item['description']}")
        if item.get("severity"):
            parts.append(f"Severity: {item['severity']}")
        if item.get("affected"):
            parts.append(f"Affected: {item['affected']}")
        if item.get("solution"):
            parts.append(f"Solution: {item['solution']}")
        return "\n".join(parts)
