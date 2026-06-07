"""RAG Knowledge Base - Vector store + retrieval pipeline."""

import os
from lib.logger import log

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


class KnowledgeBase:
    """Security knowledge base with RAG retrieval."""

    def __init__(self, embedding_service, persist_dir=None):
        self.embedding = embedding_service
        self.persist_dir = persist_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "vectordb"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", ".", " "],
        )
        self.vectorstore = None
        self.retriever = None
        self._init_vectorstore()

    def _init_vectorstore(self):
        if not HAS_LANGCHAIN:
            log.warning("LangChain not installed, knowledge base disabled")
            return

        embeddings = self.embedding.get_langchain_embeddings()
        os.makedirs(self.persist_dir, exist_ok=True)

        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=embeddings,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )
        doc_count = self.vectorstore._collection.count()
        log.info(f"Knowledge base loaded: {doc_count} documents at {self.persist_dir}")

    def add_documents(self, texts, metadatas=None):
        """Add documents to the knowledge base."""
        if not self.vectorstore:
            log.warning("Vectorstore not initialized")
            return 0

        chunks = self.text_splitter.split_text("\n\n".join(texts))
        chunk_metadatas = metadatas * (len(chunks) // len(metadatas) + 1) if metadatas else None
        chunk_metadatas = chunk_metadatas[:len(chunks)] if chunk_metadatas else None

        self.vectorstore.add_texts(chunks, metadatas=chunk_metadatas)
        log.info(f"Added {len(chunks)} chunks to knowledge base")
        return len(chunks)

    def query(self, question, k=5):
        """Retrieve relevant documents for a question."""
        if not self.retriever:
            return []
        try:
            docs = self.retriever.get_relevant_documents(question)
            return [{"content": d.page_content, "metadata": d.metadata} for d in docs[:k]]
        except Exception as e:
            log.error(f"Retrieval error: {e}")
            return []

    def get_stats(self):
        """Return knowledge base statistics."""
        if not self.vectorstore:
            return {"status": "not initialized", "documents": 0}
        count = self.vectorstore._collection.count()
        return {"status": "ready", "documents": count, "persist_dir": self.persist_dir}
