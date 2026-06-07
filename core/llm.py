"""LLM Engine - Chat completion with RAG context."""

import os
from lib.logger import log

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


SYSTEM_PROMPT = """You are a senior cybersecurity expert assistant. You help security professionals with:
- Vulnerability analysis and remediation
- Penetration testing guidance
- Security architecture review
- Incident response
- Threat intelligence

Always provide accurate, actionable advice. Include code examples when relevant.
If you reference CVE IDs, provide the full description and remediation steps.
Respond in the same language as the user's question."""


class LLMEngine:
    """LLM engine with RAG context injection."""

    def __init__(self):
        self.provider = os.environ.get("LLM_PROVIDER", "ollama")
        self.api_key = os.environ.get("LLM_API_KEY", "ollama")
        self.base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
        self.model = os.environ.get("LLM_MODEL", "qwen2.5")
        self.client = None
        self._init_client()

    def _init_client(self):
        if not HAS_OPENAI:
            log.warning("openai package not installed")
            return
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        log.info(f"LLM: {self.provider}/{self.model} at {self.base_url}")

    def chat(self, user_message, context_docs=None, stream=False):
        """Chat with optional RAG context.

        Args:
            user_message: User's question
            context_docs: List of retrieved documents from RAG
            stream: Whether to stream the response

        Returns:
            str or generator: Response text
        """
        if not self.client:
            return "LLM service not available. Please configure LLM_PROVIDER and LLM_API_KEY."

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Inject RAG context
        if context_docs:
            context_text = "\n\n".join([
                f"[Reference {i+1}] {doc.get('content', '')}"
                for i, doc in enumerate(context_docs)
            ])
            messages.append({
                "role": "system",
                "content": f"Relevant references from knowledge base:\n\n{context_text}"
            })

        messages.append({"role": "user", "content": user_message})

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
                stream=stream,
            )
            if stream:
                return self._stream_response(resp)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            log.error(f"LLM error: {e}")
            return f"LLM request failed: {str(e)}"

    def _stream_response(self, resp):
        """Generator for streaming responses."""
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
