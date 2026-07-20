"""RAGAS wiring: judge LLM + embeddings.

- Judge LLM: selected by JUDGE_MODEL. A Claude model (default `claude-sonnet-4-6`)
  runs via the Anthropic API — reliable structured output for faithfulness/relevancy,
  and a different family from the llama3.1 generator (no self-preference bias). Any
  other value falls back to that Ollama model locally (e.g. qwen2.5vl:3b).
  Embeddings stay LOCAL (our sentence-transformers model) either way, so no document
  text is embedded in the cloud.
- Cloud judging needs ANTHROPIC_API_KEY in the environment.
"""
import os
import sys
from pathlib import Path

from langchain_core.embeddings import Embeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig

# rag/ on path so we can reuse the exact embedding model used to build the index.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from embed import embed  # noqa: E402

# Default to Claude Sonnet 4.6 as the judge (fast, reliable JSON). Set JUDGE_MODEL to an
# Ollama tag (e.g. qwen2.5vl:3b) to judge locally instead.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6")
_IS_CLAUDE = JUDGE_MODEL.startswith("claude")

# Cloud judge is fast + concurrency-safe, so allow parallel workers; local stays single.
RUN_CONFIG = (RunConfig(timeout=180, max_workers=8, max_retries=3) if _IS_CLAUDE
              else RunConfig(timeout=600, max_workers=1, max_retries=2))


class LocalEmbeddings(Embeddings):
    """Adapts rag/embed.py to the LangChain Embeddings interface for RAGAS."""

    def embed_documents(self, texts):
        return embed(list(texts)).tolist()

    def embed_query(self, text):
        return embed([text])[0].tolist()


def get_judge_llm():
    if _IS_CLAUDE:
        from langchain_anthropic import ChatAnthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set — export it before running the cloud judge.")
        return LangchainLLMWrapper(ChatAnthropic(model=JUDGE_MODEL, temperature=0, max_tokens=1024))
    from langchain_ollama import ChatOllama
    return LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, temperature=0, num_ctx=8192))


def get_eval_embeddings():
    return LangchainEmbeddingsWrapper(LocalEmbeddings())
