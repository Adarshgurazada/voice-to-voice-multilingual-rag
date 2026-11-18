import os

from dotenv import load_dotenv

# Load environment variables from .env file (never committed).
load_dotenv()

# The key is optional at import time so tests, linters and offline tools can
# import this module. Code that talks to OpenAI must call require_openai_key().
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class MissingAPIKeyError(RuntimeError):
    """Raised when an OpenAI call is attempted without OPENAI_API_KEY."""


def require_openai_key() -> str:
    key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not key:
        raise MissingAPIKeyError(
            "Missing OPENAI_API_KEY. Copy .env.example to .env and set your key."
        )
    return key


# --- Model Configuration (OpenAI) ---
# Realtime voice-to-voice model (STT + LLM + TTS in one session).
REALTIME_MODEL_ID = "gpt-realtime-mini"
# Text embeddings (1536-dim).
EMBEDDING_MODEL_ID = "text-embedding-3-small"
EMBEDDING_DIM = 1536
# Vision model for image captioning during indexing.
VISION_MODEL_ID = "gpt-5.4-mini"
# Model used for reranking retrieved candidates.
RERANK_MODEL_ID = "gpt-5.4-mini"
# Model for grounded practice question generation and formative feedback.
QUIZ_MODEL_ID = "gpt-5.4-mini"
# Transcription model used for the realtime input transcription hint.
TRANSCRIBE_MODEL_ID = "gpt-4o-mini-transcribe"
# Model that writes grounded, claim-cited answers in the text pipeline.
ANSWER_MODEL_ID = "gpt-5.4-mini"
# Model that checks, before an answer is shown, that each quoted claim is really supported.
SUPPORT_MODEL_ID = "gpt-5.4-mini"
# Stronger, different model used only to judge answers during evaluation (independent of the
# in-pipeline checker so the evaluation does not simply grade the system's own opinion).
JUDGE_MODEL_ID = "gpt-5.5"
