import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Environment Variable Validation ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("Error: Missing required environment variable OPENAI_API_KEY.", file=sys.stderr)
    print("Set OPENAI_API_KEY in your .env file.", file=sys.stderr)
    sys.exit(1)

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
# Transcription model used for the realtime input transcription hint.
TRANSCRIBE_MODEL_ID = "gpt-4o-mini-transcribe"
