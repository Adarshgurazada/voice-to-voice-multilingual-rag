import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Environment Variable Validation ---
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_REGION = os.getenv("GOOGLE_CLOUD_REGION") # Often 'us-central1' for Vertex
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Needed if not using Vertex auth
USE_VERTEX_AI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "True").lower() == "true"

# --- Model Configuration ---
# This specific model is required for the Live API preview
LIVE_MODEL_ID = "gemini-2.0-flash-live-preview-04-09"

# --- Error Handling for Missing Variables ---
missing_vars = []
if not GOOGLE_CLOUD_PROJECT:
    missing_vars.append("GOOGLE_CLOUD_PROJECT")
if not GOOGLE_CLOUD_REGION:
    missing_vars.append("GOOGLE_CLOUD_REGION")
# Depending on auth method, API key might not be strictly needed if using Vertex gcloud auth
# if not GEMINI_API_KEY and not USE_VERTEX_AI:
#     missing_vars.append("GEMINI_API_KEY (required if not using Vertex AI auth)")

if missing_vars:
    print("Error: Missing required environment variables in .env file:", file=sys.stderr)
    for var in missing_vars:
        print(f"- {var}", file=sys.stderr)
    print("\nPlease ensure your .env file is correctly set up.", file=sys.stderr)
    sys.exit(1)

# --- Optional: Print Loaded Config (for debugging) ---
# print("--- Configuration Loaded ---")
# print(f"Project ID: {GOOGLE_CLOUD_PROJECT}")
# print(f"Region: {GOOGLE_CLOUD_REGION}")
# print(f"Using Vertex AI: {USE_VERTEX_AI}")
# print(f"Live Model ID: {LIVE_MODEL_ID}")
# print(f"API Key Provided: {'Yes' if GEMINI_API_KEY else 'No'}")
# print("--------------------------")

# --- Set Environment Variables for SDK ---
# The SDK looks for these specific environment variables
if GOOGLE_CLOUD_PROJECT:
    os.environ["GOOGLE_CLOUD_PROJECT"] = GOOGLE_CLOUD_PROJECT
if GOOGLE_CLOUD_REGION:
    os.environ["GOOGLE_CLOUD_LOCATION"] = GOOGLE_CLOUD_REGION # Note the SDK uses LOCATION
if USE_VERTEX_AI:
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
# If you are using Application Default Credentials (gcloud auth application-default login),
# you might not need to explicitly set the API key for Vertex AI usage.
# If using the Gemini API directly (not via Vertex), the API key is primary.
if GEMINI_API_KEY:
     os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY