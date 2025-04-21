# module_v2v_rag.py

import asyncio
import sys
import os
import pyaudio
import argparse
import wave
import tempfile
import pandas as pd
import fitz # PyMuPDF for image extraction
from PIL import Image as PIL_Image
import io
from pathlib import Path
import time # For simple delays if needed

# --- FAISS ---
try:
    import faiss
except ImportError:
    print("ERROR: faiss-cpu (or faiss-gpu) not found. Install: pip install faiss-cpu")
    sys.exit(1)
try:
    import pyarrow # Required by pandas for parquet
except ImportError:
     print("ERROR: pyarrow not found. Install: pip install pyarrow")
     sys.exit(1)

from sklearn.metrics.pairwise import cosine_similarity # Still used if needed, FAISS handles search
import numpy as np

# Google Cloud & GenAI Clients
from google.cloud import speech
import vertexai
from vertexai.generative_models import GenerativeModel, Image as VertexImage
from google import genai
from google.genai import types
from google.genai.types import ( Content, Part, LiveConnectConfig, HttpOptions, Modality,
    SpeechConfig, VoiceConfig, PrebuiltVoiceConfig, AudioTranscriptionConfig,
    EmbedContentConfig, GenerationConfig )
from google.api_core.exceptions import GoogleAPIError, FailedPrecondition, ResourceExhausted

# Import configuration loader
try:
    from config import LIVE_MODEL_ID, USE_VERTEX_AI, GOOGLE_CLOUD_REGION, GOOGLE_CLOUD_PROJECT
except ImportError: print("Error: Could not import config.py.", file=sys.stderr); sys.exit(1)

# --- Constants ---
INPUT_FORMAT = pyaudio.paInt16; OUTPUT_FORMAT = pyaudio.paInt16; CHANNELS = 1
INPUT_RATE = 16000; OUTPUT_RATE = 24000; CHUNK_SIZE = 1024
RAG_TEXT_CHUNK_SIZE = 512; RAG_TOP_K = 5 # How many total relevant items (text+image)
EMBEDDING_MODEL_ID = "text-embedding-005"
IMAGE_SAVE_DIR = "extracted_images"
IMAGE_DESC_MODEL_ID = "gemini-2.0-flash" 
""
# --- FAISS Index File Suffixes ---
FAISS_INDEX_SUFFIX = ".faissindex"
METADATA_SUFFIX = ".meta.parquet"

# --- Session Configuration ---
DEFAULT_VOICE = "Aoede"; DEFAULT_LANGUAGE = "hi-IN"
SUPPORTED_LANGUAGES = ["en-US", "es-US", "fr-FR", "de-DE", "te-IN", "bn-IN", "kn-IN", "hi-IN", "ml-IN"]
AVAILABLE_VOICES = ["Aoede", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Zephyr"]

# --- Helper Functions (RAG Logic - FAISS Integrated) ---

def get_text_embedding(client: genai.Client, text: str, task_type="RETRIEVAL_DOCUMENT") -> list | None:
    """Generates text embedding using the google-genai client."""
    if not text or text.isspace(): return None
    try:
        embedding_config = types.EmbedContentConfig(task_type=task_type)
        response = client.models.embed_content(model=EMBEDDING_MODEL_ID, contents=[text], config=embedding_config)
        if hasattr(response, 'embeddings') and response.embeddings and hasattr(response.embeddings[0], 'values') and response.embeddings[0].values: return response.embeddings[0].values
        else:
            embedding_dict = getattr(response, 'embedding', None)
            if embedding_dict and isinstance(embedding_dict, dict) and 'values' in embedding_dict: return embedding_dict['values']
            print(f"\nWarning: Could not extract text embedding: {type(response)}\n For text: '{text[:50]}...'", file=sys.stderr); return None
    except ResourceExhausted as e: print(f"\nWarning: Embedding Quota Exceeded: {e}. Retrying may be needed.", file=sys.stderr); time.sleep(5); return None # Simple backoff
    except GoogleAPIError as e: print(f"\nWarning: API Error text embedding: {type(e).__name__} - {e}", file=sys.stderr); return None
    except Exception as e: print(f"\nWarning: Unexpected text embedding error: {type(e).__name__} - {e}", file=sys.stderr); return None

def get_image_description(model: GenerativeModel, image_path: str, prompt: str) -> str | None:
    """Generates text description for an image using vertexai.GenerativeModel."""
    try:
        vertex_image = VertexImage.load_from_file(image_path)
        response = model.generate_content(contents=[prompt, vertex_image])
        if hasattr(response, 'text') and response.text: return response.text.strip()
        else:
            feedback = getattr(response, 'prompt_feedback', None); candidates = getattr(response, 'candidates', [])
            if feedback and getattr(feedback, 'block_reason', None): reason = feedback.block_reason; print(f"\nWarning: Image description BLOCKED. Reason: {reason}", file=sys.stderr); return f"Description blocked: {reason}"
            elif not candidates: print(f"\nWarning: No description candidate for {os.path.basename(image_path)}.", file=sys.stderr); return f"No description candidate"
            else:
                 try: first_candidate_text = candidates[0].content.parts[0].text
                 except (IndexError, AttributeError): first_candidate_text = None
                 if first_candidate_text: print("Warning: Accessed description via candidate."); return first_candidate_text.strip()
                 else: print(f"\nWarning: Empty description text for {os.path.basename(image_path)}.", file=sys.stderr); return f"Empty description generated"
    except FileNotFoundError: print(f"\nError: Image file not found: {image_path}", file=sys.stderr); return f"Error: Image file missing"
    except FailedPrecondition as e: print(f"\nError: FailedPrecondition image desc (API/Billing?): {e}", file=sys.stderr); return f"Error: {e}"
    except ResourceExhausted as e: print(f"\nWarning: Image Desc Quota Exceeded: {e}. Retrying may be needed.", file=sys.stderr); time.sleep(5); return None
    except Exception as e: print(f"\nWarning: Error generating description for {os.path.basename(image_path)}: {type(e).__name__} - {e}", file=sys.stderr); return f"Error: {type(e).__name__}"


# --- FAISS Index Building / Loading ---
def build_or_load_index(pdf_path_str: str, genai_client: genai.Client, image_desc_model: GenerativeModel, image_save_dir: str) -> tuple[list, list, int | None]:
    """
    Processes a single PDF: extracts text/images, embeds, builds FAISS index and metadata.
    Loads from file if index/metadata already exist.
    Returns lists of metadata and embeddings, and the detected dimension.
    """
    pdf_file = Path(pdf_path_str)
    doc_basename = pdf_file.stem
    faiss_filename = Path(f"{doc_basename}{FAISS_INDEX_SUFFIX}")
    metadata_filename = Path(f"{doc_basename}{METADATA_SUFFIX}")

    img_dir_path = Path(image_save_dir); img_dir_path.mkdir(parents=True, exist_ok=True)
    image_description_prompt = "Describe this image in detail. If it's a table, extract the data. If it's a chart, explain its key findings."

    embeddings_list = []
    metadata_list = []
    embedding_dimension = None

    print(f"\nProcessing document: {pdf_file.name}")
    try:
        doc = fitz.open(pdf_path_str); num_pages = len(doc)
        print(f"  - Found {num_pages} pages.")
        for page_num in range(num_pages):
            page = doc.load_page(page_num); page_id = page_num + 1
            print(f"\r  - Processing page {page_id}/{num_pages}...", end="")
            # 1. Process Text
            page_text = page.get_text("text")
            if page_text and not page_text.isspace():
                page_chunks = [page_text[i : i + RAG_TEXT_CHUNK_SIZE] for i in range(0, len(page_text), RAG_TEXT_CHUNK_SIZE)]
                for chunk_num, chunk_text in enumerate(page_chunks):
                    if not chunk_text or chunk_text.isspace(): continue
                    embedding = get_text_embedding(genai_client, chunk_text, task_type="RETRIEVAL_DOCUMENT")
                    if embedding:
                        if embedding_dimension is None: embedding_dimension = len(embedding); print(f"\n  [Detected Embedding Dim: {embedding_dimension}]")
                        elif len(embedding) != embedding_dimension: print(f"\nError: Embed dim mismatch! Skipping.", file=sys.stderr); continue
                        metadata_list.append({ "doc": doc_basename, "page": page_id, "type": "text", "content": chunk_text })
                        embeddings_list.append(np.array(embedding, dtype=np.float32))

            # 2. Process Images
            image_list = page.get_images(full=True)
            if image_list: print(f" Found {len(image_list)} images... ", end="")
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref); image_bytes = base_image["image"]; ext = base_image["ext"]
                    if len(image_bytes) < 5000: continue
                    image_filename = f"{doc_basename}_p{page_id}_img{img_index}.{ext}"; image_path = img_dir_path / image_filename
                    with open(image_path, "wb") as img_file: img_file.write(image_bytes)
                    description = get_image_description(image_desc_model, str(image_path), image_description_prompt)
                    if not description or "Error:" in description or "blocked" in description:
                         print(f"\nWarning: Skipping image {image_filename} (desc issue): {description}", file=sys.stderr); continue
                    # Embed the description text
                    desc_embedding = get_text_embedding(genai_client, description, task_type="RETRIEVAL_DOCUMENT")
                    if desc_embedding:
                        if embedding_dimension is None: embedding_dimension = len(desc_embedding); print(f"\n  [Detected Embedding Dim: {embedding_dimension}]")
                        elif len(desc_embedding) != embedding_dimension: print(f"\nError: Desc embed dim mismatch! Skipping.", file=sys.stderr); continue
                        metadata_list.append({"doc": doc_basename, "page": page_id, "type": "image", "content": description, "image_path": str(image_path)})
                        embeddings_list.append(np.array(desc_embedding, dtype=np.float32))
                except Exception as img_err: print(f"\nError processing image {img_index} on page {page_id}: {img_err}", file=sys.stderr)
        print() # Newline after document
    except Exception as e: print(f"Error processing document {pdf_path_str}: {e}", file=sys.stderr)

    if not metadata_list or not embeddings_list:
         print(f"Error: No content could be processed/embedded for {pdf_file.name}.", file=sys.stderr)
         return [], [], None

    print(f"  - Finished processing {pdf_file.name}. Found {len(metadata_list)} items.")
    return metadata_list, embeddings_list, embedding_dimension

# --- FAISS Retrieval ---
def find_relevant_context_faiss(query: str, faiss_index: faiss.Index, metadata_db: pd.DataFrame, genai_client: genai.Client, top_k: int) -> str:
    """Finds relevant items using FAISS index and returns formatted context."""
    print("Status: Finding relevant document context using FAISS...")
    if faiss_index is None or metadata_db is None or metadata_db.empty: return "No document context available."

    try:
        query_embedding = get_text_embedding(genai_client, query, task_type="RETRIEVAL_QUERY")
        if query_embedding is None: return "Could not process query embedding."

        # Ensure query embedding is normalized and correct shape/dtype for IndexFlatIP
        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding_np)

        print(f" - Searching FAISS index (k={top_k})...")
        distances, indices = faiss_index.search(query_embedding_np, top_k)

        if indices.size == 0 or indices[0][0] < 0: print(" - No relevant items found in index."); return "No relevant context found in documents."

        # Get the indices of valid results
        retrieved_indices = indices[0]
        valid_mask = retrieved_indices != -1
        retrieved_indices = retrieved_indices[valid_mask]

        if len(retrieved_indices) == 0: print(" - No relevant items found in index (all results were invalid)."); return "No relevant context found in documents."

        print(f" - Top {len(retrieved_indices)} relevant items found.")
        relevant_metadata = metadata_db.iloc[retrieved_indices]

        # Format context
        context_parts = []
        for _, row in relevant_metadata.iterrows():
            if row['type'] == 'text':
                context_parts.append(f"[Text Context - Source: {row['doc']}, Pg {row['page']}]\n{row['content']}")
            elif row['type'] == 'image':
                img_name = os.path.basename(row.get('image_path', 'Unknown Image'))
                context_parts.append(f"[Image Context - Source: {row['doc']}, Pg {row['page']}, Image: {img_name}]\nDescription: {row['content']}")
            context_parts.append("---")

        return "\n".join(context_parts)

    except Exception as e: print(f"Error searching FAISS index: {e}", file=sys.stderr); return "Error retrieving context from index."

# --- Main Application Class ---
class VoiceRAGMultimodalFAISS: # Renamed class
    def __init__(self, pdf_paths: list[str], language=DEFAULT_LANGUAGE, voice=DEFAULT_VOICE):
        self.language_code = language; self.selected_voice = voice; self.pdf_paths = pdf_paths
        self.faiss_index = None # Combined FAISS index
        self.metadata_db = None # Combined metadata DataFrame
        self.playback_queue = asyncio.Queue(); self.p_audio = None
        self.output_stream = None; self.sdk_session = None; self.is_active = True
        self.speech_client = None; self.genai_client = None
        self.image_desc_model = None
        self._validate_config(); self._initialize_clients()
        print("--- Gemini Voice RAG (FAISS - Transcribe -> Text+ImgDesc -> Audio) ---") # Updated title
        print(f"Output Voice: {self.selected_voice}, Language: {self.language_code}")
        print(f"Input: Mic ({INPUT_RATE} Hz), Output: Speaker ({OUTPUT_RATE} Hz)")
        print(f"Processing documents: {', '.join(os.path.basename(p) for p in pdf_paths)}")
        print("Status: Initializing...")

    def _validate_config(self):
        # (Validation remains the same)
        if self.selected_voice not in AVAILABLE_VOICES: self.selected_voice = DEFAULT_VOICE
        if self.language_code not in SUPPORTED_LANGUAGES: print(f"Warning: Lang code '{self.language_code}' not known.", file=sys.stderr)
        if not USE_VERTEX_AI: print("Error: Vertex AI required.", file=sys.stderr); sys.exit(1)
        if not GOOGLE_CLOUD_PROJECT or not GOOGLE_CLOUD_REGION: print("Error: Project/Region required.", file=sys.stderr); sys.exit(1)
        os.environ["GOOGLE_CLOUD_PROJECT"] = GOOGLE_CLOUD_PROJECT
        os.environ["GOOGLE_CLOUD_LOCATION"] = GOOGLE_CLOUD_REGION
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

    def _initialize_clients(self):
        """Initializes API clients."""
        self.p_audio = pyaudio.PyAudio()
        try:
            print(f"Status: Initializing Vertex AI SDK for Project: {GOOGLE_CLOUD_PROJECT}, Location: {GOOGLE_CLOUD_REGION}")
            vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_REGION)
            self.speech_client = speech.SpeechClient()
            self.genai_client = genai.Client()
            _ = self.genai_client.models.list()
            print(f"Status: Initializing Image Description Model: {IMAGE_DESC_MODEL_ID}")
            self.image_desc_model = GenerativeModel(IMAGE_DESC_MODEL_ID)
            print("Status: Cloud Speech-to-Text, GenAI (Live), and Vertex Gen Model clients initialized.")
        except Exception as e: print(f"Error initializing clients: {e}", file=sys.stderr); sys.exit(1)

    async def setup_rag_database(self):
        """Loads or builds FAISS index and metadata for all specified PDFs."""
        if not self.genai_client or not self.image_desc_model: print("Error: Clients not initialized.", file=sys.stderr); self.is_active = False; return

        all_metadata = []
        all_embeddings = []
        final_dimension = None

        for pdf_path in self.pdf_paths:
            base_name = Path(pdf_path).stem
            faiss_file = Path(f"{base_name}{FAISS_INDEX_SUFFIX}")
            meta_file = Path(f"{base_name}{METADATA_SUFFIX}")

            if faiss_file.exists() and meta_file.exists():
                print(f"Status: Loading existing index for {os.path.basename(pdf_path)}...")
                try:
                    loaded_index = faiss.read_index(str(faiss_file))
                    loaded_meta = pd.read_parquet(meta_file)
                    if loaded_index.ntotal != len(loaded_meta):
                        raise ValueError(f"Index/metadata count mismatch for {pdf_path} ({loaded_index.ntotal} vs {len(loaded_meta)})")
                    if final_dimension is None:
                        final_dimension = loaded_index.d
                    elif loaded_index.d != final_dimension:
                         raise ValueError(f"Inconsistent embedding dimensions found ({loaded_index.d} vs {final_dimension})")

                    # Extract embeddings from loaded index (needed for combining)
                    loaded_embeddings = loaded_index.reconstruct_n(0, loaded_index.ntotal) # Reconstruct requires index to be trained first? No, reconstructs vectors directly.

                    all_metadata.append(loaded_meta)
                    all_embeddings.append(loaded_embeddings)
                    print(f"  - Loaded {loaded_index.ntotal} vectors.")
                    continue # Skip building if loading succeeded
                except Exception as e:
                    print(f"Warning: Failed to load index for {pdf_path}: {e}. Rebuilding...", file=sys.stderr)
                    # Clean up potentially corrupt files before rebuilding
                    try: faiss_file.unlink(missing_ok=True); meta_file.unlink(missing_ok=True)
                    except OSError: pass

            # Build if loading failed or files don't exist
            print(f"Status: Building index for {os.path.basename(pdf_path)}...")
            meta_list, embeds_list, dim = await asyncio.to_thread(
                 build_or_load_index, pdf_path, self.genai_client, self.image_desc_model, IMAGE_SAVE_DIR )

            if not meta_list or not embeds_list or dim is None:
                 print(f"Warning: Failed to process {pdf_path}, skipping.", file=sys.stderr)
                 continue # Skip this document if processing failed

            if final_dimension is None:
                final_dimension = dim
            elif dim != final_dimension:
                print(f"Error: Inconsistent embedding dimensions ({dim} vs {final_dimension}). Skipping {pdf_path}.", file=sys.stderr)
                continue

            current_meta_df = pd.DataFrame(meta_list)
            current_embeddings_np = np.array(embeds_list).astype('float32')

            # Save individual index/meta files
            try:
                print(f"  - Saving index & metadata for {os.path.basename(pdf_path)}...")
                faiss.normalize_L2(current_embeddings_np) # Normalize before adding/saving
                index_single = faiss.IndexFlatIP(final_dimension)
                index_single.add(current_embeddings_np)
                faiss.write_index(index_single, str(faiss_file))
                current_meta_df.to_parquet(meta_file)
                print("  - Saved successfully.")
            except Exception as e:
                print(f"Error saving index/metadata for {pdf_path}: {e}. Skipping.", file=sys.stderr)
                continue

            all_metadata.append(current_meta_df)
            all_embeddings.append(current_embeddings_np) # Store normalized embeddings

        # Combine all loaded/built data
        if not all_metadata or not all_embeddings:
             print("Fatal Error: No data processed for any PDF.", file=sys.stderr); self.is_active = False; return

        print("\nStatus: Combining data from all documents...")
        self.metadata_db = pd.concat(all_metadata, ignore_index=True)
        combined_embeddings_np = np.vstack(all_embeddings) # Combine numpy arrays

        if final_dimension is None or combined_embeddings_np.shape[1] != final_dimension:
            print("Fatal Error: Final dimension mismatch during combination.", file=sys.stderr); self.is_active = False; return

        print(f"Status: Building final combined FAISS index (Dim: {final_dimension}, Vectors: {combined_embeddings_np.shape[0]})...")
        # Vectors are already normalized from individual saving/loading
        self.faiss_index = faiss.IndexFlatIP(final_dimension)
        self.faiss_index.add(combined_embeddings_np)
        print(f"Status: Combined FAISS index ready ({self.faiss_index.ntotal} total vectors).")

        if self.faiss_index.ntotal != len(self.metadata_db):
             print(f"FATAL ERROR: Final combined index/metadata mismatch! ({self.faiss_index.ntotal} vs {len(self.metadata_db)})", file=sys.stderr)
             self.is_active = False; return


    async def start(self):
        self.is_active = True; await self.setup_rag_database()
        if not self.is_active: print("Exiting due to RAG setup failure."); return
        try:
            async with asyncio.TaskGroup() as tg:
                connect_task = tg.create_task(self.connect_and_manage_session())
                playback_task = tg.create_task(self.play_audio_output())
            print("Status: TaskGroup finished.")
        except* Exception as eg:
            print(f"\nError group exceptions occurred:", file=sys.stderr)
            for i, exc in enumerate(eg.exceptions):
                 if isinstance(exc, asyncio.CancelledError): print(f"- Task cancelled.")
                 else: print(f"- Exception {i+1}/{len(eg.exceptions)}: {type(exc).__name__}: {exc}", file=sys.stderr)
        finally: await self.stop()

    async def stop(self):
        # (stop method remains largely the same)
        if not self.is_active: return
        print("\nStatus: Initiating shutdown..."); self.is_active = False; await asyncio.sleep(0.2)
        # Close output stream first
        if hasattr(self, 'output_stream') and self.output_stream:
            print("Status: Waiting for playback queue...");
            try: await asyncio.wait_for(self.playback_queue.join(), timeout=1.0); print("Status: Playback queue empty.")
            except asyncio.TimeoutError: print("Warning: Timeout waiting playback.", file=sys.stderr)
            except Exception as q_err: print(f"Warning: Error joining queue: {q_err}", file=sys.stderr)
            try:
                if self.output_stream.is_active(): self.output_stream.stop_stream()
                self.output_stream.close(); print("Status: Output stream closed.")
            except Exception as e: print(f"Warning: Error closing output stream: {e}", file=sys.stderr)
        # Input stream closed in record_audio
        if self.p_audio: self.p_audio.terminate(); print("Status: PyAudio terminated.")
        print("--- Gemini Voice RAG Multimodal Finished ---")


    async def connect_and_manage_session(self):
        # Check if RAG DBs and genai_client are ready
        if not self.genai_client or self.metadata_db is None or self.faiss_index is None: # Check combined DBs
             print("Error: GenAI Client or RAG DB not ready.", file=sys.stderr); self.is_active = False; return
        print("Status: Configuring Gemini Live client...");
        system_instruction_text = f"Answer questions based ONLY on provided Context (text & image descriptions). If missing, say so. Respond in {self.language_code}."
        system_instruction = Content(role="system", parts=[Part(text=system_instruction_text)])
        live_api_model_id = "gemini-2.0-flash-live-preview-04-09"
        live_config = LiveConnectConfig(
            response_modalities=[Modality.AUDIO],
            speech_config=SpeechConfig( voice_config=VoiceConfig(prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=self.selected_voice)), language_code=self.language_code,),
            system_instruction=system_instruction, output_audio_transcription=AudioTranscriptionConfig(), )
        print(f"Status: Connecting to Live API model {live_api_model_id}...")
        session = None
        try:
            async with self.genai_client.aio.live.connect(model=live_api_model_id, config=live_config) as session:
                self.sdk_session = session; print("Status: Connected."); print("\n>>> Press ENTER to record, ENTER again to stop. (Ctrl+C to quit) <<<")
                while self.is_active:
                    await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                    if not self.is_active: break
                    audio_frames = await self.record_audio()
                    if not audio_frames or not self.is_active: continue
                    print("Status: Transcribing audio...")
                    transcribed_text = await self.transcribe_audio_data(b"".join(audio_frames))
                    if not transcribed_text or not self.is_active: print("Status: Transcription failed/empty."); print("\n>>> Press ENTER to record... <<<"); continue
                    print(f"   >> You asked (Transcribed): {transcribed_text}", flush=True)
                    # --- RAG Step (FAISS) ---
                    retrieved_context = await asyncio.to_thread(
                         find_relevant_context_faiss, transcribed_text, self.faiss_index, self.metadata_db, self.genai_client, RAG_TOP_K )
                    augmented_prompt = f"Answer based ONLY on the Context provided below.\n\nCONTEXT:\n{retrieved_context}\n\nQUESTION: {transcribed_text}\n\nANSWER:"
                    print("Status: Sending augmented prompt...")
                    if self.sdk_session:
                        try: await self.sdk_session.send_client_content(turns=Content(role="user", parts=[Part(text=augmented_prompt)]))
                        except Exception as send_err: print(f"Error sending prompt: {send_err}", file=sys.stderr); continue
                    print("Status: Waiting for Gemini RAG response...")
                    current_output_transcription = ""; processing_response = False
                    async for message in self.sdk_session.receive():
                        if not self.is_active: break
                        # --- Message Processing Logic (remains the same) ---
                        output_transcription_part = getattr(message, 'output_transcription', None); server_content_part = getattr(message, 'server_content', None)
                        if server_content_part and hasattr(server_content_part, 'model_turn') and server_content_part.model_turn:
                             model_turn = server_content_part.model_turn
                             if hasattr(model_turn, 'parts') and model_turn.parts:
                                 for part in model_turn.parts:
                                     if hasattr(part, 'inline_data') and part.inline_data and hasattr(part.inline_data, 'data') and part.inline_data.data:
                                         await self.playback_queue.put(part.inline_data.data)
                                         if not processing_response: print("   Gemini: [Receiving Audio...]", end='\r', flush=True)
                                         processing_response = True
                        if output_transcription_part and hasattr(output_transcription_part, 'text') and output_transcription_part.text:
                            current_output_transcription += output_transcription_part.text
                            if self.playback_queue.empty(): print(f"   Gemini (Text): {current_output_transcription}", end='\r', flush=True)
                            processing_response = True
                        turn_complete = False
                        if server_content_part and hasattr(server_content_part, 'turn_complete'): turn_complete = server_content_part.turn_complete
                        if turn_complete:
                            print("\nStatus: Gemini turn complete.")
                            if current_output_transcription: print(f"   Gemini (Final Text): {current_output_transcription}   ", flush=True)
                            elif processing_response: print("   Gemini (Audio Only Received)", flush=True)
                            else: print("   Gemini: [No response content received]", flush=True)
                            current_output_transcription = ""; processing_response = False; break
                    if not self.is_active: break
                    print("\n>>> Press ENTER to record... <<<")
        except asyncio.CancelledError: print("Status: Connection/Manage task cancelled.")
        except Exception as e: print(f"\nError in connect_and_manage: {type(e).__name__}: {e}", file=sys.stderr); self.is_active = False; raise
        finally: print("Status: Exiting connect_and_manage loop."); self.sdk_session = None

    # --- record_audio, _read_audio_chunks, transcribe_audio_data methods unchanged ---
    async def record_audio(self):
        if not self.is_active or self.p_audio is None: return None
        frames = []; stream = None; input_stream_obj = None
        print("[Recording... Press ENTER to stop]", end="", flush=True)
        try:
            input_stream_obj = self.p_audio.open(format=INPUT_FORMAT, channels=CHANNELS, rate=INPUT_RATE, input=True, frames_per_buffer=CHUNK_SIZE)
            recording_task = asyncio.create_task(self._read_audio_chunks(input_stream_obj, frames))
            await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            print(" [Stopping Recording...]")
        except OSError as e: print(f"\nError opening input stream: {e}", file=sys.stderr); return None
        except Exception as e: print(f"\nError during recording setup: {e}", file=sys.stderr); return None
        finally:
            if 'recording_task' in locals() and recording_task and not recording_task.done():
                 recording_task.cancel();
                 try: await recording_task
                 except asyncio.CancelledError: pass
            if input_stream_obj:
                try: input_stream_obj.stop_stream(); input_stream_obj.close()
                except Exception as close_e: print(f"Warning: Error closing recording stream: {close_e}", file=sys.stderr)
        return frames if self.is_active else None

    async def _read_audio_chunks(self, stream, frames):
        try:
            while True:
                 data = await asyncio.to_thread(stream.read, CHUNK_SIZE, exception_on_overflow=False)
                 frames.append(data); await asyncio.sleep(0.01)
        except asyncio.CancelledError: pass
        except Exception as e: print(f"Error reading audio chunk: {e}", file=sys.stderr)

    async def transcribe_audio_data(self, audio_data_bytes):
        if not self.speech_client or not audio_data_bytes: return None
        print(f"Status: Transcribing {len(audio_data_bytes)} bytes...")
        try:
            audio = speech.RecognitionAudio(content=audio_data_bytes)
            config = speech.RecognitionConfig( encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=INPUT_RATE, language_code=self.language_code, enable_automatic_punctuation=True, )
            response = await asyncio.to_thread(self.speech_client.recognize, config=config, audio=audio)
            if response.results and response.results[0].alternatives:
                transcript = response.results[0].alternatives[0].transcript.strip()
                print(f"Status: Transcription result: '{transcript}'")
                return transcript
            else: print("Warning: Speech-to-Text returned no results."); return None
        except Exception as e: print(f"Error during transcription: {e}", file=sys.stderr); return None

    # --- play_audio_output method unchanged ---
    async def play_audio_output(self):
        print("Status: Waiting for PyAudio to start output stream...")
        while self.p_audio is None and self.is_active: await asyncio.sleep(0.1)
        if not self.is_active or self.p_audio is None: print("Status: Playback task exiting."); return
        print("Status: Starting audio output stream...")
        try:
            self.output_stream = self.p_audio.open(format=OUTPUT_FORMAT, channels=CHANNELS, rate=OUTPUT_RATE, output=True)
            print("Status: Output stream opened.")
            while self.is_active:
                try:
                    data = await asyncio.wait_for(self.playback_queue.get(), timeout=1.0)
                    if data is None: continue
                    await asyncio.to_thread(self.output_stream.write, data)
                    self.playback_queue.task_done()
                except asyncio.TimeoutError:
                    if not self.is_active: break; continue
                except asyncio.CancelledError: print("Status: Playback task inner cancelled."); break
                except OSError as write_err: print(f"Error writing to output stream: {write_err}", file=sys.stderr); self.is_active = False; break
                except Exception as inner_e: print(f"Error during playback loop: {type(inner_e).__name__}: {inner_e}", file=sys.stderr); self.is_active = False; break
        except asyncio.CancelledError: print("Status: Playback task setup cancelled.")
        except OSError as e: print(f"\nPyAudio OSError in playback: {e}", file=sys.stderr); self.is_active = False
        except Exception as e: print(f"\nUnexpected error in playback: {e}", file=sys.stderr); self.is_active = False; raise
        finally: print("Status: Exiting play_audio_output loop.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini Live API Voice RAG Multimodal w/ FAISS") # Updated description
    parser.add_argument("pdf_paths", nargs='+', help="Path(s) to the PDF document(s) to use for RAG.")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help=f"Input/Output language code. Default: {DEFAULT_LANGUAGE}")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help=f"Output voice name. Default: {DEFAULT_VOICE}")
    args = parser.parse_args()
    valid_paths = [p for p in args.pdf_paths if os.path.isfile(p) and p.lower().endswith(".pdf")]
    if not valid_paths: print("Error: No valid PDF documents provided.", file=sys.stderr); sys.exit(1)
    if len(valid_paths) != len(args.pdf_paths): print(f"Warning: Some invalid paths were skipped.", file=sys.stderr)
    lang_arg, voice_arg = args.language, args.voice
    if lang_arg not in SUPPORTED_LANGUAGES: lang_arg = DEFAULT_LANGUAGE; print(f"Warning: Invalid language '{args.language}', using default.")
    if voice_arg not in AVAILABLE_VOICES: voice_arg = DEFAULT_VOICE; print(f"Warning: Invalid voice '{args.voice}', using default.")
    # Instantiate the multimodal class
    client = VoiceRAGMultimodalFAISS(pdf_paths=valid_paths, language=lang_arg, voice=voice_arg) # Use correct class name
    try:
        print("Starting application...")
        asyncio.run(client.start())
    except KeyboardInterrupt: print("\nExecution cancelled by user (Ctrl+C).")
    except Exception as e: print(f"\nAn unexpected error occurred in main execution: {type(e).__name__}: {e}", file=sys.stderr)
    finally: print("Application exiting.")