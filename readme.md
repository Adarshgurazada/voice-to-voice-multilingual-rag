# Voice-to-Voice Multilingual Multimodal RAG

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

This repository contains a Python application demonstrating a real-time, voice-driven Retrieval-Augmented Generation (RAG) system. It allows users to ask questions by voice, have the system retrieve relevant information from provided PDF documents (including text and image content), and receive a spoken answer in the chosen language. The system runs entirely on OpenAI services: a single OpenAI Realtime `gpt-realtime-mini` session handles speech-to-text, reasoning, and text-to-speech, while `text-embedding-3-small` embeddings and `gpt-5.4-mini` vision/reranking power the multimodal retrieval, backed by FAISS for fast vector search.

**GitHub Repository:** [https://github.com/Adarshgurazada/voice-to-voice-multilingual-rag.git](https://github.com/Adarshgurazada/voice-to-voice-multilingual-rag.git)

## Key Features

*   **Real-time Voice Interaction:** Speak your questions naturally via your microphone and hear the answers spoken back — all through one OpenAI Realtime session that performs speech-to-text, reasoning, and text-to-speech end to end.
*   **Multimodal RAG:**
    *   Extracts both text and images from PDF documents.
    *   Generates textual descriptions for images using the `gpt-5.4-mini` vision model.
    *   Creates vector embeddings for both text chunks *and* image descriptions using OpenAI's `text-embedding-3-small` model (1536-dimensional). This represents both text and image content in the same vector space — the core of the **multimodal** capability.
    *   Uses improved overlapping, word-boundary chunking so passages stay coherent across chunk borders.
    *   Retrieves the most relevant text snippets *or* image descriptions based on semantic similarity to your spoken query.
*   **Over-fetch + Rerank Retrieval:** FAISS over-fetches roughly 20 candidate passages, then `gpt-5.4-mini` reranks them down to the top 5 most relevant results (with a graceful fallback to FAISS order if reranking is unavailable or times out).
*   **Grounded Answers via Tool Calling:** The Realtime session is given a `search_documents` tool. Each turn the model calls the tool, the app runs FAISS + rerank, and returns grounded context that the model then speaks — so answers stay tied to your documents.
*   **Multilingual Support:**
    *   Transcribes user speech and generates spoken responses in multiple languages (e.g., `en-US`, `hi-IN`, `ta-IN`, `te-IN`, `bn-IN`, `gu-IN`, `mr-IN`, `kn-IN`, `ml-IN`).
    *   Choose from a range of OpenAI Realtime output voices.
*   **Efficient Retrieval:** Utilizes FAISS (Facebook AI Similarity Search) for fast and scalable similarity searches over the document embeddings.
*   **PDF Processing:** Handles single or multiple PDF documents as the knowledge base.
*   **Asynchronous Architecture:** Uses Python's `asyncio` for efficient handling of audio I/O, API calls, and processing, enabling a more responsive experience.
*   **Persistent Indexing:** Saves the generated FAISS index and metadata locally, allowing for faster startup on subsequent runs with the same documents.

## How It Works (Workflow)

The application follows these steps:

1.  **Initialization & RAG Setup (First Run or New PDFs):**
    *   Takes one or more PDF file paths as input.
    *   **Checks for Existing Index:** For each PDF, it looks for corresponding `.faissindex` and `.meta.parquet` files. If found and valid (including a 1536-dimension check), it loads the pre-computed embeddings and metadata.
    *   **PDF Parsing:** If no index exists, it opens the PDF using `PyMuPDF` (`fitz`).
    *   **Text Extraction & Chunking:** Extracts text content page by page and splits it into overlapping, word-boundary chunks so context is preserved across chunk edges.
    *   **Image Extraction & Description:** Extracts images from each page. For significant images, it uses the `gpt-5.4-mini` vision model to generate a detailed textual description (e.g., summarizing charts, describing scenes, extracting table data). Images in formats the vision API can't read are converted (via Pillow) before captioning.
    *   **Embedding Generation:** Generates 1536-dimensional vector embeddings for each text chunk and each *image description* using OpenAI's `text-embedding-3-small` model.
    *   **Metadata Storage:** Stores metadata for each embedding (source document, page number, content type — text/image, the actual text chunk or image description, and image path if applicable) in a Pandas DataFrame.
    *   **FAISS Indexing:** Builds a FAISS `IndexFlatIP` (Inner Product similarity) using the generated embeddings (normalized L2 vectors).
    *   **Persistence:** Saves the FAISS index (`.faissindex`) and the metadata DataFrame (`.meta.parquet`) to disk, named after the source PDF.
    *   **Combination:** Loads/combines the indexes and metadata from all provided PDFs into a single, unified FAISS index and metadata DataFrame for querying.

2.  **Real-time Interaction Loop:**
    *   **Connect to OpenAI Realtime:** Establishes a Realtime session with the `gpt-realtime-mini` model via the `AsyncOpenAI` SDK (`realtime.connect`). Configures the output language, voice, grounding instructions, and the `search_documents` tool.
    *   **User Voice Input (Push-to-Talk):** Interaction is half-duplex push-to-talk. Press `Enter` to start recording from the microphone, `Enter` again to stop. Mic audio is captured as 24kHz PCM16 mono. Recording then playing (rather than full-duplex) avoids the model hearing its own output since there is no acoustic echo cancellation.
    *   **Speech-to-Text (in-session):** The captured audio is streamed to the same Realtime session, which transcribes it — no separate transcription API call.
    *   **RAG Retrieval via Tool Call:** The model calls the `search_documents` tool with a query. The app embeds the query with `text-embedding-3-small`, FAISS over-fetches ~20 candidates, `gpt-5.4-mini` reranks to the top 5, and the grounded context is returned to the session as the tool result.
    *   **Grounded Spoken Response:** With the retrieved context in hand, the model generates and streams back the spoken answer, constrained to answer from the retrieved documents.
    *   **Audio Playback:** Plays the streamed 24kHz PCM16 audio chunks back to the user using `PyAudio`.
    *   **Display Transcription:** Prints the transcript of your question and of the model's spoken response.
    *   **Loop:** Waits for the user to press Enter to ask another question. Press `Ctrl+C` to exit gracefully.

## Technology Stack

*   **Programming Language:** Python 3.11+ (the app uses `asyncio.TaskGroup` and `except*`, which require 3.11+).
*   **Core AI/ML (OpenAI):**
    *   **OpenAI Realtime API (`gpt-realtime-mini`):** A single session doing speech-to-text, reasoning, and text-to-speech, driven via the `AsyncOpenAI` SDK's `realtime.connect`.
    *   **Text Embeddings (`text-embedding-3-small`):** 1536-dimensional embeddings for text chunks and image descriptions.
    *   **Vision (`gpt-5.4-mini`):** Generates textual descriptions of extracted images.
    *   **Reranking (`gpt-5.4-mini`):** Reranks FAISS candidates down to the top 5 most relevant passages.
    *   **FAISS (`faiss-cpu`):** For efficient vector similarity search.
*   **Document Processing:**
    *   **PyMuPDF (`fitz`):** For parsing PDF files, extracting text and images.
    *   **Pillow (`PIL`):** For image handling and converting unsupported image formats before captioning.
    *   **Pandas (`pandas`):** For managing metadata associated with embeddings.
    *   **PyArrow (`pyarrow`):** Required by Pandas for efficient Parquet file I/O (metadata storage).
*   **Audio Handling:**
    *   **PyAudio (`pyaudio`):** For recording 24kHz PCM16 microphone input and playing audio output.
*   **Concurrency:**
    *   **Asyncio:** Python's built-in library for asynchronous programming.
*   **Configuration:**
    *   **Python-Dotenv (`python-dotenv`):** For loading the `OPENAI_API_KEY` from a `.env` file.
*   **Numerical Computation:**
    *   **NumPy (`numpy`):** For numerical operations, especially on embeddings.

## Prerequisites

1.  **Python 3.11+:** The application relies on `asyncio.TaskGroup` and `except*`, which are only available in Python 3.11 and newer.
2.  **OpenAI API Key:** You need an OpenAI API key with access to the Realtime, embeddings, and vision-capable chat models used above.
3.  **Audio Setup:** Ensure your microphone and speakers are correctly configured on your system. You might need to install system libraries for `pyaudio` (like `portaudio`). See the [PyAudio documentation](http://people.csail.mit.edu/hubert/pyaudio/) for platform-specific requirements if you encounter installation issues.

## Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Adarshgurazada/voice-to-voice-multilingual-rag.git
    cd voice-to-voice-multilingual-rag
    ```

2.  **Create a Virtual Environment:** (Recommended)
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If `pip install pyaudio` fails, you might need to install `portaudio` development libraries first. On Debian/Ubuntu: `sudo apt-get install portaudio19-dev python3-dev`. On macOS (using Homebrew): `brew install portaudio`.*

4.  **Configure Environment Variables:**
    *   Create a file named `.env` in the root directory of the project.
    *   Add your OpenAI API key:

    ```dotenv
    # .env file
    OPENAI_API_KEY=sk-...
    ```

    This is the only configuration the app requires.

## Usage

Run the main script from your terminal, providing the path(s) to the PDF document(s) you want to query.

```bash
python module_v2v_rag.py path/to/your/document1.pdf [path/to/another/document2.pdf ...] [--language <lang_code>] [--voice <voice_name>]
```

**Arguments:**

*   `pdf_paths`: (Required) One or more paths to PDF files.
*   `--language`: (Optional) The BCP-47 language code for both input transcription and output speech (e.g., `en-US`, `hi-IN`, `ta-IN`). Defaults to `hi-IN`. The code is mapped internally to an ISO-639-1 code (e.g. `hi-IN` → `hi`) for the transcription language hint. See `SUPPORTED_LANGUAGES` in the script for options.
*   `--voice`: (Optional) The OpenAI Realtime output voice. Defaults to `alloy`. Available voices: `alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`, `shimmer`, `verse`, `marin`, `cedar`.
*   `--selfcheck`: (Optional) Prints the Realtime session payload as JSON and exits. Requires no microphone and makes no API calls (beyond importing the module) — useful for verifying session configuration.
*   `--rebuild`: (Optional) Rebuilds the FAISS indexes from the given PDFs, then exits. No microphone is used. Use this to regenerate indexes (e.g. after upgrading embeddings).

**Interaction:**

1.  The application will initialize, process the PDFs (building/loading the index). This might take time on the first run for large documents.
2.  You will see the message: `>>> Press ENTER to record, ENTER again to stop. (Ctrl+C to quit) <<<`.
3.  Press `Enter`. The application will start recording audio from your microphone (`[Recording... Press ENTER to stop]`).
4.  Speak your question clearly.
5.  Press `Enter` again to stop recording.
6.  The application streams your audio to the Realtime session, which transcribes it; the model calls `search_documents`, the app runs FAISS + rerank, and the model speaks a grounded answer, which is played back. Your question and the response transcript are printed.
7.  The prompt `>>> Press ENTER to record... <<<` will reappear, allowing you to ask another question.
8.  Press `Ctrl+C` to exit the application gracefully.

## Development / Testing

Install the development dependencies and run the test suite:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Future Improvements

*   **Web Frontend:** Create a user-friendly web interface (using Flask, Streamlit, Gradio, etc.) to replace the command-line interaction, allowing easier PDF upload, language/voice selection, and chat history display.
*   **Enhanced RAG Strategies:**
    *   Explore query expansion techniques.
*   **Support More Document Types:** Extend parsing capabilities beyond PDFs to include `.docx`, `.html`, `.txt`, etc.
*   **GPU Acceleration:** Utilize `faiss-gpu` if a compatible GPU is available for even faster indexing and searching on very large datasets.
*   **Improved Error Handling:** Add more robust error handling and user feedback for API issues, file processing errors, or audio device problems.
*   **Caching:** Implement caching for embeddings and potentially image descriptions to speed up reprocessing.
*   **Containerization:** Package the application using Docker for easier deployment and dependency management.
*   **Conversation History:** Allow the model to consider previous turns in the conversation for more contextually relevant answers.
*   **VAD (Voice Activity Detection)**: I Couldn't get the VAD to work on this setup. Maybe it is my incapability but feel free to add VAD to this project for an ever better seemless experience while talking to your documents. 

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.
