# Voice-to-Voice Multilingual Multimodal RAG

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

This repository contains a Python application demonstrating a real-time, voice-driven Retrieval-Augmented Generation (RAG) system. It allows users to ask questions in voice, have the system retrieve relevant information from provided PDF documents (including text and image content), and receive a spoken answer in the chosen language. The system leverages Google Cloud's AI services, specifically the Gemini Live API, Vertex AI, and Cloud Speech-to-Text, along with FAISS for efficient vector search.

**GitHub Repository:** [https://github.com/Adarshgurazada/voice-to-voice-multilingual-rag.git](https://github.com/Adarshgurazada/voice-to-voice-multilingual-rag.git)

## Key Features

*   **Real-time Voice Interaction:** Speak your questions naturally via your microphone and hear the answers spoken back.
*   **Multimodal RAG:**
    *   Extracts both text and images from PDF documents.
    *   Generates textual descriptions for images using a vision model (Gemini Flash).
    *   Creates vector embeddings for both text chunks *and* image descriptions using Google's text embedding models.
    *   Retrieves the most relevant text snippets *or* image descriptions based on the semantic similarity to your spoken query.
*   **Multilingual Support:**
    *   Transcribes user speech and generates spoken responses in multiple languages (e.g., `en-US`, `hi-IN`, `es-US`, `fr-FR`, `de-DE`, `te-IN`, `bn-IN`, `kn-IN`, `ml-IN`).
    *   Allows selection of different output voices provided by the Gemini Live API.
*   **Efficient Retrieval:** Utilizes FAISS (Facebook AI Similarity Search) for fast and scalable similarity searches over the document embeddings.
*   **PDF Processing:** Handles single or multiple PDF documents as the knowledge base.
*   **Asynchronous Architecture:** Uses Python's `asyncio` for efficient handling of audio I/O, API calls, and processing, enabling a more responsive experience.
*   **Persistent Indexing:** Saves the generated FAISS index and metadata locally, allowing for faster startup on subsequent runs with the same documents.

## How It Works (Workflow)

The application follows these steps:

1.  **Initialization & RAG Setup (First Run or New PDFs):**
    *   Takes one or more PDF file paths as input.
    *   **Checks for Existing Index:** For each PDF, it looks for corresponding `.faissindex` and `.meta.parquet` files. If found and valid, it loads the pre-computed embeddings and metadata.
    *   **PDF Parsing:** If no index exists, it opens the PDF using `PyMuPDF` (`fitz`).
    *   **Text Extraction & Chunking:** Extracts text content page by page and splits it into manageable chunks.
    *   **Image Extraction & Description:** Extracts images from each page. For significant images, it uses the Vertex AI Gemini model (`gemini-2.0-flash`) to generate a detailed textual description (e.g., summarizing charts, describing scenes, extracting table data).
    *   **Embedding Generation:** Generates vector embeddings for each text chunk and each *image description* using Google's `text-embedding-005` model via the `google-genai` SDK. This is the core of the **multimodal** capability – representing both text and image content in the same vector space.
    *   **Metadata Storage:** Stores metadata for each embedding (source document, page number, content type - text/image, the actual text chunk or image description, and image path if applicable) in a Pandas DataFrame.
    *   **FAISS Indexing:** Builds a FAISS `IndexFlatIP` (Inner Product similarity) using the generated embeddings (normalized L2 vectors).
    *   **Persistence:** Saves the FAISS index (`.faissindex`) and the metadata DataFrame (`.meta.parquet`) to disk, named after the source PDF.
    *   **Combination:** Loads/combines the indexes and metadata from all provided PDFs into a single, unified FAISS index and metadata DataFrame for querying.

2.  **Real-time Interaction Loop:**
    *   **Connect to Gemini Live:** Establishes a connection to the `gemini-2.0-flash-live-preview-04-09` model using the `google-genai` SDK's `live.connect`. Configures the desired output language, voice, and system instructions.
    *   **User Voice Input:** Waits for the user to press Enter, then records audio from the microphone using `PyAudio`. Pressing Enter again stops recording.
    *   **Speech-to-Text:** Sends the recorded audio data to the Google Cloud Speech-to-Text API (via `google-cloud-speech`) for transcription in the specified input language.
    *   **RAG Retrieval (FAISS Search):**
        *   Generates an embedding for the transcribed user query (using `text-embedding-005` with `task_type="RETRIEVAL_QUERY"`).
        *   Searches the combined FAISS index for the top-K embeddings most similar (highest inner product) to the query embedding.
        *   Retrieves the corresponding metadata (text chunks or image descriptions) for the top results from the Pandas DataFrame.
    *   **Context Augmentation:** Constructs a prompt for the Gemini Live model containing:
        *   A system instruction (e.g., "Answer based only on the provided context...").
        *   The retrieved context (formatted text snippets and/or image descriptions with source information).
        *   The user's transcribed question.
    *   **Send to Gemini Live:** Sends the augmented prompt to the active Gemini Live session.
    *   **Receive Streaming Response:** Receives the response from Gemini Live, which includes:
        *   Streaming audio chunks for the spoken answer.
        *   (Optional) Text transcription of the spoken answer.
    *   **Audio Playback:** Plays the received audio chunks back to the user in real-time using `PyAudio`.
    *   **Display Transcription:** Shows the final text transcription of the model's response (if available).
    *   **Loop:** Waits for the user to press Enter to ask another question.

## Technology Stack

*   **Programming Language:** Python 3.9+
*   **Core AI/ML:**
    *   **Google Generative AI SDK (`google-genai`):** Used for Gemini Live API interaction and Text Embedding generation.
    *   **Vertex AI SDK (`vertexai`):** Used for generating image descriptions via Gemini models.
    *   **Google Cloud Speech-to-Text API (`google-cloud-speech`):** For transcribing user voice input.
    *   **FAISS (`faiss-cpu`):** For efficient vector similarity search.
*   **Document Processing:**
    *   **PyMuPDF (`fitz`):** For parsing PDF files, extracting text and images.
    *   **Pillow (`PIL`):** For basic image handling (used indirectly by `fitz`).
    *   **Pandas (`pandas`):** For managing metadata associated with embeddings.
    *   **PyArrow (`pyarrow`):** Required by Pandas for efficient Parquet file I/O (metadata storage).
*   **Audio Handling:**
    *   **PyAudio (`pyaudio`):** For recording microphone input and playing audio output.
    *   **SoundDevice (`sounddevice`):** (Often a dependency or alternative for audio I/O, though PyAudio is directly used here).
*   **Concurrency:**
    *   **Asyncio:** Python's built-in library for asynchronous programming.
*   **Configuration:**
    *   **Python-Dotenv (`python-dotenv`):** For managing environment variables (API keys, project settings).
*   **Numerical Computation:**
    *   **NumPy (`numpy`):** For numerical operations, especially on embeddings.
    *   **Scikit-learn (`scikit-learn`):** Used for `cosine_similarity` if needed, though FAISS handles the primary search here.

## Prerequisites (Google Cloud Setup)

1.  **Google Cloud Account:** You need a Google Cloud Platform account with billing enabled.
2.  **Google Cloud Project:** Create a new project or select an existing one.
3.  **Enable APIs:** In your Google Cloud project, enable the following APIs:
    *   **Vertex AI API:** Essential for accessing Gemini models for image description and potentially embeddings/generation if configured that way.
    *   **Cloud Speech-to-Text API:** Required for transcribing your voice input.
    *   *(The Generative Language API used by `google-genai` for live connection and embeddings is often implicitly enabled or managed under Vertex AI, but ensure there are no restrictions)*.
4.  **Authentication:** You need to authenticate your environment to use Google Cloud services. The recommended way is using Application Default Credentials (ADC):
    *   Install the Google Cloud CLI (`gcloud`): [Installation Guide](https://cloud.google.com/sdk/docs/install)
    *   Log in and set up ADC:
        ```bash
        gcloud auth application-default login
        gcloud config set project YOUR_PROJECT_ID
        ```
        Replace `YOUR_PROJECT_ID` with your actual Google Cloud project ID. This command will open a browser window for you to log in with your Google account.
5.  **Audio Setup:** Ensure your microphone and speakers are correctly configured on your system. You might need to install system libraries for `pyaudio` (like `portaudio`). See the [PyAudio documentation](http://people.csail.mit.edu/hubert/pyaudio/) for platform-specific requirements if you encounter installation issues.

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
    *   Add the following lines, replacing the placeholder with your Google Cloud Project ID and the region where you enabled the APIs (often `us-central1` for Vertex AI):

    ```dotenv
    # .env file
    GOOGLE_CLOUD_PROJECT=your-gcp-project-id
    GOOGLE_CLOUD_REGION=us-central1
    GOOGLE_GENAI_USE_VERTEXAI=True
    # GEMINI_API_KEY= (Optional - Usually not needed when using Vertex AI with gcloud ADC)
    ```

5.  **Set up Google Cloud Authentication:** Ensure you have run `gcloud auth application-default login` as described in the Prerequisites.

## Usage

Run the main script from your terminal, providing the path(s) to the PDF document(s) you want to query.

```bash
python module_v2v_rag.py path/to/your/document1.pdf [path/to/another/document2.pdf ...] [--language <lang_code>] [--voice <voice_name>]
```

**Arguments:**

*   `pdf_paths`: (Required) One or more paths to PDF files.
*   `--language`: (Optional) The language code for both input transcription and output speech (e.g., `en-US`, `hi-IN`, `es-US`). Defaults to `hi-IN`. See `SUPPORTED_LANGUAGES` in the script for options.
*   `--voice`: (Optional) The desired output voice name (e.g., `Aoede`, `Puck`, `Charon`). Defaults to `Aoede`. See `AVAILABLE_VOICES` in the script for options.

**Interaction:**

1.  The application will initialize, process the PDFs (building/loading the index). This might take time on the first run for large documents.
2.  You will see the message: `>>> Press ENTER to record, ENTER again to stop. (Ctrl+C to quit) <<<`.
3.  Press `Enter`. The application will start recording audio from your microphone (`[Recording... Press ENTER to stop]`).
4.  Speak your question clearly.
5.  Press `Enter` again to stop recording.
6.  The application will transcribe your audio, search the document index, send the query and context to Gemini Live, receive the audio response, and play it back. The transcribed response may also be printed.
7.  The prompt `>>> Press ENTER to record... <<<` will reappear, allowing you to ask another question.
8.  Press `Ctrl+C` to exit the application gracefully.

## Configuration (`.env` file)

*   `GOOGLE_CLOUD_PROJECT`: **Required.** Your Google Cloud project ID where APIs are enabled.
*   `GOOGLE_CLOUD_REGION`: **Required.** The Google Cloud region for Vertex AI services (e.g., `us-central1`).
*   `GOOGLE_GENAI_USE_VERTEXAI`: Should be set to `True` to use Vertex AI infrastructure and authentication (recommended).
*   `GEMINI_API_KEY`: Generally **not required** if you use `gcloud auth application-default login` (ADC). Only needed if you intend to authenticate directly with an API key via the `google-genai` SDK without Vertex integration (which is not the default setup in `config.py`).

## Future Improvements

*   **Web Frontend:** Create a user-friendly web interface (using Flask, Streamlit, Gradio, etc.) to replace the command-line interaction, allowing easier PDF upload, language/voice selection, and chat history display.
*   **Enhanced RAG Strategies:**
    *   Implement re-ranking of retrieved results for better relevance.
    *   Explore query expansion techniques.
    *   Use more sophisticated chunking strategies.
*   **Support More Document Types:** Extend parsing capabilities beyond PDFs to include `.docx`, `.html`, `.txt`, etc.
*   **GPU Acceleration:** Utilize `faiss-gpu` if a compatible GPU is available for even faster indexing and searching on very large datasets.
*   **Improved Error Handling:** Add more robust error handling and user feedback for API issues, file processing errors, or audio device problems.
*   **Caching:** Implement caching for embeddings and potentially image descriptions to speed up reprocessing.
*   **Containerization:** Package the application using Docker for easier deployment and dependency management.
*   **Conversation History:** Allow the model to consider previous turns in the conversation for more contextually relevant answers.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.
