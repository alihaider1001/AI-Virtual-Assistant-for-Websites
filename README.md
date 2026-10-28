## Website Assistant

This project indexes website and Wikipedia content locally, then answers questions with local embeddings and Google Gemini. The Streamlit frontend provides the chat interface and can be deployed directly to Streamlit Community Cloud.

### Setup

Use Python 3.10 or newer and install the dependencies into the project environment:

```powershell
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`. The embedding model downloads from Hugging Face during the first ingestion. These downloads are free, but require local disk space and memory.

### Run

From the project root, run ingestion once after configuring `.env`:

```powershell
python -m backend.ingest
```

For local development, start the API:

```powershell
uvicorn backend.main:app --reload --port 8000
```

Start the frontend in another terminal:

```powershell
streamlit run frontend/app.py
```

The Streamlit frontend now performs retrieval and Gemini generation directly, so the FastAPI service is not required for Streamlit Cloud deployment.

### Deploy to Streamlit Community Cloud

1. Push this repository to GitHub. Make sure `data/index/embeddings.npy` and
   `data/index/records.json` are committed.
2. Open [share.streamlit.io](https://share.streamlit.io/) and create a new app.
3. Select this repository, the `main` branch, and set the main file to
   `frontend/app.py`.
4. Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/apikey).
5. In the app's **Settings > Secrets**, add:

   ```toml
   GEMINI_API_KEY = "your-gemini-api-key"
   GEMINI_MODEL = "gemini-2.5-flash"
   ```

6. Deploy or reboot the app. The first startup can take a few minutes while
   the embedding model downloads.

Never commit `.env` or put the Gemini key in source code. Streamlit Cloud
secrets are exposed to the app as `st.secrets`.

### Configuration

| Variable | Purpose |
| --- | --- |
| `EMBEDDING_MODEL` | Local embedding model |
| `GENERATION_MODEL` | Ollama model, default `llama3.2:3b` |
| `OLLAMA_URL` | Ollama API URL, default `http://localhost:11434` |
| `INDEX_DIRECTORY` | Local vector index directory |
| `TOP_K` | Number of retrieved chunks |
| `ALLOWED_ORIGINS` | Comma-separated browser origins |
| `BACKEND_URL` | Frontend API URL |
| `GEMINI_API_KEY` | Google Gemini API key used by the Streamlit app |
| `GEMINI_MODEL` | Gemini model, default `gemini-2.5-flash` |
