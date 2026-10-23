# 🎓 AI StudyMate Pro — GenAI Capstone

Codomax Module 6 (Days 26–30). An educational AI assistant that combines a Gemini LLM, prompt engineering, retrieval-augmented generation (RAG), neural embeddings, a ChromaDB vector store, function calling, and session conversation memory.

## Features
- Chat about studying, coding, and research with Gemini.
- Upload PDF/TXT study notes, split into overlapping chunks, embed with `all-MiniLM-L6-v2`, and index in ChromaDB.
- Agent tool calls: `search_notes`, `calculator`, and `wikipedia_search`.
- Multi-step tool workflows and expandable execution traces.
- Session-only chat memory, document grounding, and basic error handling.

## Architecture
See [ARCHITECTURE.md](ARCHITECTURE.md) and [architecture.mmd](architecture.mmd).

## Run locally
```bash
pip install -r requirements.txt
mkdir -p .streamlit
# Create .streamlit/secrets.toml with GEMINI_API_KEY = "your_key"
streamlit run app.py
```
**Never commit API keys.** For Streamlit Community Cloud, set `GEMINI_API_KEY` in app Settings → Secrets.

## Deploy
1. Push `app.py`, `requirements.txt`, `README.md`, `ARCHITECTURE.md`, and `architecture.mmd` to a public GitHub repository.
2. Deploy from [Streamlit Community Cloud](https://share.streamlit.io/) using branch `main` and entry point `app.py`.
3. Add `GEMINI_API_KEY` in app Secrets.
4. Test with the prompts below.

## Demo test prompts
1. `Calculate (125 * 8) + 450.` (calculator)
2. `Find Alan Turing's birth year on Wikipedia, then calculate how old he would turn in 2026.` (multi-step tools)
3. Upload a Data Structures PDF: `Explain the enqueue operation according to my uploaded notes and cite the chunks.` (RAG)
4. `What is the capital of France according to my uploaded notes?` (document grounding)
5. `My favorite subject is operating systems.` Then: `What is my favorite subject?` (session memory)

## Limitations and safety
- Model answers may be incorrect; verify critical facts.
- Session memory is not persistent across restarts.
- Uploaded documents are indexed in an in-memory vector store and are lost after server restarts.
- Scanned PDFs without selectable text need OCR (not included).
- Wikipedia is a secondary source, not a guarantee of accuracy.
- The calculator parses a restricted arithmetic expression tree; it never executes arbitrary Python code.
- Tool calls are limited to five rounds per question.
- Do not upload confidential or sensitive documents to a public demonstration app.

## Capstone deliverables
GitHub repository, README/documentation, architecture diagram, short demo video, and live Streamlit URL.
