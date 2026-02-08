# PicoRAG Architecture

## Goal
Single-question, single-pass RAG:
1. Load documents from `data/*.txt`
2. Chunk by paragraph
3. Persist embeddings/index in local Chroma
4. Retrieve top-k chunks for one question
5. Generate one answer with OpenRouter using retrieved context

No multi-turn state, no command system, no runtime configuration from REPL.

## File Structure
- `config.py`: fixed runtime constants (paths, model, retrieval and generation defaults)
- `ingest.py`: document loading from local text files
- `chunking.py`: deterministic paragraph chunking with stable chunk IDs
- `retrieve.py`: persistent Chroma client, collection access, upsert indexing, semantic search
- `generate.py`: OpenRouter API call and prompt construction
- `pipeline.py`: orchestration for one question (`answer_question`)
- `app.py`: one-shot CLI app that reads one question and prints answer + retrieved context

## Data Flow
1. `app.py` reads one question.
2. `pipeline.answer_question(question)`:
   - opens Chroma `PersistentClient(path=CHROMA_PERSIST_DIR)`
   - gets/creates collection
   - loads `data/*.txt`, chunks, and `upsert`s chunks (idempotent)
   - queries top-k semantic matches
   - sends question + retrieved chunks to OpenRouter
3. Returns answer text and retrieval evidence.

## Persistence
- Chroma is stored at `CHROMA_PERSIST_DIR` (`./chroma_data`).
- Ingestion uses deterministic SHA1 chunk IDs and `upsert`, so reruns do not duplicate existing chunks.
- Retrieval always uses persisted collection data.

## External Requirements
- Environment variable: `OPENROUTER_API_KEY`
- Python packages:
  - `chromadb`
  - `httpx`

## Scope Boundaries
- No cloud vector DB.
- No OpenAI SDK.
- No multi-turn chat memory.
- No ingestion/config commands in app.
