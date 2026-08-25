# StudySage — Production Backend

> Full-stack AI learning platform. Upload PDFs, slides, docs and images → get topics, explanations, quizzes, and a RAG-powered study tutor.

---

## Architecture

```
Browser (studysage-frontend.html)
    │  JWT Bearer token
    ▼
FastAPI (main.py)
    ├── /auth/*         JWT signup / login  (bcrypt + python-jose)
    ├── /upload         File ingestion      (pdfplumber, PyMuPDF, python-docx, pytesseract)
    ├── /extract-topics NLP extraction      (spaCy NER + noun chunks)
    ├── /summarize      Text summarisation  (HuggingFace distilBART)
    ├── /generate-quiz  MCQ generation      (sentence-level NLP)
    ├── /chat           SSE streaming       (Claude API + FAISS RAG)
    └── /progress/*     Score persistence   (MongoDB)
    │
    ├── Motor (async MongoDB driver)
    │       Users · Files · Topics · Quizzes · Progress
    │
    └── In-process vector store
            sentence-transformers → embeddings
            FAISS IndexFlatIP     → nearest-neighbour retrieval
```

---

## Quick Start (Local)

### Prerequisites
| Tool | Install |
|------|---------|
| Python 3.11+ | [python.org](https://python.org) |
| MongoDB 7 | `brew install mongodb-community` / `apt install mongodb` |
| Tesseract OCR | `brew install tesseract` / `apt install tesseract-ocr` |
| Anthropic API key | [console.anthropic.com](https://console.anthropic.com) |

### One-command start
```bash
git clone <your-repo>
cd studysage-backend
bash run.sh          # handles venv, deps, spaCy model, MongoDB, and uvicorn
```

### Manual start
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

cp .env.example .env   # then add your ANTHROPIC_API_KEY

mongod --dbpath ./data/db &          # or use system service
uvicorn main:app --reload --port 8000
```

### Docker (recommended for production)
```bash
cp .env.example .env   # add ANTHROPIC_API_KEY
docker compose up --build
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGODB_DB` | `studysage` | Database name |
| `JWT_SECRET` | *(required)* | Random 64-char string for token signing |
| `TOKEN_EXPIRE_MINUTES` | `60` | JWT lifetime |
| `ANTHROPIC_API_KEY` | *(required)* | Your Claude API key |
| `CHAT_MODEL` | `claude-sonnet-4-20250514` | Model for chat & explanations |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers embedding model |
| `SUMMARISER_MODEL` | `sshleifer/distilbart-cnn-12-6` | HuggingFace summarisation model |
| `UPLOAD_DIR` | `uploads` | Directory for stored files |
| `MAX_FILE_SIZE_MB` | `20` | Upload size cap |
| `MAX_EXTRACTED_CHARS` | `50000` | Text extraction limit per file |

---

## API Reference

All protected routes require `Authorization: Bearer <token>` header.

### Auth
| Method | Path | Body | Description |
|--------|------|------|-------------|
| `POST` | `/auth/signup` | `{name, email, password}` | Create account → returns JWT |
| `POST` | `/auth/login` | `{email, password}` | Login → returns JWT |

### Files
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload file (multipart/form-data). PDF/DOCX/PPTX/images/text |
| `POST` | `/upload-text` | Upload pasted plain text `{text, name}` |
| `GET` | `/files` | List user's uploaded files |
| `DELETE` | `/files/{id}` | Delete file + its topics |

### AI Features
| Method | Path | Body | Description |
|--------|------|------|-------------|
| `POST` | `/extract-topics` | `{max_topics?}` | Extract topics via spaCy NLP |
| `POST` | `/summarize` | `{file_id?, topic?}` | Summarise content (BART/extractive) |
| `POST` | `/generate-quiz` | `{topic_id?, topic_name, num_questions}` | MCQ quiz generation |
| `POST` | `/chat` | `{messages, language, difficulty}` | SSE streaming chat (Claude + RAG) |
| `GET` | `/topics` | | List all extracted topics |
| `PATCH` | `/topics/{id}/studied` | `{studied: bool}` | Mark topic as studied |
| `DELETE` | `/topics` | | Clear all topics |

### Progress
| Method | Path | Body | Description |
|--------|------|------|-------------|
| `POST` | `/progress/score` | `{quiz_id, topic_name, score, total_questions, correct}` | Save quiz result |
| `GET` | `/progress` | | Full progress report with topic breakdown |
| `DELETE` | `/progress` | | Reset all progress |

### Health
| Method | Path |
|--------|------|
| `GET` | `/health` |

---

## Frontend Integration

Open `studysage-frontend.html` in your browser. The file reads:

```js
const API = 'http://localhost:8000';  // ← change to your deployed URL
```

Change this one line to point to your production backend and deploy the HTML anywhere (S3, Netlify, GitHub Pages, etc.).

The frontend auto-shows a Sign In modal on load. After authentication, all data persists in MongoDB across sessions.

---

## AI/ML Stack Details

### Topic Extraction (`/services/nlp_service.py`)
- **Primary:** spaCy `en_core_web_sm` — named entity recognition (ORG, PERSON, GPE, PRODUCT…) + noun chunk frequency ranking
- **Fallback:** Keyword frequency + bigram scoring (no spaCy dependency)

### Text Summarisation
- **Primary:** HuggingFace `distilbart-cnn-12-6` (CPU inference, ~300ms for 3000 chars)
- **Fallback:** Extractive — TF-IDF sentence scoring

### Quiz Generation
- Sentence-level fill-in-the-blank + distractor mining from document vocabulary
- Template fallback for topics with sparse text

### RAG Chat (`/services/rag_service.py`)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~50ms per doc chunk)
- **Index:** `faiss.IndexFlatIP` (exact inner-product search on normalised vectors)
- **Generation:** Claude API via SSE streaming with retrieved context injected as system prompt
- **Chunking:** 300-word sliding window, 50-word overlap

### OCR
- `pytesseract` wrapping Tesseract 5 — triggered automatically for image files and text-free PDFs

---

## Project Structure

```
studysage-backend/
├── main.py                    # FastAPI app, middleware, lifespan
├── Dockerfile
├── docker-compose.yml
├── run.sh                     # local dev one-liner
├── requirements.txt
├── .env.example
├── studysage-frontend.html    # updated frontend (backend-integrated)
│
├── models/
│   └── schemas.py             # Pydantic models: User, File, Topic, Quiz, Progress
│
├── routes/
│   ├── auth.py                # /auth/signup  /auth/login
│   ├── files.py               # /upload  /files  DELETE /files/{id}
│   ├── ai_features.py         # /extract-topics /summarize /generate-quiz /chat /topics
│   └── progress.py            # /progress  /progress/score
│
├── services/
│   ├── file_processor.py      # PDF / DOCX / PPTX / OCR dispatch
│   ├── nlp_service.py         # topic extraction, summarisation, quiz gen
│   └── rag_service.py         # embeddings, FAISS index, RAG context builder
│
└── utils/
    ├── auth.py                # JWT create/decode, bcrypt, FastAPI dependency
    ├── database.py            # Motor async MongoDB connect/index creation
    └── logger.py              # Structured logging setup
```

---

## Troubleshooting

**MongoDB connection refused**
```bash
mkdir -p data/db && mongod --dbpath ./data/db
```

**spaCy model not found**
```bash
python -m spacy download en_core_web_sm
```

**Tesseract not found (OCR)**
```bash
# macOS:   brew install tesseract
# Ubuntu:  sudo apt install tesseract-ocr
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

**HuggingFace model download slow / OOM**
Comment out the `transformers` and `torch` lines in `requirements.txt`. The backend falls back to extractive summarisation automatically.

**FAISS install fails**
```bash
pip install faiss-cpu  # CPU version works on all platforms
```
