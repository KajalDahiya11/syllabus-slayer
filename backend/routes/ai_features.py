"""
StudySage — AI Feature Routes
POST /extract-topics
POST /summarize
POST /generate-quiz
POST /chat           (Server-Sent Events streaming)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, AsyncGenerator, Optional

# pyrefly: ignore [missing-import]
import google.generativeai as genai
from bson import ObjectId
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse

from models.schemas import ChatRequest, QuizInDB, TopicInDB
from services.nlp_service import extract_topics, generate_quiz, summarise_text
from services.rag_service import build_rag_system_prompt, retrieve_context
from utils.auth import get_current_user
from utils.database import get_db

router = APIRouter()
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))
_raw_model = os.getenv("CHAT_MODEL", "gemini-2.0-flash").strip()
if not _raw_model or _raw_model in ["gemini-1.5-flash", "gemini-flash-latest", "gemini-2.5-flash"]:
    CHAT_MODEL = "gemini-2.0-flash"
else:
    CHAT_MODEL = _raw_model

DEFAULT_CANDIDATE_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-1.5-pro",
    "gemini-pro",
]





# ── Helper: get combined user text ─────────────────────────────────────────────

async def _get_user_text(user_id: str, file_id: Optional[str] = None) -> str:
    db = get_db()
    if file_id and ObjectId.is_valid(file_id):
        doc = await db.files.find_one({"_id": ObjectId(file_id), "user_id": user_id})
        return doc.get("extracted_text", "") if doc else ""
    # All files
    cursor = db.files.find({"user_id": user_id}, {"extracted_text": 1})
    docs = await cursor.to_list(length=20)
    return "\n\n".join(d.get("extracted_text", "") for d in docs)


# ── Extract Topics ─────────────────────────────────────────────────────────────

@router.post("/extract-topics")
async def extract_topics_route(
    body: dict,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    db = get_db()
    file_id = body.get("file_id")

    text = await _get_user_text(user_id, file_id)
    if not text.strip():
        raise HTTPException(status_code=422, detail="No text content found. Please upload files first.")

    max_topics = int(body.get("max_topics", 8))
    raw_topics = extract_topics(text, max_topics)

    # Persist topics
    saved: list[dict[str, Any]] = []
    for t in raw_topics:
        doc = {
            "user_id": user_id,
            "file_id": file_id or "all",
            "name": t["name"],
            "description": t["description"],
            "keywords": t.get("keywords", []),
            "studied": False,
            "created_at": datetime.utcnow(),
        }
        result = await db.topics.insert_one(doc)
        saved.append({
            "id": str(result.inserted_id),
            "name": doc["name"],
            "description": doc["description"],
            "keywords": doc["keywords"],
            "studied": False,
        })

    logger.info("Extracted %d topics for user %s", len(saved), user_id)
    return {"topics": saved, "count": len(saved)}


# ── Summarise ──────────────────────────────────────────────────────────────────

@router.post("/summarize")
async def summarize_route(
    body: dict,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    file_id = body.get("file_id")
    topic = body.get("topic", "")

    text = await _get_user_text(user_id, file_id)
    if not text.strip():
        raise HTTPException(status_code=422, detail="No text content found")

    # Filter to topic if specified
    if topic:
        sentences = [s for s in text.split(".") if topic.lower() in s.lower()]
        if sentences:
            text = ". ".join(sentences[:30])

    if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "YOUR_GEMINI_API_KEY_HERE":
        summary = summarise_text(text)
    else:
        try:
            summary = ""
            for m_name in [CHAT_MODEL] + [m for m in DEFAULT_CANDIDATE_MODELS if m != CHAT_MODEL]:
                try:
                    model = genai.GenerativeModel(m_name)
                    prompt = (
                        f"Explain the topic '{topic}' in very simple, easy-to-understand language. "
                        f"Imagine you are explaining it to a student who is new to this concepts. "
                        f"Use a friendly tone, clear analogies, and avoid overly technical jargon unless you explain it first. "
                        f"Base your explanation ONLY on the following text content.\n\n"
                        f"Content:\n{text[:10000]}"
                    )
                    resp = await model.generate_content_async(prompt)
                    summary = resp.text.strip()
                    if summary:
                        break
                except Exception as ex:
                    logger.warning("Gemini summarize model %s failed: %s", m_name, ex)
                    continue
            if not summary:
                summary = summarise_text(text)
        except Exception as e:
            logger.error("Gemini Summarization failed: %s", e)
            summary = summarise_text(text)


    return {"summary": summary}


# ── Generate Quiz ──────────────────────────────────────────────────────────────

@router.post("/generate-quiz")
async def generate_quiz_route(
    body: dict,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    db = get_db()

    topic_id = body.get("topic_id")
    topic_name = body.get("topic_name", "General")
    num_questions = int(body.get("num_questions", 5))
    file_id = body.get("file_id")

    # Get relevant text
    text = await _get_user_text(user_id, file_id)

    # Narrow to topic sentences if possible
    if topic_id and ObjectId.is_valid(topic_id):
        topic_doc = await db.topics.find_one({"_id": ObjectId(topic_id), "user_id": user_id})
        if topic_doc:
            topic_name = topic_doc["name"]
            keywords = topic_doc.get("keywords", [topic_name])
            relevant = [s for s in text.split(".") if any(k.lower() in s.lower() for k in keywords)]
            if relevant:
                text = ". ".join(relevant[:40])

    system_prompt = f"""You are an advanced AI assessment generator.
Generate a high-quality quiz with {num_questions} questions for a student based on the provided text content about the topic '{topic_name}'.
The quiz MUST be challenging and cover:
1. Deep conceptual understanding (avoid basic fill-in-the-blank).
2. Numerical or analytical reasoning where applicable.
3. Logical Multiple Select Questions (MSQs) where multiple options can be correct.

Include a balanced mix of Single Choice MCQs and Multiple Select Questions (MSQs).

Return ONLY a valid JSON array of question objects matching this exact schema:
[
  {{
    "question": "The question text.",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": 0, // Integer index if only one option is correct. USE AN ARRAY OF INTEGERS if it is an MSQ (e.g., [0, 2]).
    "explanation": "A short, clear explanation of why the answer is correct."
  }}
]
Ensure the response is purely the JSON array and NOTHING else. No markdown blocks, no intro, no outro."""

    prompt = f"Topic: {topic_name}\n\nContent:\n{text[:15000]}"

    questions = []
    try:
        if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "YOUR_GEMINI_API_KEY_HERE":
            questions = generate_quiz(text, topic_name, num_questions)
        else:
            candidate_models = [CHAT_MODEL] + [m for m in DEFAULT_CANDIDATE_MODELS if m != CHAT_MODEL]
            resp = None
            last_err = None
            for m_name in candidate_models:
                try:
                    model = genai.GenerativeModel(
                        model_name=m_name,
                        system_instruction=system_prompt,
                        generation_config=genai.GenerationConfig(
                            response_mime_type="application/json",
                            temperature=0.2,
                        )
                    )
                    resp = await model.generate_content_async(prompt)
                    if resp and resp.text:
                        break
                except Exception as ex:
                    last_err = ex
                    logger.warning("Gemini quiz model %s failed, trying fallback: %s", m_name, ex)
                    continue

            if not resp or not resp.text:
                if last_err:
                    raise last_err
                raise RuntimeError("Quiz generation failed across all candidate models.")

            raw_text = resp.text.strip()
            # Clean possible markdown formatting
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()
            
            questions = json.loads(raw_text)
            
            # Enforce constraints
            if len(questions) > num_questions:
                questions = questions[:num_questions]
    except Exception as e:
        logger.error("LLM Quiz Generation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM Quiz Generation failed: {str(e)}")

    # Persist quiz
    quiz_doc = {
        "user_id": user_id,
        "topic_id": topic_id,
        "topic_name": topic_name,
        "questions": [q for q in questions],
        "created_at": datetime.utcnow(),
    }
    result = await db.quizzes.insert_one(quiz_doc)

    logger.info("Generated %d-question quiz on '%s' for user %s", len(questions), topic_name, user_id)
    return {
        "quiz_id": str(result.inserted_id),
        "topic_name": topic_name,
        "questions": questions,
    }


# ── Chat (SSE Streaming) ───────────────────────────────────────────────────────

async def _stream_sse(user_id: str, request: ChatRequest) -> AsyncGenerator[str, None]:
    """Yield SSE events with streamed tokens from Claude."""
    try:
        # Retrieve RAG context (skip for simple Topic Explainer to avoid slow CPU embeddings)
        last_user_msg = next(
            (m.content for m in reversed(request.messages) if m.role == "user"), ""
        )
        if last_user_msg.startswith("Topic: "):
            rag_ctx = ""
        else:
            rag_ctx = retrieve_context(user_id, last_user_msg)

        # Get topic names for context
        db = get_db()
        topic_docs = await db.topics.find({"user_id": user_id}, {"name": 1}).to_list(20)
        topic_names = [t["name"] for t in topic_docs]

        system_prompt = build_rag_system_prompt(
            context=rag_ctx,
            topics=topic_names,
            language=request.language,
            difficulty=request.difficulty,
        )

        if len(request.messages) == 1:
            contents = request.messages[0].content
        else:
            contents = []
            last_role = None
            for m in request.messages[-10:]:
                role = "model" if m.role == "assistant" else "user"
                if role == last_role:
                    continue
                contents.append({"role": role, "parts": [m.content]})
                last_role = role
        # Combine system prompt into prompt to support all Gemini models (including gemini-pro and flash versions)
        if isinstance(contents, str):
            final_prompt = f"{system_prompt}\n\nUser Question: {contents}"
        else:
            final_prompt = [{"role": "user", "parts": [system_prompt]}] + contents

        response_started = False
        candidate_models = [CHAT_MODEL] + [m for m in DEFAULT_CANDIDATE_MODELS if m != CHAT_MODEL]
        last_err = None

        for m_name in candidate_models:
            try:
                model = genai.GenerativeModel(model_name=m_name)
                res = await model.generate_content_async(
                    final_prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=1024,
                        temperature=0.7,
                    ),
                    stream=True
                )
                async for chunk in res:
                    if chunk.text:
                        payload = json.dumps({"token": chunk.text})
                        yield f"data: {payload}\n\n"
                        response_started = True

                if response_started:
                    break
            except Exception as ex:
                last_err = ex
                logger.warning("Gemini model %s failed, trying fallback: %s", m_name, ex)
                if response_started:
                    break
                continue

        if not response_started:
            err_msg = f"Could not connect to Gemini AI models ({str(last_err)}). Please verify GOOGLE_API_KEY in Render." if last_err else "Please check your GOOGLE_API_KEY."
            raise RuntimeError(err_msg)


        yield "data: [DONE]\n\n"



    except Exception as e:
        logger.error("Chat stream error: %s", e, exc_info=True)
        err = json.dumps({"error": f"AI error: {str(e)}"})
        yield f"data: {err}\n\n"



@router.post("/chat")
async def chat_route(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "YOUR_GEMINI_API_KEY_HERE":
        raise HTTPException(status_code=503, detail="AI chat service not configured (missing GOOGLE_API_KEY)")

    return StreamingResponse(
        _stream_sse(user_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Topic management helpers ───────────────────────────────────────────────────

@router.get("/topics")
async def list_topics(current_user: dict = Depends(get_current_user)):
    db = get_db()
    docs = await db.topics.find({"user_id": current_user["user_id"]}).to_list(200)
    return {
        "topics": [
            {
                "id": str(d["_id"]),
                "name": d["name"],
                "description": d["description"],
                "keywords": d.get("keywords", []),
                "studied": d.get("studied", False),
                "file_id": d.get("file_id", ""),
            }
            for d in docs
        ]
    }


@router.patch("/topics/{topic_id}/studied")
async def mark_topic_studied(
    topic_id: str,
    body: dict,
    current_user: dict = Depends(get_current_user),
):
    db = get_db()
    if not ObjectId.is_valid(topic_id):
        raise HTTPException(status_code=400, detail="Invalid topic ID")

    studied = bool(body.get("studied", True))
    result = await db.topics.update_one(
        {"_id": ObjectId(topic_id), "user_id": current_user["user_id"]},
        {"$set": {"studied": studied}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Topic not found")
    return {"success": True, "studied": studied}


@router.delete("/topics")
async def clear_topics(current_user: dict = Depends(get_current_user)):
    db = get_db()
    result = await db.topics.delete_many({"user_id": current_user["user_id"]})
    return {"success": True, "deleted": result.deleted_count}
