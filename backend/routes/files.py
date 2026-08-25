"""
StudySage — File Upload Routes
POST /upload  — upload and process a file
GET  /files   — list user's files
DELETE /files/{file_id}
"""
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from bson import ObjectId
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from models.schemas import FileOut
from services.file_processor import process_file
from services.rag_service import clear_index, index_document
from utils.auth import get_current_user
from utils.database import get_db

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "20")) * 1024 * 1024  # bytes


def _file_to_out(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "filename": doc["filename"],
        "original_name": doc["original_name"],
        "content_type": doc["content_type"],
        "file_size": doc["file_size"],
        "uploaded_at": doc["uploaded_at"].isoformat(),
        "has_text": bool(doc.get("extracted_text", "")),
        "ocr_used": doc.get("ocr_used", False),
    }


async def _background_index(user_id: str, text: str) -> None:
    """Index document embeddings in background so upload response is fast."""
    try:
        n = index_document(user_id, text)
        logger.info("Background indexing complete: %d chunks for user %s", n, user_id)
    except Exception as e:
        logger.warning("Background indexing failed: %s", e)


@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    db = get_db()

    # Size check
    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)} MB",
        )

    ct = file.content_type or "application/octet-stream"

    # Extract text
    text, ocr_used = process_file(file.filename or "upload", ct, data)

    if not text.strip():
        logger.warning("No text extracted from %s", file.filename)

    # Persist file to disk (optional — can also store in GridFS)
    safe_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    file_path = UPLOAD_DIR / safe_name
    file_path.write_bytes(data)

    # Save metadata to MongoDB
    doc = {
        "user_id": user_id,
        "filename": safe_name,
        "original_name": file.filename,
        "content_type": ct,
        "file_size": len(data),
        "extracted_text": text,
        "ocr_used": ocr_used,
        "uploaded_at": datetime.utcnow(),
    }
    result = await db.files.insert_one(doc)
    doc["_id"] = result.inserted_id

    # Rebuild vector index in background
    clear_index(user_id)
    if text:
        background_tasks.add_task(_background_index, user_id, text)

    logger.info(
        "Uploaded: %s | user=%s | size=%d | ocr=%s | chars=%d",
        file.filename, user_id, len(data), ocr_used, len(text),
    )

    return {
        "success": True,
        "file": _file_to_out(doc),
        "extracted_chars": len(text),
        "ocr_used": ocr_used,
        "message": "File uploaded and processed successfully",
    }


@router.post("/upload-text")
async def upload_text(
    background_tasks: BackgroundTasks,
    body: dict,
    current_user: dict = Depends(get_current_user),
):
    """Accept pasted plain text from the frontend."""
    user_id = current_user["user_id"]
    db = get_db()
    text = (body.get("text") or "").strip()
    name = (body.get("name") or "Pasted text").strip()

    if not text:
        raise HTTPException(status_code=422, detail="No text provided")

    doc = {
        "user_id": user_id,
        "filename": f"text_{uuid.uuid4().hex}.txt",
        "original_name": name,
        "content_type": "text/plain",
        "file_size": len(text.encode()),
        "extracted_text": text[:50000],
        "ocr_used": False,
        "uploaded_at": datetime.utcnow(),
    }
    result = await db.files.insert_one(doc)
    doc["_id"] = result.inserted_id

    clear_index(user_id)
    background_tasks.add_task(_background_index, user_id, text)

    return {"success": True, "file": _file_to_out(doc)}


@router.get("/files")
async def list_files(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    db = get_db()
    cursor = db.files.find({"user_id": user_id}, {"extracted_text": 0}).sort("uploaded_at", -1)
    docs = await cursor.to_list(length=100)
    return {"files": [_file_to_out(d) for d in docs]}


@router.delete("/files/{file_id}")
async def delete_file(file_id: str, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    db = get_db()

    if not ObjectId.is_valid(file_id):
        raise HTTPException(status_code=400, detail="Invalid file ID")

    doc = await db.files.find_one({"_id": ObjectId(file_id), "user_id": user_id})
    if not doc:
        raise HTTPException(status_code=404, detail="File not found")

    # Delete disk file
    try:
        (UPLOAD_DIR / doc["filename"]).unlink(missing_ok=True)
    except Exception:
        pass

    await db.files.delete_one({"_id": ObjectId(file_id)})
    await db.topics.delete_many({"file_id": file_id, "user_id": user_id})

    # Rebuild index without this file
    remaining = await db.files.find({"user_id": user_id}).to_list(length=100)
    clear_index(user_id)
    combined = " ".join(f.get("extracted_text", "") for f in remaining)
    if combined:
        index_document(user_id, combined)

    return {"success": True, "message": "File deleted"}
