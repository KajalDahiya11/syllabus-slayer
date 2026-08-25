"""
StudySage — Progress Tracking Routes
POST /progress/score   — record a quiz score
GET  /progress         — get full progress report
"""
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from models.schemas import ScoreSubmit
from utils.auth import get_current_user
from utils.database import get_db

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/progress/score")
async def record_score(
    body: ScoreSubmit,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    db = get_db()

    entry = {
        "quiz_id": body.quiz_id,
        "topic_name": body.topic_name,
        "score": body.score,
        "total_questions": body.total_questions,
        "correct": body.correct,
        "taken_at": datetime.utcnow(),
    }

    await db.progress.update_one(
        {"user_id": user_id},
        {"$push": {"scores": entry}, "$set": {"updated_at": datetime.utcnow()}},
        upsert=True,
    )
    logger.info("Score recorded: user=%s topic=%s score=%d%%", user_id, body.topic_name, body.score)
    return {"success": True, "score": body.score}


@router.get("/progress")
async def get_progress(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    db = get_db()

    doc = await db.progress.find_one({"user_id": user_id})
    if not doc or not doc.get("scores"):
        return {
            "total_tests": 0,
            "average_score": 0,
            "best_score": 0,
            "scores": [],
            "topic_breakdown": {},
        }

    scores = doc["scores"]
    values = [s["score"] for s in scores]
    avg = round(sum(values) / len(values), 1)
    best = max(values)

    # Topic breakdown
    breakdown: dict[str, list[int]] = defaultdict(list)
    for s in scores:
        breakdown[s["topic_name"]].append(s["score"])

    topic_breakdown: dict[str, Any] = {}
    for topic, sc in breakdown.items():
        topic_breakdown[topic] = {
            "tests": len(sc),
            "average": round(sum(sc) / len(sc), 1),
            "best": max(sc),
        }

    # Serialise datetimes
    serialised = []
    for s in scores:
        serialised.append({
            "quiz_id": s.get("quiz_id", ""),
            "topic_name": s.get("topic_name", ""),
            "score": s.get("score", 0),
            "total_questions": s.get("total_questions", 0),
            "correct": s.get("correct", 0),
            "taken_at": s["taken_at"].isoformat() if isinstance(s.get("taken_at"), datetime) else str(s.get("taken_at", "")),
        })

    return {
        "total_tests": len(scores),
        "average_score": avg,
        "best_score": best,
        "scores": serialised,
        "topic_breakdown": topic_breakdown,
    }


@router.delete("/progress")
async def reset_progress(current_user: dict = Depends(get_current_user)):
    db = get_db()
    await db.progress.delete_one({"user_id": current_user["user_id"]})
    return {"success": True, "message": "Progress reset"}
