"""
StudySage — Database models (Pydantic v2 + Motor/MongoDB)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from bson import ObjectId
from pydantic import BaseModel, EmailStr, Field, field_validator


# ── Helpers ────────────────────────────────────────────────────────────────────

class PyObjectId(str):
    """Custom type so ObjectId serialises cleanly to str."""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return str(v)
        if ObjectId.is_valid(str(v)):
            return str(v)
        raise ValueError(f"Invalid ObjectId: {v}")


# ── User ───────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserInDB(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    name: str
    email: str
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    preferences: dict[str, Any] = Field(default_factory=lambda: {
        "language": "English",
        "difficulty": "Intermediate",
        "mode": "Exam focused",
    })

    model_config = {"populate_by_name": True, "arbitrary_types_allowed": True}


class UserOut(BaseModel):
    id: str
    name: str
    email: str
    created_at: datetime
    preferences: dict[str, Any]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ── File ───────────────────────────────────────────────────────────────────────

class FileInDB(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    user_id: str
    filename: str
    original_name: str
    content_type: str
    file_size: int
    extracted_text: str = ""
    ocr_used: bool = False
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"populate_by_name": True}


class FileOut(BaseModel):
    id: str
    filename: str
    original_name: str
    content_type: str
    file_size: int
    uploaded_at: datetime


# ── Topic ──────────────────────────────────────────────────────────────────────

class TopicInDB(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    user_id: str
    file_id: str
    name: str
    description: str
    keywords: list[str] = Field(default_factory=list)
    studied: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"populate_by_name": True}


class TopicOut(BaseModel):
    id: str
    name: str
    description: str
    keywords: list[str]
    studied: bool
    file_id: str


# ── Quiz ───────────────────────────────────────────────────────────────────────

from typing import Union

class QuizQuestion(BaseModel):
    question: str
    options: list[str]          # exactly 4 options
    answer: Union[int, list[int]]   # index 0-3, or list of indices for MSQ
    explanation: str = ""


class QuizInDB(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    user_id: str
    topic_id: Optional[str] = None
    topic_name: str
    questions: list[QuizQuestion]
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"populate_by_name": True}


class QuizOut(BaseModel):
    id: str
    topic_name: str
    questions: list[QuizQuestion]


# ── Progress ──────────────────────────────────────────────────────────────────

class ScoreEntry(BaseModel):
    quiz_id: str
    topic_name: str
    score: int                  # percentage 0-100
    total_questions: int
    correct: int
    taken_at: datetime = Field(default_factory=datetime.utcnow)


class ProgressInDB(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    user_id: str
    scores: list[ScoreEntry] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"populate_by_name": True}


class ProgressOut(BaseModel):
    total_tests: int
    average_score: float
    best_score: int
    scores: list[ScoreEntry]
    topic_breakdown: dict[str, Any]


class ScoreSubmit(BaseModel):
    quiz_id: str
    topic_name: str
    score: int
    total_questions: int
    correct: int


# ── Chat ───────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    file_id: Optional[str] = None
    language: str = "English"
    difficulty: str = "Intermediate"
