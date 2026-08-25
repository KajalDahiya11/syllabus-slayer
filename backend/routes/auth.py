"""
StudySage — Authentication Routes
POST /auth/signup
POST /auth/login
"""
import logging
from datetime import datetime

from bson import ObjectId
from fastapi import APIRouter, HTTPException, status

from models.schemas import TokenResponse, UserCreate, UserInDB, UserLogin, UserOut
from utils.auth import create_access_token, hash_password, verify_password
from utils.database import get_db

router = APIRouter()
logger = logging.getLogger(__name__)


def _user_to_out(doc: dict) -> UserOut:
    return UserOut(
        id=str(doc["_id"]),
        name=doc["name"],
        email=doc["email"],
        created_at=doc.get("created_at", datetime.utcnow()),
        preferences=doc.get("preferences", {}),
    )


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(body: UserCreate):
    db = get_db()
    existing = await db.users.find_one({"email": body.email})
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    doc = {
        "name": body.name,
        "email": body.email,
        "hashed_password": hash_password(body.password),
        "created_at": datetime.utcnow(),
        "preferences": {
            "language": "English",
            "difficulty": "Intermediate",
            "mode": "Exam focused",
        },
    }
    result = await db.users.insert_one(doc)
    doc["_id"] = result.inserted_id
    user_id = str(result.inserted_id)

    token = create_access_token(user_id, body.email)
    logger.info("New user registered: %s", body.email)
    return TokenResponse(access_token=token, user=_user_to_out(doc))


@router.post("/login", response_model=TokenResponse)
async def login(body: UserLogin):
    db = get_db()
    doc = await db.users.find_one({"email": body.email})
    if not doc or not verify_password(body.password, doc["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user_id = str(doc["_id"])
    token = create_access_token(user_id, body.email)
    logger.info("User logged in: %s", body.email)
    return TokenResponse(access_token=token, user=_user_to_out(doc))
