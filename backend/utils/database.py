"""
StudySage — Async MongoDB connection via Motor
"""
import logging
import os

import certifi
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def connect_db() -> None:
    global _client, _db
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB", "studysage")

    kwargs: dict = {"serverSelectionTimeoutMS": 10000}
    if "mongodb+srv" in uri or "ssl=true" in uri.lower() or "tls=true" in uri.lower():
        kwargs["tlsCAFile"] = certifi.where()

    _client = AsyncIOMotorClient(uri, **kwargs)

    # Verify connection
    await _client.admin.command("ping")
    _db = _client[db_name]
    # Indexes
    await _db.users.create_index("email", unique=True)
    await _db.files.create_index("user_id")
    await _db.topics.create_index([("user_id", 1), ("file_id", 1)])
    await _db.quizzes.create_index("user_id")
    await _db.progress.create_index("user_id", unique=True)
    logger.info("MongoDB connected → db=%s", db_name)


async def close_db() -> None:
    global _client
    if _client:
        _client.close()
        logger.info("MongoDB connection closed")


def get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("Database not initialised — call connect_db() first")
    return _db
