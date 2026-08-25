"""
StudySage — NLP / AI Service
Topic extraction, summarisation, quiz generation.
Uses spaCy + HuggingFace transformers where available,
with a robust rule-based fallback so the server always responds.
"""
from __future__ import annotations

import json
import logging
import os
import re
import string
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

# ── Optional heavy imports ─────────────────────────────────────────────────────

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    _SPACY = True
    logger.info("spaCy loaded (en_core_web_sm)")
except Exception:
    _SPACY = False
    _nlp = None
    logger.warning("spaCy not available — using keyword fallback")

try:
    from transformers import pipeline, Pipeline
    _summariser: Pipeline | None = pipeline(
        "summarization",
        model=os.getenv("SUMMARISER_MODEL", "sshleifer/distilbart-cnn-12-6"),
        device=-1,   # CPU
    )
    _SUMMARISER = True
    logger.info("HuggingFace summariser loaded")
except Exception as e:
    _SUMMARISER = False
    _summariser = None
    logger.warning("HuggingFace summariser not available: %s", e)

# ── Stopwords (lightweight, no NLTK needed) ────────────────────────────────────
_STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall","can",
    "to","of","in","on","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","from","up","down","out",
    "off","over","under","again","further","then","once","and","but","or","nor",
    "so","yet","both","either","neither","not","only","own","same","than","too",
    "very","just","because","as","until","while","this","that","these","those",
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "he","him","his","she","her","hers","it","its","they","them","their","theirs",
    "what","which","who","whom","when","where","why","how","all","each","every",
    "both","few","more","most","other","some","such","no","nor","not","only",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sentences(text: str) -> list[str]:
    """Simple sentence splitter (no NLTK)."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]


def _keyword_topics(text: str, n: int = 8) -> list[dict[str, Any]]:
    """Extract topics using keyword frequency (no spaCy fallback)."""
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    freq = Counter(w for w in words if w not in _STOPWORDS)
    # Bigrams
    tokens = [w for w in words if w not in _STOPWORDS]
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
    bg_freq = Counter(bigrams)
    # Merge and pick top n
    combined = {**{k: v*1.5 for k,v in bg_freq.most_common(20)}, **dict(freq.most_common(40))}
    top = sorted(combined, key=combined.get, reverse=True)[:n]

    topics = []
    for phrase in top:
        # Find a sentence mentioning this phrase
        desc = next(
            (s for s in _sentences(text) if phrase.lower() in s.lower()),
            f"A key concept related to {phrase}."
        )
        topics.append({
            "name": phrase.title(),
            "description": desc[:200],
            "keywords": phrase.split(),
        })
    return topics


def _spacy_topics(text: str, n: int = 8) -> list[dict[str, Any]]:
    """Use spaCy NER + noun chunks for richer topic extraction."""
    doc = _nlp(text[:10000])   # limit for speed
    candidates: dict[str, int] = {}

    # Named entities (weighted higher)
    for ent in doc.ents:
        if ent.label_ in ("ORG","PERSON","GPE","PRODUCT","EVENT","WORK_OF_ART","LAW","LANGUAGE","NORP"):
            key = ent.text.strip()
            candidates[key] = candidates.get(key, 0) + 3

    # Noun chunks
    for chunk in doc.noun_chunks:
        key = chunk.text.strip().lower()
        if len(key) > 3 and key not in _STOPWORDS:
            candidates[key.title()] = candidates.get(key.title(), 0) + 1

    top = sorted(candidates, key=candidates.get, reverse=True)[:n]
    topics = []
    sentences = _sentences(text)
    for name in top:
        desc = next(
            (s for s in sentences if name.lower() in s.lower()),
            f"{name} is a key concept in this material."
        )
        topics.append({
            "name": name,
            "description": desc[:200],
            "keywords": [w.lower() for w in name.split() if w.lower() not in _STOPWORDS],
        })
    return topics


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_topics(text: str, max_topics: int = 8) -> list[dict[str, Any]]:
    """Extract key topics from text. Uses spaCy if available, else keyword fallback."""
    if not text.strip():
        return []
    if _SPACY:
        topics = _spacy_topics(text, max_topics)
    else:
        topics = _keyword_topics(text, max_topics)
    return topics[:max_topics]


def summarise_text(text: str, max_length: int = 300) -> str:
    """Summarise text. Uses HuggingFace BART if available, else extractive fallback."""
    if not text.strip():
        return "No content to summarise."

    if _SUMMARISER and _summariser is not None:
        try:
            # HuggingFace models have input token limits
            chunk = text[:3000]
            result = _summariser(
                chunk,
                max_length=max_length,
                min_length=60,
                do_sample=False,
            )
            return result[0]["summary_text"]
        except Exception as e:
            logger.warning("Transformer summariser failed: %s — using extractive fallback", e)

    # Extractive fallback: pick top sentences by keyword density
    sentences = _sentences(text)
    if not sentences:
        return text[:500]

    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    freq = Counter(w for w in words if w not in _STOPWORDS)

    def score(s: str) -> float:
        ws = re.findall(r"\b[a-zA-Z]{4,}\b", s.lower())
        return sum(freq.get(w, 0) for w in ws) / (len(ws) + 1)

    ranked = sorted(sentences, key=score, reverse=True)
    top = ranked[:5]
    # Restore original order
    ordered = [s for s in sentences if s in top]
    return " ".join(ordered)[:max_length * 3]


def generate_quiz(text: str, topic_name: str, num_questions: int = 5) -> list[dict[str, Any]]:
    """
    Generate MCQ quiz from text using a template-based NLP approach.
    Returns a list of question dicts.
    """
    questions: list[dict[str, Any]] = []
    sentences = _sentences(text)

    # Filter sentences that have enough content
    rich = [s for s in sentences if len(s.split()) >= 8]

    if not rich:
        return _fallback_quiz(topic_name, num_questions)

    used: set[int] = set()

    for i, sent in enumerate(rich):
        if len(questions) >= num_questions:
            break
        if i in used:
            continue

        q = _sentence_to_question(sent, text, topic_name)
        if q:
            questions.append(q)
            used.add(i)

    # Pad with fallback questions if needed
    while len(questions) < num_questions:
        questions.extend(_fallback_quiz(topic_name, num_questions - len(questions)))

    return questions[:num_questions]


def _sentence_to_question(sent: str, full_text: str, topic: str) -> dict[str, Any] | None:
    """Convert a sentence into a fill-the-blank or factual MCQ."""
    words = sent.split()
    # Pick a meaningful word to blank out
    candidates = [
        (i, w) for i, w in enumerate(words)
        if len(w) > 4
        and w.lower() not in _STOPWORDS
        and w[0].isalpha()
        and not w.isupper()  # skip acronyms
    ]
    if not candidates:
        return None

    # Pick the last candidate (usually the subject of the sentence)
    idx, answer_word = candidates[-1]
    clean_answer = answer_word.strip(string.punctuation)
    if len(clean_answer) < 3:
        return None

    # Build blanked question
    blanked = words.copy()
    blanked[idx] = "______"
    question_text = f"Complete the sentence: \"{' '.join(blanked)}\""

    # Generate distractors from the document's vocabulary
    all_words = re.findall(r"\b[A-Za-z]{4,}\b", full_text)
    distractors = list({
        w for w in all_words
        if w.lower() != clean_answer.lower()
        and w.lower() not in _STOPWORDS
        and len(w) > 3
        and w[0].isalpha()
    })

    # Sample 3 distractors
    import random
    random.shuffle(distractors)
    distractors = distractors[:3]

    if len(distractors) < 3:
        distractors += [f"None of these ({i})" for i in range(3 - len(distractors))]

    options = distractors[:3] + [clean_answer]
    random.shuffle(options)
    correct_idx = options.index(clean_answer)

    return {
        "question": question_text,
        "options": options,
        "answer": correct_idx,
        "explanation": f'The correct word is "{clean_answer}". Original sentence: {sent}',
    }


def _fallback_quiz(topic: str, n: int) -> list[dict[str, Any]]:
    """Generic conceptual questions when text-based generation falls short."""
    templates = [
        {
            "question": f"Which of the following best describes '{topic}'?",
            "options": [
                f"A fundamental concept in the study of {topic}",
                "An unrelated field of science",
                "A historical event from the 18th century",
                "A programming language",
            ],
            "answer": 0,
            "explanation": f"'{topic}' is described as a fundamental concept in this material.",
        },
        {
            "question": f"Why is understanding '{topic}' important?",
            "options": [
                "It is not relevant to modern study",
                f"It provides foundational knowledge about {topic}",
                "It only applies to advanced researchers",
                "It is purely theoretical with no applications",
            ],
            "answer": 1,
            "explanation": f"Understanding {topic} provides foundational knowledge.",
        },
        {
            "question": f"What is a key characteristic of '{topic}'?",
            "options": [
                "It cannot be studied or measured",
                "It has no real-world applications",
                f"It is a core concept in its domain",
                "It was disproven in recent years",
            ],
            "answer": 2,
            "explanation": f"{topic} is considered a core concept in its domain.",
        },
    ]
    import random
    random.shuffle(templates)
    return templates[:n]
