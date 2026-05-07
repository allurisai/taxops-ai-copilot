from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Optional

import faiss
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
QUESTION_STOPWORDS = {
    "a",
    "about",
    "and",
    "are",
    "can",
    "described",
    "document",
    "does",
    "education",
    "experience",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "listed",
    "me",
    "mentioned",
    "name",
    "of",
    "skills",
    "tell",
    "the",
    "this",
    "what",
    "which",
    "who",
}
WORKFLOW_HINTS = {"process", "workflow", "steps", "procedure", "handle", "handled", "checklist", "review"}
RISK_HINTS = {"risk", "risks", "issue", "issues", "missing", "uncategorized", "inconsistent"}
BOOKKEEPING_HINTS = {"quickbooks", "bookkeeping", "transactions", "category", "categories", "expense"}
STRATEGY_HINTS = {"strategy", "profit", "profits", "estimated", "entity", "s-corp", "tax"}


@dataclass
class LocalVectorStore:
    """Container for the retrieval index and original text chunks."""
    index: Optional[faiss.IndexFlatIP]
    chunks: list
    use_embeddings: bool
    retrieval_mode: str


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the sentence-transformer model once and reuse it."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)


def _embed_texts(texts):
    """Create normalized embeddings for a list of strings."""
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype("float32")


def _keyword_overlap_score(question, chunk_text):
    """Boost chunks that share important words with the question."""
    lowered_question = question.lower()
    question_terms = {
        token
        for token in re.findall(r"\b[a-zA-Z0-9]{3,}\b", lowered_question)
        if token not in QUESTION_STOPWORDS
    }

    chunk_text_lower = chunk_text.lower()
    score = 0.0

    if question_terms:
        matched_terms = [token for token in question_terms if token in chunk_text_lower]
        coverage = len(matched_terms) / len(question_terms)
        score += coverage * 0.30

    if "name" in lowered_question and "name" in chunk_text_lower:
        score += 0.12
    if "education" in lowered_question and any(
        keyword in chunk_text_lower for keyword in ["education", "university", "college", "degree"]
    ):
        score += 0.12
    if "skills" in lowered_question and "skills" in chunk_text_lower:
        score += 0.12
    if "experience" in lowered_question and any(
        keyword in chunk_text_lower for keyword in ["experience", "worked", "role", "responsible"]
    ):
        score += 0.12
    if WORKFLOW_HINTS.intersection(question_terms) and any(keyword in chunk_text_lower for keyword in WORKFLOW_HINTS):
        score += 0.16
    if RISK_HINTS.intersection(question_terms) and any(keyword in chunk_text_lower for keyword in RISK_HINTS):
        score += 0.14
    if BOOKKEEPING_HINTS.intersection(question_terms) and any(keyword in chunk_text_lower for keyword in BOOKKEEPING_HINTS):
        score += 0.14
    if STRATEGY_HINTS.intersection(question_terms) and any(keyword in chunk_text_lower for keyword in STRATEGY_HINTS):
        score += 0.14

    return score


def build_vector_store(documents):
    """Create a FAISS vector store from chunked documents, with keyword fallback if needed."""
    if not documents:
        raise ValueError("No documents were available to index.")

    try:
        texts = [document["search_text"] for document in documents]
        embeddings = _embed_texts(texts)
        embedding_dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(embedding_dimension)
        index.add(embeddings)

        return LocalVectorStore(
            index=index,
            chunks=documents,
            use_embeddings=True,
            retrieval_mode="Semantic retrieval",
        )
    except Exception:
        return LocalVectorStore(
            index=None,
            chunks=documents,
            use_embeddings=False,
            retrieval_mode="Keyword fallback",
        )


def search_vector_store(vector_store, question, k=3):
    """Find the top chunks for a user question using hybrid ranking."""
    if not vector_store.chunks:
        return []

    matches = []

    if vector_store.use_embeddings and vector_store.index is not None:
        question_embedding = _embed_texts([question])
        result_count = min(max(k * 4, 8), len(vector_store.chunks))
        scores, indices = vector_store.index.search(question_embedding, result_count)

        for score, chunk_index in zip(scores[0], indices[0]):
            if chunk_index == -1:
                continue
            chunk = vector_store.chunks[chunk_index]
            hybrid_score = float(score) + _keyword_overlap_score(question, chunk["search_text"])
            matches.append((chunk, hybrid_score))
    else:
        for chunk in vector_store.chunks:
            matches.append((chunk, _keyword_overlap_score(question, chunk["search_text"])))

    matches.sort(key=lambda item: item[1], reverse=True)
    return matches[:k]
