import re

from utils.ollama_client import query_ollama
from utils.vector_store import search_vector_store


NO_ANSWER_MESSAGE = "I could not find this in the provided documents."
NO_EVIDENCE_MESSAGE = "No evidence found"
UNKNOWN_MESSAGE = "Unknown"

FACTUAL_KEYWORDS = {
    "name",
    "email",
    "phone",
    "education",
    "skill",
    "skills",
    "language",
    "languages",
    "publication",
    "publications",
    "project",
    "projects",
    "entity",
    "owner",
    "system",
    "systems",
    "profit",
    "deadline",
}
LIST_TYPE_KEYWORDS = {
    "education": "Education",
    "skills": "Skills",
    "experience": "Experience",
    "projects": "Projects",
    "publications": "Publications",
    "publication": "Publications",
}
ASSESSMENT_KEYWORDS = (
    "pov",
    "opinion",
    "assess",
    "assessment",
    "evaluate",
    "evaluation",
    "strong",
    "weak",
    "good fit",
    "recommendation",
    "recommend",
    "summary",
    "summarize",
    "impression",
    "issues",
    "issue",
    "risk",
    "risks",
)
WORKFLOW_KEYWORDS = (
    "process",
    "workflow",
    "steps",
    "procedure",
    "handle",
    "handled",
    "how should",
    "how do",
    "how should we",
    "checklist",
)
STRATEGY_ANALYSIS_KEYWORDS = (
    "strategy",
    "planning",
    "plan",
    "profits",
    "profit",
    "estimated tax",
    "s corp",
    "s-corp",
)
YES_NO_STARTERS = ("does ", "do ", "is ", "are ", "has ", "have ", "can ", "did ", "was ", "were ")
PROOF_STOPWORDS = {"what", "which", "tell", "about", "document", "does", "know", "knows", "the", "is", "are"}
COMPOUND_CONNECTOR_PATTERN = re.compile(r"\s+(and|or|plus|also)\s+", re.IGNORECASE)


def _format_sources(scored_documents):
    """Convert retrieved chunks into a UI-friendly structure."""
    sources = []

    for document, score in scored_documents:
        sources.append(
            {
                "document_name": document.get("source", "Unknown document"),
                "document_type": document.get("document_type", "General Document"),
                "section_label": document.get("section_label", "Document"),
                "section_title": document.get("section_title", document.get("section_label", "Document")),
                "page_number": document.get("page_number"),
                "chunk_text": document.get("chunk_text", ""),
                "score": max(0.0, float(score)),
                "chunk_id": document.get("chunk_id"),
            }
        )

    return sources


def _calculate_confidence(sources, use_embeddings):
    """Estimate a simple confidence score from the retrieved chunks."""
    if not sources:
        return 0.0

    top_score = sources[0]["score"]
    average_score = sum(source["score"] for source in sources) / len(sources)

    if use_embeddings:
        confidence = (top_score * 0.60) + (average_score * 0.40)
        return min(max(confidence, 0.0), 0.98)

    fallback_confidence = 0.20 + (average_score * 1.35)
    return min(max(fallback_confidence, 0.0), 0.70)


def _detect_question_style(question):
    """Classify the question so the answer format matches the business need."""
    lowered_question = question.lower().strip()

    if any(keyword in lowered_question for keyword in ASSESSMENT_KEYWORDS):
        return "assessment"

    if any(keyword in lowered_question for keyword in WORKFLOW_KEYWORDS):
        return "workflow"

    if lowered_question.startswith(YES_NO_STARTERS):
        return "yes_no"

    if _detect_list_category(question):
        return "list"

    if any(keyword in lowered_question for keyword in FACTUAL_KEYWORDS):
        return "factual"

    return "complex"


def _detect_list_category(question):
    """Return the list category when the question asks for multiple items."""
    lowered_question = question.lower().strip()

    if lowered_question.startswith(YES_NO_STARTERS):
        return None

    if any(keyword in lowered_question for keyword in ASSESSMENT_KEYWORDS):
        return None

    for keyword, label in LIST_TYPE_KEYWORDS.items():
        if keyword in lowered_question:
            return label

    return None


def _determine_top_k(question, question_style):
    """Choose a retrieval depth based on the question type."""
    lowered_question = question.lower()
    if question_style in {"list", "workflow", "assessment", "compound"}:
        return 5
    if any(keyword in lowered_question for keyword in STRATEGY_ANALYSIS_KEYWORDS):
        return 5
    return 3


def _is_compound_question(question):
    """Return True when the question appears to contain multiple claims."""
    return bool(COMPOUND_CONNECTOR_PATTERN.search(question.strip()))


def _clean_claim_label(text):
    """Convert a subquestion into a short recruiter-friendly label."""
    cleaned_text = text.strip().rstrip("?.")
    cleaned_text = re.sub(
        r"^(have any|has any|have|has|know|knows|include|includes|list|lists|mention|mentions|show|shows|use|uses|used)\s+",
        "",
        cleaned_text,
        flags=re.IGNORECASE,
    )
    cleaned_text = re.sub(
        r"^(does|do|did|is|are|was|were|has|have|can)\s+(this client|the client|this firm|the firm|the business|the document)\s+",
        "",
        cleaned_text,
        flags=re.IGNORECASE,
    )
    cleaned_text = re.sub(r"^(what|which)\s+", "", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\b(is|are|was|were|does|do|did)\b.*$", "", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = cleaned_text.strip(" -:")

    if cleaned_text:
        return cleaned_text[:1].upper() + cleaned_text[1:]

    return "Claim"


def _infer_shared_prefix(first_part):
    """Infer the shared verb phrase for yes/no compound questions."""
    lowered_part = first_part.lower().strip()
    prefix_options = [
        "have any",
        "have",
        "has any",
        "has",
        "know",
        "knows",
        "include",
        "includes",
        "list",
        "lists",
        "mention",
        "mentions",
        "show",
        "shows",
        "use",
        "uses",
        "used",
        "a",
        "an",
    ]

    for prefix in prefix_options:
        if lowered_part.startswith(prefix):
            return first_part[: len(prefix)].strip()

    return ""


def _split_yes_no_question(question):
    """Split a yes/no question into separate claim questions when possible."""
    cleaned_question = question.strip().rstrip("?")
    pattern = re.compile(
        r"^(Does|Do|Did|Has|Have|Can|Is|Are|Was|Were)\s+(.+?)\s+(.+)$",
        re.IGNORECASE,
    )
    match = pattern.match(cleaned_question)

    if not match or not _is_compound_question(cleaned_question):
        return None

    auxiliary, subject, predicate = match.groups()
    claim_parts = [part.strip(" ,") for part in COMPOUND_CONNECTOR_PATTERN.split(predicate) if part.strip(" ,")]
    claim_parts = [part for index, part in enumerate(claim_parts) if index % 2 == 0]

    if len(claim_parts) < 2:
        return None

    shared_prefix = _infer_shared_prefix(claim_parts[0])
    subquestions = []

    for index, claim_part in enumerate(claim_parts):
        normalized_claim = claim_part
        lowered_claim = claim_part.lower()

        if index > 0 and shared_prefix and not lowered_claim.startswith(shared_prefix.lower()):
            normalized_claim = f"{shared_prefix} {claim_part}"

        subquestions.append(
            {
                "label": _clean_claim_label(normalized_claim),
                "question": f"{auxiliary} {subject} {normalized_claim}?".replace("  ", " "),
            }
        )

    return subquestions


def _split_what_question(question):
    """Split simple factual list questions such as education and skills."""
    cleaned_question = question.strip().rstrip("?")
    pattern = re.compile(
        r"^(What|Which)\s+(.+?)\s+(is|are|was|were|does|do|did)\s+(.+)$",
        re.IGNORECASE,
    )
    match = pattern.match(cleaned_question)

    if not match:
        return None

    opener, compound_subject, linking_verb, remainder = match.groups()
    if not _is_compound_question(compound_subject):
        return None

    subject_parts = [
        part.strip(" ,")
        for part in COMPOUND_CONNECTOR_PATTERN.split(compound_subject)
        if part.strip(" ,")
    ]
    subject_parts = [part for index, part in enumerate(subject_parts) if index % 2 == 0]

    if len(subject_parts) < 2:
        return None

    subquestions = []
    for subject_part in subject_parts:
        adjusted_verb = linking_verb
        if linking_verb.lower() in {"is", "are", "was", "were"}:
            if subject_part.lower().endswith("s"):
                adjusted_verb = "are" if linking_verb.lower() in {"is", "are"} else "were"
            else:
                adjusted_verb = "is" if linking_verb.lower() in {"is", "are"} else "was"

        subquestions.append(
            {
                "label": _clean_claim_label(subject_part),
                "question": f"{opener} {subject_part} {adjusted_verb} {remainder}?".replace("  ", " "),
            }
        )

    return subquestions


def _split_compound_question(question):
    """Split a compound question into atomic claim questions when possible."""
    yes_no_split = _split_yes_no_question(question)
    if yes_no_split:
        return yes_no_split

    factual_split = _split_what_question(question)
    if factual_split:
        return factual_split

    return None


def _build_prompt(question, context, question_style):
    """Create a prompt that fits the type of question asked."""
    if question_style == "yes_no":
        response_instruction = (
            "Answer with exactly one of the following values: Yes, Unknown, or No evidence found. "
            "Use Yes only when the context directly supports the claim."
        )
    elif question_style == "assessment":
        response_instruction = (
            "Write a short recruiter-friendly assessment in 2 to 4 sentences. Base it only on the retrieved evidence. "
            "Mention the strongest supporting signals first, then mention notable gaps or missing evidence if relevant. "
            f"If the retrieved context does not support a meaningful assessment, respond exactly with: {NO_ANSWER_MESSAGE}"
        )
    elif question_style == "workflow":
        response_instruction = (
            "Return a short ordered workflow grounded only in the retrieved documents. "
            "Use 3 to 6 numbered steps. If the process is not documented, respond exactly with: "
            f"{NO_ANSWER_MESSAGE}"
        )
    elif question_style == "factual":
        response_instruction = (
            "Answer with only the exact fact, name, skill, language, education item, email, phone, "
            f"or short phrase needed to answer the question. If the fact is not clearly present, respond exactly with: {NO_ANSWER_MESSAGE}"
        )
    else:
        response_instruction = (
            "Answer in 2 to 4 short sentences using only the context below. "
            "If the question asks for a strategy, recommendation, or planning takeaway, summarize the matching guidance directly from the documents. "
            f"If the answer is not clearly present, respond exactly with: {NO_ANSWER_MESSAGE}"
        )

    return f"""
You are TaxCopilot, a careful AI workflow assistant for tax and bookkeeping teams.
Use only the retrieved context below.

Instructions:
- {response_instruction}
- Do not invent, infer, or use outside knowledge.

Retrieved context:
{context}

Question:
{question}

Answer:
"""


def _build_list_prompt(question, context, list_label):
    """Create a prompt for list-style extraction across several chunks."""
    return f"""
You are TaxCopilot, a careful AI workflow assistant for tax and bookkeeping teams.
Use only the retrieved context below.

Instructions:
- Extract ALL unique {list_label.lower()} items that are clearly supported by the context.
- Merge information across multiple chunks.
- Do not invent missing details.
- Remove duplicates.
- Return only a clean bullet list with one item per line, each line starting with "- ".
- Preserve useful details like institution names, roles, dates, project names, technologies, or publication titles when available.
- If no {list_label.lower()} items are clearly present, respond exactly with:
{NO_ANSWER_MESSAGE}

Retrieved context:
{context}

Question:
{question}

Answer:
"""


def _normalize_answer(answer, question_style):
    """Trim model output into the expected demo-friendly format."""
    cleaned_answer = answer.strip().strip("'\"")
    cleaned_answer = re.sub(r"^answer:\s*", "", cleaned_answer, flags=re.IGNORECASE).strip()

    if question_style == "yes_no":
        lowered_answer = cleaned_answer.lower()
        if lowered_answer.startswith("yes"):
            return "Yes"
        if lowered_answer.startswith("no evidence"):
            return NO_EVIDENCE_MESSAGE
        if lowered_answer.startswith("no"):
            return NO_EVIDENCE_MESSAGE
        if lowered_answer.startswith("unknown"):
            return UNKNOWN_MESSAGE
        return UNKNOWN_MESSAGE

    if question_style == "factual":
        cleaned_answer = cleaned_answer.splitlines()[0].strip()
        cleaned_answer = re.sub(
            r"^(the )?(answer|name|email|phone|education|skills|languages?|publications?|projects?)\s*:\s*",
            "",
            cleaned_answer,
            flags=re.IGNORECASE,
        )
        cleaned_answer = re.sub(r"^(the name is|it is)\s+", "", cleaned_answer, flags=re.IGNORECASE)
    elif question_style in {"assessment", "complex"}:
        cleaned_answer = re.sub(r"^(overall|assessment|summary)\s*:\s*", "", cleaned_answer, flags=re.IGNORECASE)

    return cleaned_answer.strip()


def _normalize_list_answer(answer, list_label):
    """Normalize model output into a clean, deduplicated bullet list."""
    cleaned_answer = answer.strip().strip("'\"")
    cleaned_answer = re.sub(r"^answer:\s*", "", cleaned_answer, flags=re.IGNORECASE).strip()
    cleaned_answer = re.sub(
        rf"^{re.escape(list_label)}\s*:\s*",
        "",
        cleaned_answer,
        flags=re.IGNORECASE,
    ).strip()

    if cleaned_answer.lower() == NO_ANSWER_MESSAGE.lower():
        return NO_ANSWER_MESSAGE, []

    items = []
    for line in cleaned_answer.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            continue
        stripped_line = re.sub(r"^[-*•]\s*", "", stripped_line)
        stripped_line = re.sub(r"^\d+[\).\s-]*", "", stripped_line)
        stripped_line = stripped_line.strip(" -")
        if stripped_line:
            items.append(stripped_line)

    if not items and cleaned_answer:
        items = [segment.strip() for segment in cleaned_answer.split(";") if segment.strip()]

    unique_items = []
    seen_items = set()
    for item in items:
        normalized_item = re.sub(r"\s+", " ", item).strip()
        if not normalized_item:
            continue
        dedupe_key = normalized_item.casefold()
        if dedupe_key in seen_items:
            continue
        seen_items.add(dedupe_key)
        unique_items.append(normalized_item)

    if not unique_items:
        return NO_ANSWER_MESSAGE, []

    formatted_answer = f"{list_label}:\n" + "\n".join(f"- {item}" for item in unique_items)
    return formatted_answer, unique_items


def _clip_proof_text(text, max_chars=240):
    """Keep the proof snippet short and readable."""
    if len(text) <= max_chars:
        return text

    shortened_text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{shortened_text}..."


def _is_heading_like(line):
    """Detect short heading-like lines that need extra supporting text."""
    stripped_line = line.strip(" :-")
    words = stripped_line.split()

    if not stripped_line or len(words) > 4:
        return False

    return stripped_line.upper() == stripped_line and not any(character.isdigit() for character in stripped_line)


def _build_proof_snippet(question, answer, source, question_style="complex"):
    """Pick a short quote from the retrieved chunk that best supports the answer."""
    if not source or not source.get("chunk_text"):
        return None, -1

    chunk_text = source["chunk_text"]
    lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
    if not lines:
        return None, -1

    lowered_question = question.lower()
    question_terms = {
        token
        for token in re.findall(r"\b[a-zA-Z0-9@.+-]{2,}\b", lowered_question)
        if token not in PROOF_STOPWORDS
    }
    answer_terms = set()

    if answer not in {NO_ANSWER_MESSAGE, NO_EVIDENCE_MESSAGE, UNKNOWN_MESSAGE, "Yes"}:
        answer_terms = {
            token
            for token in re.findall(r"\b[a-zA-Z0-9@.+-]{2,}\b", answer.lower())
            if token not in PROOF_STOPWORDS
        }

    ranked_lines = []
    best_line = lines[0]
    best_score = -1

    for line_index, line in enumerate(lines):
        candidate_text = line
        if _is_heading_like(line) and line_index + 1 < len(lines):
            candidate_text = f"{line} — {lines[line_index + 1]}"

        lowered_line = candidate_text.lower()
        score = 0

        score += sum(3 for token in answer_terms if token in lowered_line)
        score += sum(1 for token in question_terms if token in lowered_line)

        if "name" in lowered_question and line_index == 0:
            score += 1
        if "education" in lowered_question and any(
            keyword in lowered_line for keyword in ["education", "university", "college", "degree", "bachelor", "master"]
        ):
            score += 2
        if any(keyword in lowered_question for keyword in ["skill", "skills", "language", "languages"]) and any(
            keyword in lowered_line for keyword in ["skills", "languages", "python", "sql", "quickbooks", "excel"]
        ):
            score += 2
        if "experience" in lowered_question and any(
            keyword in lowered_line for keyword in ["experience", "worked", "role", "responsible", "close", "review"]
        ):
            score += 2
        if any(keyword in lowered_question for keyword in ["strategy", "profits", "profit", "estimated tax", "s corp", "s-corp"]) and any(
            keyword in lowered_line for keyword in ["strategy", "profit", "payroll", "estimated tax", "s corporation", "s corp"]
        ):
            score += 2
        if any(keyword in lowered_question for keyword in ["risk", "risks", "issue", "issues"]) and any(
            keyword in lowered_line for keyword in ["risk", "issue", "uncategorized", "duplicate", "missing", "support"]
        ):
            score += 2
        if question_style == "workflow" and any(
            keyword in lowered_line for keyword in ["step", "review", "reclassify", "escalate", "document", "send", "follow"]
        ):
            score += 2

        if score > best_score:
            best_score = score
            best_line = candidate_text

        ranked_lines.append((candidate_text, score))

    if question_style in {"assessment", "list"}:
        ranked_lines.sort(key=lambda item: item[1], reverse=True)
        chosen_lines = []
        for candidate_text, score in ranked_lines:
            if score <= 0 and chosen_lines:
                continue
            if candidate_text not in chosen_lines:
                chosen_lines.append(candidate_text)
            if len(chosen_lines) == 2:
                break

        if chosen_lines:
            return _clip_proof_text(" | ".join(chosen_lines), max_chars=280), max(score for _, score in ranked_lines)

    return _clip_proof_text(best_line), best_score


def _build_citation(question, answer, source, question_style):
    """Create a source citation and proof snippet from the best retrieved chunk."""
    if not source:
        return None

    proof_snippet, proof_score = _build_proof_snippet(question, answer, source, question_style)
    citation = {
        "document_name": source["document_name"],
        "document_type": source["document_type"],
        "section_label": source["section_label"],
        "section_title": source["section_title"],
        "page_number": source["page_number"],
        "chunk_id": source["chunk_id"],
        "proof_snippet": proof_snippet,
    }

    if question_style == "yes_no" and answer == "Yes" and proof_score < 1:
        return None

    return citation


def _build_citations(question, answer, sources, question_style):
    """Create a clean list of citations and proof snippets from multiple chunks."""
    citations = []
    seen_citations = set()

    for source in sources:
        citation = _build_citation(question, answer, source, question_style)
        if citation is None:
            continue

        citation_key = (
            citation["document_name"],
            citation["section_label"],
            citation["chunk_id"],
        )
        if citation_key in seen_citations:
            continue

        seen_citations.add(citation_key)
        citations.append(citation)

    return citations


def _assemble_context(sources):
    """Build the retrieved context string used in generation prompts."""
    return "\n\n".join(
        [
            (
                f"Document: {source['document_name']}\n"
                f"Type: {source['document_type']}\n"
                f"Section: {source['section_title']}\n"
                f"Chunk: {source['chunk_id']}\n"
                f"Content: {source['chunk_text']}"
            )
            for source in sources
        ]
    )


def _answer_list_question(question, vector_store, label=None):
    """Answer list-type questions by merging evidence from several chunks."""
    list_label = _detect_list_category(question) or "Items"
    question_style = "list"
    scored_documents = search_vector_store(vector_store, question, k=_determine_top_k(question, question_style))
    sources = _format_sources(scored_documents)
    confidence = _calculate_confidence(sources, vector_store.use_embeddings)

    if not sources:
        return {
            "label": label or list_label,
            "question": question,
            "answer": NO_ANSWER_MESSAGE,
            "confidence": 0.0,
            "sources": [],
            "retrieval_mode": vector_store.retrieval_mode,
            "question_style": question_style,
            "citation": None,
            "citations": [],
            "debug": {"retrieved_chunks": []},
        }

    prompt = _build_list_prompt(question, _assemble_context(sources), list_label)
    answer = query_ollama(prompt).strip()
    normalized_answer, extracted_items = _normalize_list_answer(answer, list_label)

    if normalized_answer == NO_ANSWER_MESSAGE:
        return {
            "label": label or list_label,
            "question": question,
            "answer": NO_ANSWER_MESSAGE,
            "confidence": 0.0,
            "sources": sources,
            "retrieval_mode": vector_store.retrieval_mode,
            "question_style": question_style,
            "citation": None,
            "citations": [],
            "debug": {"retrieved_chunks": sources},
        }

    citations = _build_citations(question, "\n".join(extracted_items), sources, question_style)
    return {
        "label": label or list_label,
        "question": question,
        "answer": normalized_answer,
        "confidence": confidence,
        "sources": sources,
        "retrieval_mode": vector_store.retrieval_mode,
        "question_style": question_style,
        "citation": citations[0] if citations else None,
        "citations": citations,
        "debug": {"retrieved_chunks": sources},
    }


def _answer_single_question(question, vector_store, label=None):
    """Answer one atomic question with retrieved sources and a proof citation."""
    list_category = _detect_list_category(question)
    if list_category:
        return _answer_list_question(question, vector_store, label=label or list_category)

    question_style = _detect_question_style(question)
    scored_documents = search_vector_store(vector_store, question, k=_determine_top_k(question, question_style))
    sources = _format_sources(scored_documents)
    confidence = _calculate_confidence(sources, vector_store.use_embeddings)

    if not sources:
        return {
            "label": label or _clean_claim_label(question),
            "question": question,
            "answer": NO_ANSWER_MESSAGE,
            "confidence": 0.0,
            "sources": [],
            "retrieval_mode": vector_store.retrieval_mode,
            "question_style": question_style,
            "citation": None,
            "citations": [],
            "debug": {"retrieved_chunks": []},
        }

    prompt = _build_prompt(question, _assemble_context(sources), question_style)
    answer = query_ollama(prompt).strip()
    normalized_answer = _normalize_answer(answer, question_style)

    if normalized_answer.lower() == NO_ANSWER_MESSAGE.lower():
        return {
            "label": label or _clean_claim_label(question),
            "question": question,
            "answer": NO_ANSWER_MESSAGE,
            "confidence": 0.0,
            "sources": sources,
            "retrieval_mode": vector_store.retrieval_mode,
            "question_style": question_style,
            "citation": None,
            "citations": [],
            "debug": {"retrieved_chunks": sources},
        }

    citations = _build_citations(question, normalized_answer, sources, question_style)
    citation = citations[0] if citations else None

    if question_style == "yes_no":
        if normalized_answer == "Yes" and citation is None:
            normalized_answer = UNKNOWN_MESSAGE
        elif normalized_answer == NO_EVIDENCE_MESSAGE:
            citation = _build_citation(question, normalized_answer, sources[0], "complex")
            citations = [citation] if citation else []
        elif normalized_answer == UNKNOWN_MESSAGE:
            citations = []
            citation = None

    return {
        "label": label or _clean_claim_label(question),
        "question": question,
        "answer": normalized_answer,
        "confidence": confidence,
        "sources": sources,
        "retrieval_mode": vector_store.retrieval_mode,
        "question_style": question_style,
        "citation": citation,
        "citations": citations,
        "debug": {"retrieved_chunks": sources},
    }


def answer_question(question, vector_store):
    """Search documents and return an answer with source-backed proof."""
    subquestions = _split_compound_question(question)
    if subquestions:
        compound_results = [
            _answer_single_question(
                question=subquestion["question"],
                vector_store=vector_store,
                label=subquestion["label"],
            )
            for subquestion in subquestions
        ]

        flattened_sources = []
        for claim_result in compound_results:
            flattened_sources.extend(claim_result["sources"])

        return {
            "answer": None,
            "confidence": 0.0,
            "sources": flattened_sources,
            "retrieval_mode": vector_store.retrieval_mode,
            "question_style": "compound",
            "citation": None,
            "citations": [],
            "compound_results": compound_results,
            "debug": {"retrieved_chunks": flattened_sources},
        }

    single_result = _answer_single_question(question, vector_store)
    single_result["compound_results"] = []
    return single_result
