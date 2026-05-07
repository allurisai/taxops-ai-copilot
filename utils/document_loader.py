import re
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


SUPPORTED_FILE_TYPES = {".pdf", ".txt", ".csv"}
DOCUMENT_TYPES = [
    "SOP",
    "Client Profile",
    "Strategy Guide",
    "Financial Notes",
    "Meeting Notes",
    "Bookkeeping Data",
    "General Document",
]


def _normalize_text(text):
    """Clean extracted text while keeping readable line breaks."""
    cleaned_text = text.replace("\xa0", " ").replace("\u2022", "- ")
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    cleaned_text = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned_text)
    cleaned_text = re.sub(r" ?\n ?", "\n", cleaned_text)
    return cleaned_text.strip()


def _looks_like_heading(line):
    """Return True when a line resembles a document heading."""
    stripped_line = line.strip(" :-")
    if not stripped_line:
        return False
    if len(stripped_line) > 80:
        return False
    if stripped_line.endswith(":"):
        return True
    if stripped_line.isupper() and len(stripped_line.split()) <= 8:
        return True
    if stripped_line.lower().startswith(("section ", "step ", "workflow ", "process ", "checklist ")):
        return True
    return False


def detect_section_title(text, fallback_label):
    """Detect a section title from the current block of text when possible."""
    for line in text.splitlines():
        candidate = line.strip()
        if _looks_like_heading(candidate):
            return candidate.strip(" :")

    return fallback_label


def classify_document_type(source_name, text, file_type):
    """Classify the document into a tax and bookkeeping friendly type."""
    lowered_name = source_name.lower()
    lowered_text = text[:4000].lower()

    if file_type == "csv":
        return "Bookkeeping Data"

    if any(keyword in lowered_name for keyword in ["onboarding", "cleanup", "close", "reporting", "sop", "workflow", "checklist"]):
        return "SOP"
    if any(keyword in lowered_name for keyword in ["strategy", "optimization", "s_corp", "s-corp", "estimated_tax", "faq", "deduction"]):
        return "Strategy Guide"
    if "profile" in lowered_name:
        return "Client Profile"
    if any(keyword in lowered_name for keyword in ["meeting", "call_notes", "conversation"]):
        return "Meeting Notes"
    if any(keyword in lowered_name for keyword in ["financial", "review_notes", "quarterly", "summary", "notes"]):
        return "Financial Notes"

    if any(keyword in lowered_text for keyword in ["standard operating procedure", "workflow", "step 1", "step 2", "escalate", "triage"]):
        return "SOP"
    if any(keyword in lowered_text for keyword in ["client overview", "ownership", "entity type", "filing status", "client profile"]):
        return "Client Profile"
    if any(keyword in lowered_text for keyword in ["strategy", "tax planning", "s-corp", "estimated tax", "optimization"]):
        return "Strategy Guide"
    if any(keyword in lowered_text for keyword in ["meeting notes", "follow-up", "discussion", "next steps"]):
        return "Meeting Notes"
    if any(keyword in lowered_text for keyword in ["p&l", "cash flow", "profit", "bookkeeping issues", "financial risk"]):
        return "Financial Notes"

    return "General Document"


def extract_pdf_sections(file_bytes):
    """Extract readable text from each PDF page as its own searchable section."""
    reader = PdfReader(BytesIO(file_bytes))
    sections = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = _normalize_text(page.extract_text() or "")
        if not page_text:
            continue

        fallback_label = f"Page {page_number}"
        sections.append(
            {
                "text": page_text,
                "section_label": fallback_label,
                "section_title": detect_section_title(page_text, fallback_label),
                "page_number": page_number,
            }
        )

    return sections


def extract_text_from_txt(file_bytes):
    """Read a TXT file as UTF-8 text and ignore malformed characters."""
    return _normalize_text(file_bytes.decode("utf-8", errors="ignore"))


def extract_csv_sections(file_bytes, source_name):
    """Convert a CSV into searchable bookkeeping-friendly text blocks."""
    decoded_text = file_bytes.decode("utf-8", errors="ignore")
    dataframe = pd.read_csv(StringIO(decoded_text))
    dataframe.columns = [str(column).strip() for column in dataframe.columns]

    overview_lines = [
        f"File: {source_name}",
        f"Columns: {', '.join(dataframe.columns)}",
        f"Total rows: {len(dataframe)}",
    ]

    sections = [
        {
            "text": _normalize_text("\n".join(overview_lines)),
            "section_label": "Overview",
            "section_title": "Transactions Overview",
            "page_number": None,
        }
    ]

    chunk_size = 15
    for start_index in range(0, len(dataframe), chunk_size):
        chunk_df = dataframe.iloc[start_index : start_index + chunk_size]
        row_lines = []
        for row_number, (_, row) in enumerate(chunk_df.iterrows(), start=start_index + 1):
            values = [f"{column}: {row[column]}" for column in dataframe.columns]
            row_lines.append(f"Row {row_number}: " + " | ".join(values))

        section_text = _normalize_text("\n".join(row_lines))
        if not section_text:
            continue

        section_number = (start_index // chunk_size) + 1
        sections.append(
            {
                "text": section_text,
                "section_label": f"Rows {start_index + 1}-{start_index + len(chunk_df)}",
                "section_title": f"Transactions Batch {section_number}",
                "page_number": None,
            }
        )

    return sections


def _build_document_sections(source_name, file_bytes):
    """Build searchable sections from a single file payload."""
    file_extension = Path(source_name).suffix.lower()

    if file_extension == ".pdf":
        sections = extract_pdf_sections(file_bytes)
        file_type = "pdf"
    elif file_extension == ".txt":
        text = extract_text_from_txt(file_bytes)
        if not text:
            return []
        sections = [
            {
                "text": text,
                "section_label": "Document",
                "section_title": detect_section_title(text, "Document"),
                "page_number": None,
            }
        ]
        file_type = "txt"
    elif file_extension == ".csv":
        sections = extract_csv_sections(file_bytes, source_name)
        file_type = "csv"
    else:
        return []

    documents = []
    for section in sections:
        document_type = classify_document_type(source_name, section["text"], file_type)
        documents.append(
            {
                "text": section["text"],
                "source": source_name,
                "file_type": file_type,
                "document_type": document_type,
                "section_label": section["section_label"],
                "section_title": section["section_title"],
                "page_number": section["page_number"],
            }
        )

    return documents


def load_uploaded_documents(uploaded_files):
    """Convert uploaded files into page-aware dictionaries for local retrieval."""
    documents = []

    for uploaded_file in uploaded_files:
        documents.extend(
            _build_document_sections(
                uploaded_file.name,
                uploaded_file.getvalue(),
            )
        )

    return documents


def load_documents_from_paths(file_paths):
    """Load local files from disk into the same structure used by the uploader."""
    documents = []

    for file_path in file_paths:
        path = Path(file_path)
        if path.suffix.lower() not in SUPPORTED_FILE_TYPES or not path.is_file():
            continue

        documents.extend(
            _build_document_sections(
                path.name,
                path.read_bytes(),
            )
        )

    return documents


def _split_text_into_chunks(text, chunk_size=800, chunk_overlap=150):
    """Split text into overlapping chunks that preserve nearby context."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            for separator in ["\n\n", "\n", ". ", " "]:
                split_point = text.rfind(separator, start, end)
                if split_point > start + (chunk_size // 2):
                    end = split_point + len(separator)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = max(start + 1, end - chunk_overlap)
        start = next_start

    return chunks


def chunk_documents(documents, chunk_size=800, chunk_overlap=150):
    """Split each document section into chunks and retain tax-specific metadata."""
    chunked_documents = []

    for document in documents:
        chunks = _split_text_into_chunks(
            document["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_index, chunk_text in enumerate(chunks, start=1):
            section_title = detect_section_title(chunk_text, document["section_title"])
            search_text = (
                f"Document: {document['source']}\n"
                f"Type: {document['document_type']}\n"
                f"Section: {section_title}\n"
                f"Content: {chunk_text}"
            )
            chunked_documents.append(
                {
                    "chunk_text": chunk_text,
                    "search_text": search_text,
                    "source": document["source"],
                    "file_type": document["file_type"],
                    "document_type": document["document_type"],
                    "section_label": document["section_label"],
                    "section_title": section_title,
                    "page_number": document["page_number"],
                    "chunk_id": chunk_index,
                }
            )

    return chunked_documents
