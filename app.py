from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.data_cleaner import (
    highlight_review_rows,
    process_dataframe,
    read_csv_file,
    read_csv_path,
)
from utils.document_loader import chunk_documents, load_documents_from_paths, load_uploaded_documents
from utils.ollama_client import DEFAULT_OLLAMA_MODEL
from utils.qa_chain import NO_ANSWER_MESSAGE, NO_EVIDENCE_MESSAGE, UNKNOWN_MESSAGE, answer_question
from utils.summarizer import (
    explain_for_client,
    generate_client_report,
    generate_email,
    generate_explainer,
    generate_newsletter,
    generate_social_post,
)

# Maps the Strategy Content Studio selectbox value → the generator function.
# Routing is done here in app.py so it is visible and easy to debug.
_FORMAT_ROUTER = {
    "client email draft":     generate_email,
    "short newsletter draft": generate_newsletter,
    "educational explainer":  generate_explainer,
    "social post draft":      generate_social_post,
}
from utils.vector_store import build_vector_store


APP_TITLE = "TaxCopilot"
APP_SUBTITLE = "Internal AI Assistant for Tax & Bookkeeping Teams"
PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_DATA_DIR = PROJECT_ROOT / "data" / "demo_tax_firm"
DEMO_TRANSACTIONS_DIR = DEMO_DATA_DIR / "transactions"
SUPPORTED_UPLOAD_TYPES = ["pdf", "txt", "csv"]
NO_PROOF_ANSWERS = {NO_ANSWER_MESSAGE, NO_EVIDENCE_MESSAGE, UNKNOWN_MESSAGE}
FEATURE_CARDS = [
    {
        "title": "Internal AI Brain",
        "description": "Search SOPs, client notes, and strategy guides with proof-backed answers.",
        "input": "PDF, TXT, or CSV knowledge documents",
        "output": "Answer, source citation, and proof snippet",
        "icon": "🧠",
        "accent": "#6366f1",
    },
    {
        "title": "Bookkeeping Copilot",
        "description": "Review bookkeeping exports and surface rows that need manual attention.",
        "input": "Transaction CSV files",
        "output": "Review summary, flagged rows, and cleaned CSV",
        "icon": "📊",
        "accent": "#0ea5e9",
    },
    {
        "title": "Client Communication",
        "description": "Turn internal notes into client-ready summaries, action items, and email drafts.",
        "input": "Client notes, financial summaries, and strategy docs",
        "output": "Summary, issues, recommendations, and email draft",
        "icon": "💬",
        "accent": "#10b981",
    },
    {
        "title": "Strategy Content Studio",
        "description": "Convert internal tax strategy notes into polished outward-facing content.",
        "input": "Strategy notes or internal tax guidance",
        "output": "Explainer, newsletter, social post, or email",
        "icon": "✍️",
        "accent": "#f59e0b",
    },
    {
        "title": "Automations",
        "description": "Pre-built workflow triggers that connect document intake, bookkeeping, and client communication.",
        "input": "Trigger events and connected systems",
        "output": "Automated actions and notifications",
        "icon": "⚡",
        "accent": "#ef4444",
    },
]
INTERNAL_BRAIN_EXAMPLES = [
    "What is the client onboarding process?",
    "How should uncategorized transactions be handled?",
    "What issues does this client have?",
    "What strategy applies when profits exceed $100,000?",
    "Summarize the bookkeeping risks in this file.",
]
SAMPLE_QUESTIONS = [
    "What is the client onboarding process?",
    "How should uncategorized transactions be handled?",
    "What issues does this client have?",
    "What strategy applies to this client?",
    "Summarize the financial risks in this note.",
    "Does this client have missing transaction categories?",
]
STRATEGY_OUTPUT_OPTIONS = [
    "client email draft",
    "educational explainer",
    "short newsletter draft",
    "social post draft",
]
TONE_OPTIONS = ["Professional", "Educational", "Concise", "Client-friendly"]


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _initialize_session_state():
    """Create the shared state used by the full product experience."""
    defaults = {
        "workspace_name": "No workspace loaded",
        "workspace_kind": None,
        "workspace_signature": None,
        "workspace_documents": [],
        "workspace_chunks": [],
        "vector_store": None,
        "workspace_file_names": [],
        "workspace_document_types": [],
        "workspace_file_types": [],
        "workspace_loaded_at": None,
        "retrieval_mode": None,
        "brain_question": "",
        "run_brain_query": False,
        "last_brain_result": None,
        "last_brain_question": "",
        "last_brain_timestamp": None,
        "recent_queries": [],
        "show_debug_panels": False,
        "client_doc_selection": [],
        "strategy_doc_selection": [],
        "bookkeeping_demo_choice": "",
        "bookkeeping_download_name": "cleaned_transactions.csv",
        "dashboard_message": "",
        "client_report_output": None,
        "client_explanation_output": "",
        "strategy_output_text": "",
        "strategy_output_label": "",
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _apply_page_style():
    """Apply corporate professional styling."""
    st.markdown(
        """
        <style>
            /* ── Global font & base ─────────────────────────────────────── */
            html, body, [class*="css"], [class*="st-"] {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                             'Helvetica Neue', Arial, sans-serif !important;
            }
            .stApp { background: #f1f5f9; }
            .block-container {
                max-width: 1200px;
                padding-top: 1rem;
                padding-bottom: 2.5rem;
            }

            /* ── Sidebar ────────────────────────────────────────────────── */
            section[data-testid="stSidebar"] {
                min-width: 260px !important;
                max-width: 260px !important;
                background: #ffffff;
                border-right: 1px solid #e2e8f0;
            }
            section[data-testid="stSidebar"] > div {
                min-width: 260px !important;
                max-width: 260px !important;
            }
            [data-testid="stSidebar"] .block-container {
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            /* ── Tabs — clean underline style ───────────────────────────── */
            [data-testid="stTabs"] {
                border-bottom: 1px solid #e2e8f0;
            }
            [data-testid="stTabs"] button {
                border-radius: 0;
                padding: 0.65rem 1.1rem;
                font-weight: 500;
                font-size: 0.875rem;
                color: #64748b;
                background: transparent;
                border: none;
                border-bottom: 2px solid transparent;
                margin-bottom: -1px;
            }
            [data-testid="stTabs"] button:hover { color: #1e293b; }
            [data-testid="stTabs"] button[aria-selected="true"] {
                color: #2563eb;
                border-bottom: 2px solid #2563eb;
                background: transparent;
                box-shadow: none;
                font-weight: 600;
            }

            /* ── Buttons ────────────────────────────────────────────────── */
            div.stButton > button {
                border-radius: 6px;
                padding: 0.5rem 0.95rem;
                font-weight: 500;
                font-size: 0.875rem;
                border: 1px solid #e2e8f0;
                background: #ffffff;
                color: #1e293b;
                box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                transition: background 0.12s ease, border-color 0.12s ease, box-shadow 0.12s ease;
            }
            div.stButton > button:hover {
                background: #f8fafc;
                border-color: #cbd5e1;
                box-shadow: 0 1px 4px rgba(0,0,0,0.08);
            }
            div.stButton > button[kind="primary"] {
                background: #2563eb;
                color: #ffffff;
                border-color: #2563eb;
                box-shadow: 0 1px 3px rgba(37,99,235,0.25);
            }
            div.stButton > button[kind="primary"]:hover {
                background: #1d4ed8;
                border-color: #1d4ed8;
                box-shadow: 0 2px 6px rgba(37,99,235,0.3);
            }
            div.stDownloadButton > button {
                border-radius: 6px;
                font-weight: 500;
                background: #1e293b;
                color: #ffffff;
                border-color: #1e293b;
                box-shadow: none;
            }
            div.stDownloadButton > button:hover {
                background: #0f172a;
                border-color: #0f172a;
            }

            /* ── Containers ─────────────────────────────────────────────── */
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            }

            /* ── Metrics ────────────────────────────────────────────────── */
            [data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 0.85rem 1rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            }
            [data-testid="stMetricValue"] {
                font-size: 1.5rem;
                font-weight: 700;
                color: #1e293b;
            }
            [data-testid="stMetricLabel"] {
                color: #64748b;
                font-size: 0.8rem;
                font-weight: 500;
            }

            /* ── Inputs / file uploader / expanders ─────────────────────── */
            [data-testid="stFileUploader"] section {
                border-radius: 6px;
                border: 1px dashed #cbd5e1;
                background: #f8fafc;
            }
            [data-baseweb="input"] > div,
            [data-baseweb="textarea"] > div,
            [data-baseweb="select"] > div,
            [data-baseweb="tag"] { border-radius: 6px !important; }
            [data-testid="stExpander"] details {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: none;
            }

            /* ── HERO SECTION ───────────────────────────────────────────── */
            .hero-shell {
                background: #1e293b;
                border-radius: 8px;
                padding: 2.5rem 3rem;
                margin-bottom: 1.25rem;
                display: grid;
                grid-template-columns: 1fr auto;
                gap: 3rem;
                align-items: center;
                box-shadow: 0 2px 8px rgba(0,0,0,0.12);
            }
            .hero-badge {
                display: inline-block;
                padding: 0.2rem 0.6rem;
                background: rgba(37,99,235,0.18);
                border: 1px solid rgba(37,99,235,0.32);
                border-radius: 4px;
                color: #93c5fd;
                font-size: 0.64rem;
                font-weight: 700;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                margin-bottom: 0.6rem;
            }
            .hero-title {
                color: #ffffff !important;
                font-size: 2.4rem !important;
                font-weight: 700 !important;
                letter-spacing: -0.02em !important;
                margin: 0 0 0.4rem !important;
                line-height: 1.1 !important;
            }
            .hero-subtitle {
                color: #94a3b8;
                font-size: 0.9rem;
                margin: 0 0 1.1rem;
                line-height: 1.5;
            }
            .hero-chip {
                display: inline-block;
                padding: 0.2rem 0.6rem;
                background: rgba(255,255,255,0.07);
                border: 1px solid rgba(255,255,255,0.11);
                border-radius: 4px;
                color: #94a3b8;
                font-size: 0.72rem;
                font-weight: 500;
                margin-right: 0.35rem;
                margin-bottom: 0.25rem;
            }
            .hero-stats {
                display: grid;
                grid-template-columns: repeat(2, 148px);
                gap: 0.6rem;
            }
            .hero-stat-card {
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.09);
                border-radius: 6px;
                padding: 0.8rem 0.9rem;
            }
            .hero-stat-icon { font-size: 0.95rem; margin-bottom: 0.2rem; display: block; }
            .hero-stat-label {
                color: #64748b;
                font-size: 0.62rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.15rem;
            }
            .hero-stat-value {
                color: #e2e8f0;
                font-size: 0.82rem;
                font-weight: 600;
                word-break: break-word;
            }

            /* ── FEATURE CARDS GRID ─────────────────────────────────────── */
            .feat-grid {
                display: grid;
                gap: 0.75rem;
                margin-bottom: 1.1rem;
                align-items: stretch;
            }
            .feat-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-top: 2px solid var(--feat-accent, #2563eb);
                border-radius: 6px;
                padding: 1.2rem 1.25rem;
                display: flex;
                flex-direction: column;
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
                transition: box-shadow 0.15s ease, border-top-color 0.15s ease;
            }
            .feat-card:hover {
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            }
            .feat-icon {
                font-size: 1.4rem;
                margin-bottom: 0.55rem;
                display: block;
            }
            .feat-title {
                font-size: 0.9rem;
                font-weight: 600;
                color: #1e293b;
                margin: 0 0 0.35rem;
            }
            .feat-desc {
                font-size: 0.82rem;
                color: #475569;
                line-height: 1.55;
                margin: 0 0 auto;
                padding-bottom: 0.75rem;
            }
            .feat-io {
                font-size: 0.76rem;
                color: #94a3b8;
                line-height: 1.5;
                border-top: 1px solid #f1f5f9;
                padding-top: 0.65rem;
                margin-top: 0.1rem;
            }
            .feat-io strong { color: #64748b; }

            /* ── ACTION STRIP & WORKSPACE SECTION ───────────────────────── */
            .action-strip {
                display: grid;
                grid-template-columns: 1fr 1fr auto;
                gap: 0.75rem;
                align-items: center;
                margin-bottom: 1rem;
            }
            .action-meta { color: #64748b; font-size: 0.85rem; text-align: right; }
            .ws-section-title {
                font-size: 0.72rem;
                font-weight: 700;
                color: #94a3b8;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.6rem;
            }
            .ws-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.25rem;
                padding: 0.22rem 0.6rem;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 4px;
                color: #475569;
                font-size: 0.74rem;
                font-weight: 500;
                margin-right: 0.35rem;
                margin-bottom: 0.3rem;
            }

            /* ── MISC HELPERS ───────────────────────────────────────────── */
            .section-copy { color: #64748b; font-size: 0.875rem; margin-bottom: 0.35rem; }
            .chip-row { margin-top: 0.3rem; margin-bottom: 0.15rem; }
            .tag-chip {
                display: inline-block;
                margin-right: 0.35rem; margin-bottom: 0.35rem;
                padding: 0.2rem 0.6rem;
                border-radius: 4px;
                background: #f1f5f9; color: #334155;
                font-size: 0.74rem; font-weight: 500;
                border: 1px solid #e2e8f0;
            }
            .eyebrow { color: #64748b; font-size: 0.74rem; font-weight: 600; margin-bottom: 0.15rem; }
            .soft-note {
                background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px;
                padding: 0.9rem 1rem; margin-bottom: 0.75rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            }
            .soft-note h4 { margin: 0 0 0.2rem; color: #1e293b; font-size: 0.9rem; font-weight: 600; }
            .soft-note p, .soft-note li { color: #64748b; font-size: 0.85rem; line-height: 1.45; }
            .soft-note ul { margin: 0.3rem 0 0; padding-left: 1rem; }
            .microcopy { color: #64748b; font-size: 0.84rem; }
            .answer-proof {
                background: #f8fafc; border: 1px solid #e2e8f0;
                border-left: 3px solid #2563eb;
                padding: 0.75rem 0.9rem; border-radius: 6px;
                color: #334155; font-size: 0.88rem;
            }
            .module-tile { height: 100%; display: flex; flex-direction: column; gap: 0.5rem; }
            .module-tile-header { display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; }
            .module-tile-title { color: #1e293b; font-size: 0.9rem; font-weight: 600; }
            .module-pill {
                font-size: 0.7rem; padding: 0.18rem 0.45rem; border-radius: 4px;
                background: #eff6ff; color: #2563eb; border: 1px solid #dbeafe; font-weight: 500;
            }
            .module-copy { color: #64748b; font-size: 0.875rem; line-height: 1.5; }
            .module-meta { color: #475569; font-size: 0.8rem; line-height: 1.45; }
            .workspace-actions-note { color: #64748b; font-size: 0.8rem; margin-top: 0.15rem; }

            /* ── RESPONSIVE ─────────────────────────────────────────────── */
            @media (max-width: 960px) {
                .hero-shell { grid-template-columns: 1fr; padding: 2rem; gap: 2rem; }
                .hero-stats { grid-template-columns: repeat(4, 1fr); }
                .hero-title { font-size: 1.9rem !important; }
                .feat-grid { grid-template-columns: repeat(2, 1fr) !important; }
                .action-strip { grid-template-columns: 1fr; }
                .action-meta { text-align: left; }
            }
            @media (max-width: 640px) {
                .hero-stats { grid-template-columns: repeat(2, 1fr); }
                .feat-grid { grid-template-columns: 1fr !important; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _get_demo_file_paths():
    """Return the bundled synthetic demo dataset paths."""
    if not DEMO_DATA_DIR.exists():
        return tuple()

    file_paths = [
        str(path)
        for path in sorted(DEMO_DATA_DIR.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".csv"}
    ]
    return tuple(file_paths)


@st.cache_data(show_spinner=False)
def _load_cached_demo_documents(file_paths):
    """Load the sample workspace files from disk once."""
    return load_documents_from_paths(file_paths)


def _format_timestamp(timestamp_value):
    """Format a session timestamp for display."""
    if not timestamp_value:
        return "Not loaded"
    return datetime.fromisoformat(timestamp_value).strftime("%b %d, %Y at %I:%M %p")


def _dedupe_preserve_order(items):
    """Remove duplicates while preserving the original order."""
    seen_items = set()
    deduped_items = []

    for item in items:
        if item in seen_items:
            continue
        seen_items.add(item)
        deduped_items.append(item)

    return deduped_items


def _get_inventory_dataframe(documents):
    """Build a file-level inventory table for the current workspace."""
    inventory = {}

    for document in documents:
        source_name = document["source"]
        if source_name not in inventory:
            inventory[source_name] = {
                "Document": source_name,
                "Type": document.get("document_type", "General Document"),
                "File Type": str(document.get("file_type", "txt")).upper(),
                "Sections": 0,
            }

        inventory[source_name]["Sections"] += 1

    if not inventory:
        return pd.DataFrame(columns=["Document", "Type", "File Type", "Sections"])

    return pd.DataFrame(inventory.values()).sort_values(["Type", "Document"]).reset_index(drop=True)


def _get_default_doc_selection(documents, limit=4):
    """Choose a sensible default doc set for generation workflows."""
    preferred_types = {"Client Profile", "Financial Notes", "Meeting Notes", "Strategy Guide", "SOP"}
    preferred_files = _dedupe_preserve_order(
        [
            document["source"]
            for document in documents
            if document.get("file_type") != "csv" and document.get("document_type") in preferred_types
        ]
    )
    fallback_files = _dedupe_preserve_order(
        [document["source"] for document in documents if document.get("file_type") != "csv"]
    )
    selection = preferred_files or fallback_files
    return selection[:limit]


def _clear_query_results():
    """Reset the active Q&A output."""
    st.session_state["last_brain_result"] = None
    st.session_state["last_brain_question"] = ""
    st.session_state["last_brain_timestamp"] = None


def _refresh_generation_defaults():
    """Refresh doc selections used by the report and content modules."""
    documents = st.session_state.get("workspace_documents", [])
    available_files = _dedupe_preserve_order([document["source"] for document in documents if document.get("file_type") != "csv"])
    default_selection = _get_default_doc_selection(documents)

    for key in ["client_doc_selection", "strategy_doc_selection"]:
        existing_selection = [value for value in st.session_state.get(key, []) if value in available_files]
        st.session_state[key] = existing_selection or default_selection


def _build_workspace(documents, workspace_name, workspace_kind, workspace_signature):
    """Chunk documents, build retrieval state, and store the active workspace."""
    if not documents:
        raise ValueError(
            "No readable content was found. If you uploaded scanned PDFs, try text-based or OCR-processed files."
        )

    chunked_documents = chunk_documents(documents, chunk_size=800, chunk_overlap=150)
    if not chunked_documents:
        raise ValueError("The files were read successfully, but they did not produce searchable chunks.")

    vector_store = build_vector_store(chunked_documents)

    st.session_state["workspace_name"] = workspace_name
    st.session_state["workspace_kind"] = workspace_kind
    st.session_state["workspace_signature"] = workspace_signature
    st.session_state["workspace_documents"] = documents
    st.session_state["workspace_chunks"] = chunked_documents
    st.session_state["vector_store"] = vector_store
    st.session_state["workspace_file_names"] = _dedupe_preserve_order([document["source"] for document in documents])
    st.session_state["workspace_document_types"] = sorted(
        {document.get("document_type", "General Document") for document in documents}
    )
    st.session_state["workspace_file_types"] = sorted(
        {str(document.get("file_type", "txt")).upper() for document in documents}
    )
    st.session_state["workspace_loaded_at"] = datetime.now().isoformat()
    st.session_state["retrieval_mode"] = vector_store.retrieval_mode
    st.session_state["client_report_output"] = None
    st.session_state["client_explanation_output"] = ""
    st.session_state["strategy_output_text"] = ""
    st.session_state["strategy_output_label"] = ""
    _clear_query_results()
    _refresh_generation_defaults()


def _load_demo_workspace():
    """Load the bundled recruiter demo dataset into the active workspace."""
    demo_file_paths = _get_demo_file_paths()
    if not demo_file_paths:
        raise ValueError("The bundled demo dataset was not found.")

    demo_documents = _load_cached_demo_documents(demo_file_paths)
    _build_workspace(
        documents=demo_documents,
        workspace_name="Sample Tax Firm",
        workspace_kind="demo",
        workspace_signature=("demo",) + demo_file_paths,
    )
    st.session_state["dashboard_message"] = "Sample workspace loaded. You can start with the sample questions below."


def _index_uploaded_workspace(uploaded_files):
    """Build a workspace from newly uploaded user files."""
    if not uploaded_files:
        raise ValueError("Upload one or more PDF, TXT, or CSV files before indexing.")

    documents = load_uploaded_documents(uploaded_files)
    file_signature = tuple(sorted((uploaded_file.name, uploaded_file.size) for uploaded_file in uploaded_files))

    _build_workspace(
        documents=documents,
        workspace_name="Uploaded Internal Files",
        workspace_kind="uploaded",
        workspace_signature=file_signature,
    )
    st.session_state["dashboard_message"] = "Uploaded files indexed successfully."


def _queue_sample_question(question):
    """Populate a quick-start question and run it on the next rerun."""
    st.session_state["brain_question"] = question
    st.session_state["run_brain_query"] = True
    st.rerun()


def _record_recent_query(question, answer):
    """Track recent queries to make the demo flow easier to repeat."""
    recent_queries = st.session_state.get("recent_queries", [])
    recent_queries.insert(
        0,
        {
            "question": question,
            "answer_preview": answer[:140] if answer else "",
            "timestamp": datetime.now().strftime("%I:%M %p"),
        },
    )
    st.session_state["recent_queries"] = recent_queries[:6]


def _build_context_from_selected_files(selected_files, max_sections=10, max_chars=14000):
    """Assemble readable context blocks from selected workspace documents."""
    if not selected_files:
        return ""

    selected_documents = [
        document
        for document in st.session_state.get("workspace_documents", [])
        if document["source"] in selected_files and document.get("file_type") != "csv"
    ]

    context_blocks = []
    total_chars = 0

    for document in selected_documents:
        block = (
            f"Document: {document['source']}\n"
            f"Type: {document.get('document_type', 'General Document')}\n"
            f"Section: {document.get('section_title', document.get('section_label', 'Document'))}\n"
            f"Content:\n{document['text']}"
        )

        if total_chars + len(block) > max_chars and context_blocks:
            break

        if total_chars + len(block) > max_chars:
            remaining_chars = max_chars - total_chars
            block = block[:remaining_chars].rstrip()

        context_blocks.append(block)
        total_chars += len(block)

        if len(context_blocks) >= max_sections:
            break

    return "\n\n".join(context_blocks)


def _render_tag_row(tags):
    """Render metadata tags for loaded document types and file formats."""
    if not tags:
        return

    chips_html = "".join(f"<span class='tag-chip'>{tag}</span>" for tag in tags)
    st.markdown(f"<div class='chip-row'>{chips_html}</div>", unsafe_allow_html=True)


def _render_empty_state(title, description):
    """Render a minimal empty-state card."""
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.caption(description)


def _render_step_header(title, description=""):
    """Render a compact section header."""
    st.markdown(f"#### {title}")
    if description:
        st.caption(description)


def _render_output_card(title, content, height=None, key=None):
    """Render a consistent output card."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if height:
            st.text_area(title, value=content, height=height, key=key, label_visibility="collapsed")
        else:
            st.write(content)


def _render_selected_files_preview(uploaded_files):
    """Preview files selected in the uploader before indexing."""
    if not uploaded_files:
        return

    st.markdown("**Selected files**")
    for uploaded_file in uploaded_files:
        st.caption(f"- {uploaded_file.name}")


def _render_feature_cards():
    """Render all feature cards in a single CSS grid for guaranteed equal heights.
    4 cards → 4-column single row. 5 cards → 3+2 layout."""
    n = len(FEATURE_CARDS)
    cols = 4 if n <= 4 else 3

    cards_html = "".join(
        f"""<div class="feat-card" style="--feat-accent: {card.get('accent', '#2563eb')};">
                <span class="feat-icon">{card.get('icon', '✦')}</span>
                <div class="feat-title">{card['title']}</div>
                <div class="feat-desc">{card['description']}</div>
                <div class="feat-io">
                    <strong>Input</strong> &middot; {card['input']}<br/>
                    <strong>Output</strong> &middot; {card['output']}
                </div>
            </div>"""
        for card in FEATURE_CARDS
    )

    st.markdown(
        f'<div class="feat-grid" style="grid-template-columns: repeat({cols}, 1fr);">'
        f'{cards_html}</div>',
        unsafe_allow_html=True,
    )


def _render_dashboard():
    """Render the main product header and workspace summary."""
    workspace_name = st.session_state.get("workspace_name", "No workspace loaded")
    file_count = len(st.session_state.get("workspace_file_names", []))
    section_count = len(st.session_state.get("workspace_documents", []))
    chunk_count = len(st.session_state.get("workspace_chunks", []))
    retrieval_mode = st.session_state.get("retrieval_mode") or "None"
    last_loaded = _format_timestamp(st.session_state.get("workspace_loaded_at"))

    # Dark navy hero card with glassy stat grid
    st.markdown(
        f"""
        <div class="hero-shell">
            <div>
                <div class="hero-badge">Tax &amp; Bookkeeping Workspace</div>
                <h1 class="hero-title">{APP_TITLE}</h1>
                <p class="hero-subtitle">AI Assistant for Tax &amp; Bookkeeping Workflows</p>
                <span class="hero-chip">Document Search</span>
                <span class="hero-chip">Bookkeeping Review</span>
                <span class="hero-chip">Client Reports</span>
                <span class="hero-chip">Content Generation</span>
            </div>
            <div class="hero-stats">
                <div class="hero-stat-card">
                    <span class="hero-stat-icon">🗂️</span>
                    <div class="hero-stat-label">Workspace</div>
                    <div class="hero-stat-value">{workspace_name}</div>
                </div>
                <div class="hero-stat-card">
                    <span class="hero-stat-icon">📄</span>
                    <div class="hero-stat-label">Files</div>
                    <div class="hero-stat-value">{file_count}</div>
                </div>
                <div class="hero-stat-card">
                    <span class="hero-stat-icon">🔍</span>
                    <div class="hero-stat-label">Retrieval</div>
                    <div class="hero-stat-value">{retrieval_mode}</div>
                </div>
                <div class="hero-stat-card">
                    <span class="hero-stat-icon">🕐</span>
                    <div class="hero-stat-label">Updated</div>
                    <div class="hero-stat-value">{last_loaded}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Primary action strip
    hero_cta_col, hero_demo_col, hero_meta_col = st.columns([1.05, 1.05, 0.9], gap="small")
    if hero_cta_col.button(
        "Open Document Workspace",
        type="primary",
        use_container_width=True,
        key="hero_upload_documents_to_start",
    ):
        st.session_state["dashboard_message"] = (
            "Go to Internal AI Brain, upload files, and click Index Uploaded Files."
        )
        st.rerun()
    if hero_demo_col.button("Load Sample Workspace", use_container_width=True, key="hero_load_demo_workspace"):
        with st.spinner("Loading the sample tax firm workspace..."):
            _load_demo_workspace()
        st.rerun()
    with hero_meta_col:
        st.markdown(
            f"<div class='action-meta'>Model <strong>{DEFAULT_OLLAMA_MODEL}</strong><br/>Local mode</div>",
            unsafe_allow_html=True,
        )

    # Feature cards (icon + accent colour per module)
    _render_feature_cards()

    # Workspace detail / quick-start strip
    with st.container(border=True):
        workspace_loaded = bool(st.session_state.get("workspace_documents"))

        if workspace_loaded:
            type_counter = Counter(
                document.get("document_type", "General Document")
                for document in st.session_state.get("workspace_documents", [])
            )
            file_type_tags = st.session_state.get("workspace_file_types", [])
            meta_pills = [
                f"<span class='ws-pill'>Updated · {last_loaded}</span>",
                f"<span class='ws-pill'>Retrieval · {retrieval_mode}</span>",
                f"<span class='ws-pill'>{file_count} files · {section_count} sections · {chunk_count} chunks</span>",
            ]
            meta_pills += [f"<span class='ws-pill'>{ft}</span>" for ft in file_type_tags]
            st.markdown("<div class='ws-section-title'>Active Workspace</div>", unsafe_allow_html=True)
            st.markdown(f"<div>{''.join(meta_pills)}</div>", unsafe_allow_html=True)
            if type_counter:
                _render_tag_row([f"{label} • {count}" for label, count in sorted(type_counter.items())])
        else:
            st.markdown("<div class='ws-section-title'>Quick Start</div>", unsafe_allow_html=True)

        q1, q2, q3, q4 = st.columns(4)
        if q1.button("Load Sample Workspace", use_container_width=True, key="dashboard_load_demo_workspace"):
            with st.spinner("Loading the sample tax firm workspace..."):
                _load_demo_workspace()
            st.rerun()
        if q2.button("Run Sample Query", use_container_width=True, key="dashboard_ask_onboarding_question"):
            if not st.session_state.get("workspace_documents"):
                with st.spinner("Loading the sample workspace first..."):
                    _load_demo_workspace()
            _queue_sample_question("What is the client onboarding process?")
        if q3.button("Open Sample Transactions", use_container_width=True, key="dashboard_review_demo_transactions"):
            st.session_state["bookkeeping_demo_choice"] = "transactions_abc.csv"
            st.session_state["dashboard_message"] = (
                "Sample transactions are ready. Open the Bookkeeping Copilot tab to review them."
            )
            st.rerun()
        if q4.button("Prepare Client Report", use_container_width=True, key="dashboard_prepare_client_report"):
            if not st.session_state.get("workspace_documents"):
                with st.spinner("Loading the sample workspace first..."):
                    _load_demo_workspace()
            st.session_state["dashboard_message"] = (
                "The sample workspace is ready. Open the Client Communication tab and click Generate."
            )
            st.rerun()

    if st.session_state.get("dashboard_message"):
        st.success(st.session_state["dashboard_message"])
        st.session_state["dashboard_message"] = ""


def _render_sidebar():
    """Render a compact sidebar."""
    with st.sidebar:
        st.markdown("### Workspace")

        if st.session_state.get("workspace_documents"):
            st.caption(st.session_state.get("workspace_name", "Workspace loaded"))
            st.caption(
                f"{len(st.session_state.get('workspace_file_names', []))} files • "
                f"{len(st.session_state.get('workspace_documents', []))} sections • "
                f"{len(st.session_state.get('workspace_chunks', []))} chunks"
            )
            if st.session_state.get("workspace_document_types"):
                st.caption("Loaded document types")
                for document_type in st.session_state["workspace_document_types"]:
                    st.markdown(f"- {document_type}")
        else:
            st.caption("No workspace loaded")

        st.markdown("### Files")
        st.markdown("- PDF and TXT for documents")
        st.markdown("- CSV for bookkeeping exports")

        recent_queries = st.session_state.get("recent_queries", [])
        if recent_queries:
            st.markdown("### Recent Queries")
            for item in recent_queries[:4]:
                st.caption(f"{item['timestamp']} • {item['question']}")

        feedback_log = st.session_state.get("feedback_log", [])
        if feedback_log:
            st.markdown("### Feedback")
            positive = sum(1 for f in feedback_log if f["feedback"] == "positive")
            negative = sum(1 for f in feedback_log if f["feedback"] == "negative")
            total = len(feedback_log)
            if total > 0:
                st.caption(f"👍 {positive} helpful · 👎 {negative} not helpful")
                st.caption(f"Satisfaction: {positive/total:.0%}")

        st.markdown("### Status")
        st.caption(f"Ollama model: `{DEFAULT_OLLAMA_MODEL}`")
        st.toggle("Show debug panels", key="show_debug_panels")
        if st.session_state.get("retrieval_mode"):
            st.caption(f"Retrieval mode: {st.session_state['retrieval_mode']}")


def _render_citation_blocks(result):
    """Render source and proof cards for the current answer."""
    citations = result.get("citations") or []
    if not citations and result.get("citation"):
        citations = [result["citation"]]

    if not citations or result.get("answer") in NO_PROOF_ANSWERS:
        return

    for citation in citations:
        with st.container(border=True):
            st.markdown("**Source**")
            st.write(
                f"{citation['document_name']} — {citation['section_label']} — Chunk {citation['chunk_id']}"
            )
            st.caption(f"Document type: {citation['document_type']}")
            if citation.get("proof_snippet"):
                st.markdown("**Proof**")
                st.markdown(f"<div class='answer-proof'>\"{citation['proof_snippet']}\"</div>", unsafe_allow_html=True)


def _render_source_chunk_list(sources, heading="Retrieved Source Chunks"):
    """Render the actual retrieved chunk text in expandable panels."""
    if not sources:
        st.info("No source chunks were available for this result.")
        return

    st.markdown(f"### {heading}")
    for index, source in enumerate(sources, start=1):
        title = (
            f"Source {index}: {source['document_name']} • {source['section_label']} • Chunk {source['chunk_id']}"
        )
        with st.expander(title):
            st.caption(
                f"Type: {source['document_type']} • "
                f"Section: {source['section_title']} • "
                f"Relevance: {source['score']:.2f}"
            )
            st.write(source["chunk_text"])


def _render_document_debug_panels(result):
    """Keep debug details available without making them noisy by default."""
    if not st.session_state.get("show_debug_panels"):
        return

    with st.expander("Developer Debug: Extracted Documents"):
        for document in st.session_state.get("workspace_documents", []):
            st.markdown(
                f"**{document['source']} • {document.get('document_type', 'General Document')} • "
                f"{document.get('section_label', 'Document')}**"
            )
            st.text(document["text"][:3500] or "[No text extracted]")

    with st.expander("Developer Debug: Chunks"):
        for chunk in st.session_state.get("workspace_chunks", []):
            st.markdown(
                f"**{chunk['source']} • {chunk.get('section_label', 'Document')} • Chunk {chunk['chunk_id']}**"
            )
            st.text(chunk["chunk_text"][:1800])

    if result:
        with st.expander("Developer Debug: Retrieved Chunks"):
            for source in result.get("debug", {}).get("retrieved_chunks", []):
                st.markdown(
                    f"**{source['document_name']} • {source['section_label']} • "
                    f"Chunk {source['chunk_id']} • Score {source['score']:.2f}**"
                )
                st.text(source["chunk_text"][:1800])


def _run_brain_query():
    """Execute the internal knowledge search flow."""
    vector_store = st.session_state.get("vector_store")
    question = st.session_state.get("brain_question", "").strip()

    if vector_store is None:
        st.warning("Load the sample workspace or index uploaded files before asking a question.")
        return

    if not question:
        st.warning("Enter a question to search your internal knowledge base.")
        return

    try:
        with st.spinner("Searching internal knowledge and grounding the answer..."):
            result = answer_question(question=question, vector_store=vector_store)
    except RuntimeError as error:
        st.error(str(error))
        return

    st.session_state["last_brain_result"] = result
    st.session_state["last_brain_question"] = question
    st.session_state["last_brain_timestamp"] = datetime.now().isoformat()
    _record_recent_query(question, result.get("answer") or "")


def _render_brain_result():
    """Render the answer, source, proof, and retrieved chunks."""
    result = st.session_state.get("last_brain_result")
    if not result:
        with st.container(border=True):
            st.markdown("**Answer**")
            st.caption("Results appear here")
        return

    timestamp_text = _format_timestamp(st.session_state.get("last_brain_timestamp"))
    st.caption(f"Generated at {timestamp_text}")

    if result.get("compound_results"):
        for claim_result in result["compound_results"]:
            with st.container(border=True):
                st.markdown(f"**Answer: {claim_result['label']}**")
                st.markdown(claim_result["answer"])
                confidence = claim_result.get("confidence", 0.0)
                if confidence >= 0.70:
                    st.caption(f"🟢 High confidence · {confidence:.0%}")
                elif confidence >= 0.40:
                    st.caption(f"🟡 Medium confidence · {confidence:.0%}")
                elif confidence > 0:
                    st.caption(f"🔴 Low confidence · Review source below · {confidence:.0%}")
                _render_citation_blocks(claim_result)

        for claim_result in result["compound_results"]:
            _render_source_chunk_list(
                claim_result["sources"],
                heading=f"{claim_result['label']} — Retrieved Chunks",
            )

        _render_document_debug_panels(result)
        return

    with st.container(border=True):
        st.markdown("**Answer**")
        st.markdown(result["answer"])
        confidence = result.get("confidence", 0.0)
        if confidence >= 0.70:
            st.caption(f"🟢 High confidence · {confidence:.0%}")
        elif confidence >= 0.40:
            st.caption(f"🟡 Medium confidence · {confidence:.0%}")
        elif confidence > 0:
            st.caption(f"🔴 Low confidence · Review source below · {confidence:.0%}")
        feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 1, 4])
        with feedback_col1:
            if st.button("👍 Helpful", key=f"feedback_up_{st.session_state.get('last_brain_timestamp', '')}"):
                st.session_state.setdefault("feedback_log", []).append({
                    "question": st.session_state.get("last_brain_question", ""),
                    "feedback": "positive",
                    "timestamp": st.session_state.get("last_brain_timestamp", ""),
                })
                st.toast("Thanks for the feedback!")
        with feedback_col2:
            if st.button("👎 Not helpful", key=f"feedback_down_{st.session_state.get('last_brain_timestamp', '')}"):
                st.session_state.setdefault("feedback_log", []).append({
                    "question": st.session_state.get("last_brain_question", ""),
                    "feedback": "negative",
                    "timestamp": st.session_state.get("last_brain_timestamp", ""),
                })
                st.toast("Thanks — we'll use this to improve.")
        _render_citation_blocks(result)

    if result["answer"] == NO_ANSWER_MESSAGE:
        st.info(
            "Try a more specific question, a different document, or a clearer text-based PDF. "
            "The retrieved chunks below show what the system found most relevant."
        )

    _render_source_chunk_list(result.get("sources", []))
    _render_document_debug_panels(result)


def _render_recent_queries():
    """Render a compact recent query history."""
    recent_queries = st.session_state.get("recent_queries", [])
    if not recent_queries:
        return

    with st.expander("Recent Query History"):
        for item in recent_queries:
            st.markdown(f"- `{item['timestamp']}` {item['question']}")


def _render_loaded_document_cards():
    """Show a simple list of the documents currently in the workspace."""
    inventory_df = _get_inventory_dataframe(st.session_state.get("workspace_documents", []))
    if inventory_df.empty:
        return

    st.markdown("#### Loaded Documents")
    for _, row in inventory_df.iterrows():
        with st.container(border=True):
            st.markdown(f"**{row['Document']}**")
            st.caption(f"{row['Type']} • {row['File Type']} • {row['Sections']} searchable sections")


def _render_workspace_inventory():
    """Render the loaded document inventory in a recruiter-friendly table."""
    inventory_df = _get_inventory_dataframe(st.session_state.get("workspace_documents", []))
    if inventory_df.empty:
        st.caption("Upload documents to begin")
        return

    st.markdown("#### Inventory")
    st.dataframe(inventory_df, use_container_width=True, hide_index=True)


def _render_internal_ai_brain_tab():
    """Render the highest-priority module: local document search and evidence-backed Q&A."""
    st.subheader("Internal AI Brain")
    st.markdown(
        "<p class='section-copy'>Search uploaded documents and review answer, source, and proof.</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.get("workspace_documents"):
        _render_empty_state(
            title="Upload documents to begin",
            description="PDF, TXT, or CSV",
        )

    left_col, right_col = st.columns([1, 1.25], gap="large")

    with left_col:
        with st.container(border=True):
            _render_step_header("Documents", "PDF, TXT, or CSV")

            if st.button("Load Sample Workspace", type="primary", use_container_width=True):
                with st.spinner("Loading the sample workspace..."):
                    _load_demo_workspace()
                st.rerun()

            uploaded_files = st.file_uploader(
                "Upload PDF, TXT, or CSV files",
                type=SUPPORTED_UPLOAD_TYPES,
                accept_multiple_files=True,
                key="brain_workspace_uploader",
                help="Upload text-based PDFs, TXT notes, or CSV exports. CSV files are converted into searchable transaction summaries for retrieval.",
            )
            _render_selected_files_preview(uploaded_files)

            if st.button("Index Uploaded Files", use_container_width=True):
                try:
                    with st.spinner("Indexing uploaded files..."):
                        _index_uploaded_workspace(uploaded_files)
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not build the workspace: {error}")

            if st.session_state.get("retrieval_mode") == "Keyword fallback":
                st.caption("🔍 Using keyword-based retrieval")

        if st.session_state.get("workspace_documents"):
            _render_loaded_document_cards()
        _render_workspace_inventory()

    with right_col:
        with st.container(border=True):
            _render_step_header("Ask", "Try a sample prompt or enter your own question")

            question_rows = [st.columns(2), st.columns(2), st.columns(2)]
            question_columns = question_rows[0] + question_rows[1] + question_rows[2]
            for index, sample_question in enumerate(SAMPLE_QUESTIONS):
                if question_columns[index].button(
                    sample_question,
                    key=f"sample_question_{index}",
                    use_container_width=True,
                ):
                    if not st.session_state.get("workspace_documents"):
                        with st.spinner("Loading the sample workspace first..."):
                            _load_demo_workspace()
                    _queue_sample_question(sample_question)

            st.text_area(
                "Ask a question about your uploaded documents",
                key="brain_question",
                height=110,
                placeholder="Ask about your SOPs, client notes, strategy guides, or financial documents",
            )

            ask_col, clear_col = st.columns([1.5, 1])
            ask_clicked = ask_col.button(
                "Search Internal Knowledge",
                type="primary",
                use_container_width=True,
            )
            clear_clicked = clear_col.button("Clear Question", use_container_width=True)

            if clear_clicked:
                st.session_state["brain_question"] = ""
                _clear_query_results()
                st.rerun()

            should_run = ask_clicked or st.session_state.pop("run_brain_query", False)
            if should_run:
                _run_brain_query()

        _render_brain_result()
        _render_recent_queries()


def _resolve_demo_csv_path(selected_demo_file):
    """Resolve a bundled transactions file from the demo dataset."""
    if not selected_demo_file:
        return None

    candidate_path = DEMO_TRANSACTIONS_DIR / selected_demo_file
    if candidate_path.exists():
        return candidate_path
    return None


def _load_bookkeeping_dataframe(uploaded_csv):
    """Load either an uploaded CSV or a bundled demo CSV."""
    if uploaded_csv is not None:
        return read_csv_file(uploaded_csv), uploaded_csv.name

    selected_demo_file = st.session_state.get("bookkeeping_demo_choice")
    demo_path = _resolve_demo_csv_path(selected_demo_file)
    if demo_path is not None:
        return read_csv_path(demo_path), demo_path.name

    return None, ""


def _render_bookkeeping_copilot_tab():
    """Render the bookkeeping cleanup and triage workflow."""
    st.subheader("Bookkeeping Copilot")
    st.markdown(
        "<p class='section-copy'>Review bookkeeping CSVs and surface rows that need attention.</p>",
        unsafe_allow_html=True,
    )

    control_col, summary_col = st.columns([1.05, 1.2], gap="large")

    with control_col:
        with st.container(border=True):
            _render_step_header("Data", "Upload a transaction CSV")

            uploaded_csv = st.file_uploader(
                "Upload a bookkeeping CSV",
                type=["csv"],
                key="bookkeeping_uploader",
            )

            demo_a_col, demo_b_col = st.columns(2)
            if demo_a_col.button("Use Sample Transactions: ABC", use_container_width=True):
                st.session_state["bookkeeping_demo_choice"] = "transactions_abc.csv"
                st.rerun()
            if demo_b_col.button("Use Sample Transactions: XYZ", use_container_width=True):
                st.session_state["bookkeeping_demo_choice"] = "transactions_xyz.csv"
                st.rerun()

            st.caption("Suggested categories are assistive only and should be reviewed before use.")

        raw_df, source_name = _load_bookkeeping_dataframe(uploaded_csv)
        if raw_df is None:
            _render_empty_state(
                title="Upload a bookkeeping CSV to begin",
                description="CSV only",
            )
            return

        with st.spinner("Reviewing the transaction file and preparing suggestions..."):
            cleaned_df, report = process_dataframe(raw_df)

        st.session_state["bookkeeping_download_name"] = f"cleaned_{source_name}"

        st.markdown("#### Review Summary")
        metric1, metric2 = st.columns(2)
        metric3, metric4 = st.columns(2)
        metric1.metric("Total Rows", report["total_rows"])
        metric2.metric("Missing Category Matches", report["missing_category_count"])
        metric3.metric("Duplicate Rows", report["duplicate_count"])
        metric4.metric("Rows Needing Review", report["review_count"])
        st.caption(
            f"Unusual amount flags: {report['anomaly_count']} • Amount column detected: {report['amount_column']}"
        )

        if report["vendor_suggestions"]:
            st.markdown("#### Vendor Normalization Suggestions")
            st.dataframe(
                pd.DataFrame(report["vendor_suggestions"]).rename(
                    columns={
                        "raw_vendor": "Raw Vendor",
                        "normalized_vendor": "Normalized Vendor",
                        "count": "Rows",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        if report["missing_value_counts"]:
            st.markdown("#### Missing Value Counts")
            st.json(report["missing_value_counts"])

    with summary_col:
        with st.container(border=True):
            _render_step_header("Review", "")
            st.caption(f"Current file: {source_name}")

        rows_needing_review = cleaned_df[cleaned_df["Needs Review"]]
        with st.container(border=True):
            st.markdown("#### Rows Needing Attention")
            if rows_needing_review.empty:
                st.success("No rows currently require manual review.")
            else:
                st.dataframe(rows_needing_review.head(12), use_container_width=True, hide_index=True)

        review_tabs = st.tabs(["Uploaded Data", "Cleaned Output", "Rows Needing Attention"])

        with review_tabs[0]:
            st.caption(f"Source file: {source_name}")
            st.dataframe(raw_df, use_container_width=True, hide_index=True)

        with review_tabs[1]:
            st.dataframe(
                cleaned_df.style.apply(highlight_review_rows, axis=1),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download Cleaned CSV",
                data=cleaned_df.to_csv(index=False).encode("utf-8"),
                file_name=st.session_state.get("bookkeeping_download_name", "cleaned_transactions.csv"),
                mime="text/csv",
            )

        with review_tabs[2]:
            if rows_needing_review.empty:
                st.success("No rows currently require manual review.")
            else:
                st.dataframe(rows_needing_review, use_container_width=True, hide_index=True)


def _get_non_csv_workspace_files():
    """Return non-CSV file names for content-generation workflows."""
    return _dedupe_preserve_order(
        [
            document["source"]
            for document in st.session_state.get("workspace_documents", [])
            if document.get("file_type") != "csv"
        ]
    )


def _render_client_communication_tab():
    """Render the client-ready report generation workflow."""
    st.subheader("Client Communication")
    st.markdown(
        "<p class='section-copy'>Generate structured client-ready outputs from uploaded documents.</p>",
        unsafe_allow_html=True,
    )

    available_files = _get_non_csv_workspace_files()
    if not available_files:
        _render_empty_state(
            title="Select documents to generate a report",
            description="Load documents in Internal AI Brain first",
        )
        return

    control_col, output_col = st.columns([1, 1.25], gap="large")

    # Maps Client Communication format selector → generator function + label
    _CLIENT_FORMAT_ROUTER = {
        "Full Report":   (generate_client_report, "report"),
        "Client Email":  (generate_email,         "email"),
        "Explainer":     (generate_explainer,      "single"),
        "Newsletter":    (generate_newsletter,     "single"),
    }

    with control_col:
        with st.container(border=True):
            _render_step_header("Inputs", "Select source documents")
            st.multiselect(
                "Select supporting documents",
                options=available_files,
                key="client_doc_selection",
                help="Choose the client notes, strategy documents, or financial summaries you want to use.",
            )
            client_output_format = st.selectbox(
                "Output format",
                list(_CLIENT_FORMAT_ROUTER.keys()),
                key="client_output_format",
            )
            client_tone = st.selectbox(
                "Tone",
                TONE_OPTIONS,
                key="client_tone",
            )
            extra_instruction = st.text_area(
                "Optional guidance",
                height=90,
                key="client_extra_instruction",
                placeholder="e.g. summary only  /  3 lines  /  focus on bookkeeping issues",
            )
            generate_clicked = st.button(
                "Generate",
                type="primary",
                use_container_width=True,
            )

    selected_files = st.session_state.get("client_doc_selection", [])
    context = _build_context_from_selected_files(selected_files)

    with output_col:
        if generate_clicked:
            if not context:
                st.warning("Select one or more non-CSV documents first.")
            else:
                generator_fn, output_kind = _CLIENT_FORMAT_ROUTER[client_output_format]
                st.caption(
                    f"Format: **{client_output_format}** → `{generator_fn.__name__}()` | "
                    f"Tone: {client_tone} | Guidance: {extra_instruction.strip() or 'none'}"
                )
                try:
                    with st.spinner(f"Generating {client_output_format.lower()}..."):
                        if output_kind == "report":
                            result = generator_fn(context, extra_instruction=extra_instruction)
                            st.session_state["client_report_output"] = result
                        else:
                            text = generator_fn(context, guidance=extra_instruction, tone=client_tone)
                            st.session_state["client_report_output"] = {
                                "summary": text,
                                "key_issues": "",
                                "recommendations": "",
                                "action_items": "",
                                "client_email": "",
                            }
                        st.session_state["client_explanation_output"] = ""
                except RuntimeError as error:
                    st.error(str(error))

        report_output = st.session_state.get("client_report_output")
        if report_output:
            st.markdown("#### Output")
            _REPORT_SECTIONS = [
                ("Summary", "summary", None, None),
                ("Key Issues", "key_issues", None, None),
                ("Recommendations", "recommendations", None, None),
                ("Action Items", "action_items", None, None),
                ("Client Email Draft", "client_email", 240, "client_report_email_output"),
            ]
            for label, key, height, card_key in _REPORT_SECTIONS:
                value = report_output.get(key, "")
                if value:
                    _render_output_card(label, value, height=height, key=card_key)
        elif not st.session_state.get("client_explanation_output"):
            with st.container(border=True):
                _render_step_header("Output", "Generated output appears here")



def _render_strategy_content_studio_tab():
    """Render the strategy-to-content workflow."""
    st.subheader("Strategy Content Studio")
    st.markdown(
        "<p class='section-copy'>Create polished outward-facing content from internal strategy notes.</p>",
        unsafe_allow_html=True,
    )

    available_files = _get_non_csv_workspace_files()
    if not available_files:
        _render_empty_state(
            title="Select documents to generate content",
            description="Load documents in Internal AI Brain first",
        )
        return

    left_col, right_col = st.columns([1, 1.2], gap="large")

    with left_col:
        with st.container(border=True):
            _render_step_header("Inputs", "Choose documents, format, and tone")
            st.multiselect(
                "Select source documents",
                options=available_files,
                key="strategy_doc_selection",
                help="Choose strategy notes, client guidance, or internal messaging documents.",
            )
            output_type = st.selectbox(
                "Choose an output format",
                STRATEGY_OUTPUT_OPTIONS,
                key="strategy_output_type",
            )
            tone = st.selectbox(
                "Choose a tone",
                TONE_OPTIONS,
                key="strategy_tone",
            )
            extra_instruction = st.text_area(
                "Optional guidance",
                height=110,
                key="strategy_extra_instruction",
                placeholder="Example: keep the tone educational and avoid technical tax jargon.",
            )

            # Clear stale output whenever any input changes.
            _current_sig = (
                output_type,
                tone,
                tuple(sorted(st.session_state.get("strategy_doc_selection", []))),
            )
            if st.session_state.get("_strategy_input_sig") != _current_sig:
                st.session_state["strategy_output_text"] = ""
                st.session_state["strategy_output_label"] = ""
                st.session_state["_strategy_input_sig"] = _current_sig

            generate_clicked = st.button(
                "Generate Content",
                type="primary",
                use_container_width=True,
            )

    with right_col:
        if generate_clicked:
            selected_files = st.session_state.get("strategy_doc_selection", [])
            context = _build_context_from_selected_files(selected_files)

            if not context:
                st.warning("Select one or more non-CSV documents before generating strategy content.")
            else:
                generator_fn = _FORMAT_ROUTER.get(output_type.lower(), generate_explainer)
                st.caption(
                    f"Format: **{output_type}** → `{generator_fn.__name__}()` | "
                    f"Tone: {tone} | Guidance: {extra_instruction.strip() or 'none'}"
                )
                try:
                    with st.spinner(f"Generating {output_type}..."):
                        output_text = generator_fn(
                            context=context,
                            guidance=extra_instruction,
                            tone=tone,
                        )
                except RuntimeError as error:
                    st.error(str(error))
                else:
                    st.session_state["strategy_output_text"] = output_text
                    st.session_state["strategy_output_label"] = f"{output_type.title()} • {tone}"

        if st.session_state.get("strategy_output_text"):
            _render_output_card(
                st.session_state.get("strategy_output_label", "Generated Content"),
                st.session_state["strategy_output_text"],
                height=420,
                key="strategy_generated_output",
            )
        else:
            with st.container(border=True):
                _render_step_header("Output", "Generated content appears here")


def _render_automations_tab():
    """Render the Automations tab — pre-built workflow triggers and integration roadmap."""
    st.subheader("Automations")
    st.markdown(
        "<p class='section-copy'>Pre-built workflow automations that connect your internal systems.</p>",
        unsafe_allow_html=True,
    )

    AUTOMATIONS = [
        {
            "name": "New Client Onboarding",
            "trigger": "New client profile uploaded",
            "actions": [
                "Index client documents into AI Brain",
                "Generate welcome email draft",
                "Create onboarding checklist from SOP",
            ],
            "status": "Active",
            "icon": "👤",
            "integrations": "AI Brain → Client Communication → Email",
        },
        {
            "name": "Transaction Review Pipeline",
            "trigger": "CSV file uploaded to Bookkeeping Copilot",
            "actions": [
                "Auto-categorize transactions",
                "Flag duplicates and missing data",
                "Generate review summary for bookkeeper",
            ],
            "status": "Active",
            "icon": "📊",
            "integrations": "Bookkeeping Copilot → QuickBooks Online (planned)",
        },
        {
            "name": "Client Report Generation",
            "trigger": "Quarterly review date reached",
            "actions": [
                "Pull latest client notes and financials",
                "Generate structured report with action items",
                "Draft client email with key updates",
            ],
            "status": "Active",
            "icon": "📋",
            "integrations": "AI Brain → Client Communication → Email",
        },
        {
            "name": "Strategy Content Pipeline",
            "trigger": "New strategy guide added to workspace",
            "actions": [
                "Generate educational explainer",
                "Create newsletter draft",
                "Prepare social media post",
            ],
            "status": "Active",
            "icon": "✍️",
            "integrations": "AI Brain → Strategy Content Studio → Social/Email",
        },
        {
            "name": "QuickBooks Sync",
            "trigger": "Daily sync at 6:00 AM",
            "actions": [
                "Import latest transactions from QuickBooks Online",
                "Run auto-categorization",
                "Send review digest to bookkeeping team",
            ],
            "status": "Planned",
            "icon": "🔄",
            "integrations": "QuickBooks Online → Bookkeeping Copilot → Email",
        },
        {
            "name": "Document Change Monitor",
            "trigger": "SOP or strategy guide updated",
            "actions": [
                "Re-index updated document",
                "Notify team of changes",
                "Update AI Brain knowledge base",
            ],
            "status": "Planned",
            "icon": "📡",
            "integrations": "File System → AI Brain → Notifications",
        },
    ]

    active = [a for a in AUTOMATIONS if a["status"] == "Active"]
    planned = [a for a in AUTOMATIONS if a["status"] == "Planned"]

    st.markdown("#### Active Workflows")
    for automation in active:
        with st.container(border=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**{automation['icon']} {automation['name']}**")
                st.caption(f"Trigger: {automation['trigger']}")
                for action in automation["actions"]:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;→ {action}")
            with col2:
                st.markdown(f"**Status:** 🟢 {automation['status']}")
                st.caption(f"Flow: {automation['integrations']}")

    st.markdown("#### Planned Integrations")
    for automation in planned:
        with st.container(border=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**{automation['icon']} {automation['name']}**")
                st.caption(f"Trigger: {automation['trigger']}")
                for action in automation["actions"]:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;→ {action}")
            with col2:
                st.markdown(f"**Status:** 🟡 {automation['status']}")
                st.caption(f"Flow: {automation['integrations']}")

    with st.container(border=True):
        st.markdown("#### Integration Roadmap")
        st.caption("Planned connections to external systems")
        roadmap_cols = st.columns(4)
        roadmap_items = [
            ("QuickBooks Online", "Q3 2026", "🟡"),
            ("Zapier / Make / n8n", "Q3 2026", "🟡"),
            ("CRM Integration", "Q4 2026", "⚪"),
            ("Slack Notifications", "Q4 2026", "⚪"),
        ]
        for col, (name, timeline, dot) in zip(roadmap_cols, roadmap_items):
            with col:
                st.markdown(f"{dot} **{name}**")
                st.caption(timeline)


def main():
    _initialize_session_state()
    _apply_page_style()
    _render_sidebar()
    _render_dashboard()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Internal AI Brain",
            "Bookkeeping Copilot",
            "Client Communication",
            "Strategy Content Studio",
            "Automations",
        ]
    )

    with tab1:
        _render_internal_ai_brain_tab()

    with tab2:
        _render_bookkeeping_copilot_tab()

    with tab3:
        _render_client_communication_tab()

    with tab4:
        _render_strategy_content_studio_tab()

    with tab5:
        _render_automations_tab()


if __name__ == "__main__":
    main()
