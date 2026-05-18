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
        "description": "Retrieve suggested answers from uploaded SOPs, strategy guides, and client documents — grounded in source material to reduce hallucination risk.",
        "input": "PDF, TXT, or CSV knowledge documents",
        "output": "Suggested answer, source citation, and supporting excerpt",
        "icon": "🧠",
        "accent": "#6366f1",
    },
    {
        "title": "Bookkeeping Copilot",
        "description": "Surface transactions that need attention and suggest categories for bookkeeper review — outputs are review-ready, not import-ready.",
        "input": "Transaction CSV files",
        "output": "Review summary, flagged rows, and review-ready CSV export",
        "icon": "📊",
        "accent": "#0ea5e9",
    },
    {
        "title": "Client Communication",
        "description": "Draft client-facing summaries, action items, and email templates from internal notes — for advisor review before sending.",
        "input": "Client notes, financial summaries, and strategy docs",
        "output": "Draft summary, issues, recommendations, and email draft for advisor review",
        "icon": "💬",
        "accent": "#10b981",
    },
    {
        "title": "Strategy Content Studio",
        "description": "Convert internal strategy notes into draft client-friendly content — review for accuracy and alignment before publishing.",
        "input": "Strategy notes or internal tax guidance",
        "output": "Draft explainer, newsletter, social post, or email — for review before publishing",
        "icon": "✍️",
        "accent": "#f59e0b",
    },
    {
        "title": "Automations",
        "description": "Workflow blueprints showing how document intake, bookkeeping, and client communication can be connected after process validation.",
        "input": "Trigger events and connected systems",
        "output": "Workflow blueprint and integration roadmap",
        "icon": "⚡",
        "accent": "#ef4444",
    },
    {
        "title": "Client Dashboard",
        "description": "Track income, expenses, and financial health with directional insights for advisor review and client discussion.",
        "input": "Transaction CSV (type, amount, category, date)",
        "output": "Directional metrics, charts, and review-ready export",
        "icon": "📈",
        "accent": "#0891b2",
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
    "What is the onboarding process for a new client?",
    "What documents are needed before a strategy session?",
    "Summarize the bookkeeping cleanup SOP.",
    "What should the team do if client documents are missing?",
    "What strategy applies when profits exceed $100,000?",
    "How should uncategorized transactions be handled?",
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
        # ── Auth ──────────────────────────────────────────────────────────
        "authenticated": False,
        "user_name": "",
        "user_email": "",
        "login_view": "signin",
        "login_error": False,
        # ── App ───────────────────────────────────────────────────────────
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
        # ── Prototype usage KPIs ──────────────────────────────────────────
        "kpi_questions_asked": 0,
        "kpi_transactions_analyzed": 0,
        "kpi_issues_flagged": 0,
        "kpi_drafts_generated": 0,
        "kpi_content_pieces": 0,
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _apply_page_style():
    """Apply premium dark SaaS theme — Linear / Vercel aesthetic."""
    st.markdown(
        """
        <style>
        /* ════════════════════════════════════════════════════════════════
           GLOBAL RESET & FONT
        ════════════════════════════════════════════════════════════════ */
        *, *::before, *::after { box-sizing: border-box; }
        html, body, .main .block-container, p, span, div, label, input, textarea, select {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                         'Inter', 'Helvetica Neue', Arial, sans-serif !important;
            color: #FFFFFF;
        }

        /* ── Hide ALL default Streamlit branding & chrome ─────────────── */
        #MainMenu, footer, header { display: none !important; }
        [data-testid="stToolbar"]     { display: none !important; }
        [data-testid="stDecoration"]  { display: none !important; }
        [data-testid="stStatusWidget"]{ display: none !important; }

        /* ── Fix sidebar toggle icon text (Material Icons ligature fallback) */
        [data-testid="collapsedControl"] span,
        button[data-testid="baseButton-header"] span {
            font-size: 0 !important; visibility: hidden !important;
        }
        [data-testid="collapsedControl"] button::after,
        button[data-testid="baseButton-header"]::after {
            content: "›"; font-size: 1.4rem; font-weight: 700;
            visibility: visible !important; color: #8892A4; font-family: serif;
        }

        /* ════════════════════════════════════════════════════════════════
           APP BACKGROUND
        ════════════════════════════════════════════════════════════════ */
        .stApp {
            background: #0A0F1E !important;
        }
        .block-container {
            max-width: 1240px;
            padding-top: 1.25rem;
            padding-bottom: 3rem;
        }

        /* ════════════════════════════════════════════════════════════════
           SIDEBAR
        ════════════════════════════════════════════════════════════════ */
        section[data-testid="stSidebar"] {
            background: #0D1117 !important;
            border-right: 1px solid #1E2533 !important;
            min-width: 264px !important;
            max-width: 264px !important;
        }
        section[data-testid="stSidebar"] > div {
            min-width: 264px !important;
            max-width: 264px !important;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        /* Sidebar text */
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown { color: #8892A4 !important; }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 { color: #FFFFFF !important; font-size: 0.78rem !important;
            text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
        /* Sidebar nav items hover */
        [data-testid="stSidebar"] .stMarkdown li {
            padding: 0.35rem 0.6rem;
            border-radius: 6px;
            transition: background 0.2s ease;
            cursor: pointer;
        }
        [data-testid="stSidebar"] .stMarkdown li:hover { background: #1E2533; }

        /* ════════════════════════════════════════════════════════════════
           TABS
        ════════════════════════════════════════════════════════════════ */
        [data-testid="stTabs"] {
            border-bottom: 1px solid #1E2533;
            background: transparent;
        }
        [data-testid="stTabs"] button {
            border-radius: 0;
            padding: 0.7rem 1.15rem;
            font-weight: 500;
            font-size: 0.875rem;
            color: #8892A4 !important;
            background: transparent !important;
            border: none;
            border-bottom: 2px solid transparent;
            margin-bottom: -1px;
            transition: color 0.2s ease, border-color 0.2s ease;
        }
        [data-testid="stTabs"] button:hover { color: #FFFFFF !important; }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: #4F8EF7 !important;
            border-bottom: 2px solid #4F8EF7 !important;
            background: transparent !important;
            box-shadow: none;
            font-weight: 600;
        }

        /* ════════════════════════════════════════════════════════════════
           BUTTONS
        ════════════════════════════════════════════════════════════════ */
        div.stButton > button {
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            font-size: 0.875rem;
            border: 1px solid #2D3748;
            background: #161B27;
            color: #FFFFFF !important;
            box-shadow: none;
            transition: background 0.2s ease, border-color 0.2s ease,
                        transform 0.2s ease, box-shadow 0.2s ease;
        }
        div.stButton > button:hover {
            background: #1E2533;
            border-color: #4F8EF7;
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(79,142,247,0.2);
        }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #4F8EF7 0%, #7B5CF7 100%) !important;
            border: none !important;
            color: #FFFFFF !important;
            font-weight: 600;
            box-shadow: 0 2px 12px rgba(79,142,247,0.3);
        }
        div.stButton > button[kind="primary"]:hover {
            opacity: 0.9;
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(79,142,247,0.45);
        }
        div.stButton > button[kind="secondary"] {
            background: transparent !important;
            border: 1px solid #2D3748 !important;
            color: #8892A4 !important;
        }
        div.stButton > button[kind="secondary"]:hover {
            border-color: #4F8EF7 !important;
            color: #4F8EF7 !important;
            transform: translateY(-1px);
        }
        /* Download button */
        div.stDownloadButton > button {
            border-radius: 8px;
            background: linear-gradient(135deg, #4F8EF7 0%, #7B5CF7 100%);
            border: none;
            color: #FFFFFF !important;
            font-weight: 600;
            transition: opacity 0.2s ease, transform 0.2s ease;
        }
        div.stDownloadButton > button:hover {
            opacity: 0.88;
            transform: translateY(-1px);
        }

        /* ════════════════════════════════════════════════════════════════
           CARDS / CONTAINERS
        ════════════════════════════════════════════════════════════════ */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #161B27 !important;
            border: 1px solid #2D3748 !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.3) !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(79,142,247,0.3) !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
        }

        /* ════════════════════════════════════════════════════════════════
           METRIC CARDS
        ════════════════════════════════════════════════════════════════ */
        [data-testid="stMetric"] {
            background: #161B27 !important;
            border: 1px solid #2D3748 !important;
            border-left: 3px solid #4F8EF7 !important;
            border-radius: 12px !important;
            padding: 1.25rem 1.25rem !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.25) !important;
            transition: border-color 0.2s ease, transform 0.2s ease;
        }
        [data-testid="stMetric"]:hover {
            border-color: #4F8EF7 !important;
            transform: translateY(-1px);
        }
        [data-testid="stMetricValue"] {
            font-size: 1.6rem !important;
            font-weight: 700 !important;
            color: #FFFFFF !important;
        }
        [data-testid="stMetricLabel"] {
            color: #8892A4 !important;
            font-size: 0.78rem !important;
            font-weight: 500 !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

        /* ════════════════════════════════════════════════════════════════
           INPUT FIELDS
        ════════════════════════════════════════════════════════════════ */
        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stNumberInput"] input {
            background: #1E2533 !important;
            border: 1px solid #2D3748 !important;
            border-radius: 8px !important;
            color: #FFFFFF !important;
            font-size: 0.9rem !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        div[data-testid="stTextInput"] input::placeholder,
        div[data-testid="stTextArea"] textarea::placeholder { color: #4a5568 !important; }
        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stTextArea"] textarea:focus,
        div[data-testid="stNumberInput"] input:focus {
            border-color: #4F8EF7 !important;
            box-shadow: 0 0 0 3px rgba(79,142,247,0.15) !important;
            outline: none !important;
        }
        div[data-testid="stTextInput"] > div,
        div[data-testid="stTextArea"] > div { border: none !important; background: transparent !important; }
        div[data-testid="stTextInput"] label,
        div[data-testid="stTextArea"] label,
        div[data-testid="stNumberInput"] label { color: #8892A4 !important; font-size: 0.82rem !important; }

        /* ── Selectbox / multiselect — input box ──────────────────────── */
        [data-baseweb="select"] > div {
            background: #1E2533 !important;
            border: 1px solid #2D3748 !important;
            border-radius: 8px !important;
            color: #FFFFFF !important;
            transition: border-color 0.2s ease;
        }
        [data-baseweb="select"] > div:focus-within { border-color: #4F8EF7 !important; }
        [data-baseweb="input"] > div { background: #1E2533 !important; border-radius: 8px !important; }
        [data-baseweb="tag"] {
            background: rgba(79,142,247,0.15) !important;
            border: 1px solid rgba(79,142,247,0.3) !important;
            border-radius: 6px !important;
            color: #FFFFFF !important;
        }
        [data-baseweb="tag"] span { color: #FFFFFF !important; }

        /* ── Dropdown popup — THE white modal fix ──────────────────────
           Streamlit renders selectbox/multiselect options inside a
           floating [data-baseweb="popover"] container. Without this,
           the popup is white and covers the content beneath it.       */
        [data-baseweb="popover"],
        [data-baseweb="popover"] > div,
        [data-baseweb="popover"] > div > div {
            background: #1E2533 !important;
            border: 1px solid #2D3748 !important;
            border-radius: 8px !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.6) !important;
        }
        [data-baseweb="menu"],
        [data-baseweb="menu"] ul,
        ul[data-baseweb="menu-list"],
        [role="listbox"] {
            background: #1E2533 !important;
            border: 1px solid #2D3748 !important;
            border-radius: 8px !important;
        }
        [data-baseweb="option"],
        [role="option"] {
            background: #1E2533 !important;
            color: #FFFFFF !important;
        }
        [data-baseweb="option"]:hover,
        [role="option"]:hover {
            background: #2D3748 !important;
            color: #FFFFFF !important;
        }
        [data-baseweb="option"][aria-selected="true"],
        [role="option"][aria-selected="true"] {
            background: rgba(79,142,247,0.18) !important;
            color: #4F8EF7 !important;
        }

        /* ── Tooltip / help popup ──────────────────────────────────────── */
        [data-baseweb="tooltip"],
        [data-baseweb="tooltip"] > div,
        [data-testid="stTooltipContent"],
        [data-testid="stTooltipContent"] > div {
            background: #1E2533 !important;
            border: 1px solid #2D3748 !important;
            border-radius: 8px !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.5) !important;
        }

        /* ── Material Icons carve-out ─────────────────────────────────────
           The universal Inter override above breaks Material Symbols Rounded
           (used by Streamlit for expander arrows etc.), causing icon names
           to render as raw "_arr" text. Restore the correct font here.       */
        [data-testid="stIconMaterial"] {
            font-family: 'Material Symbols Rounded' !important;
            font-style: normal !important;
            font-weight: normal !important;
            font-size: 20px !important;
            line-height: 1 !important;
            display: inline-block !important;
            white-space: nowrap !important;
            direction: ltr !important;
            -webkit-font-feature-settings: 'liga' !important;
            font-feature-settings: 'liga' !important;
            text-rendering: optimizeLegibility !important;
        }
        /* Hide expander toggle arrow entirely — expander is already clickable */
        [data-testid="stExpander"] summary [data-testid="stIconMaterial"],
        [data-testid="stExpander"] summary [data-testid="stExpanderToggleIcon"],
        [data-testid="stExpander"] summary svg {
            display: none !important;
        }

        /* File uploader */
        [data-testid="stFileUploader"] section {
            background: #1E2533 !important;
            border: 1px dashed #2D3748 !important;
            border-radius: 10px !important;
            transition: border-color 0.2s ease;
        }
        [data-testid="stFileUploader"] section:hover { border-color: #4F8EF7 !important; }
        [data-testid="stFileUploadDropzone"] span { color: #8892A4 !important; font-size: 0.8rem !important; }
        [data-testid="stFileUploadDropzone"] small { color: #556070 !important; font-size: 0.72rem !important; }

        /* Expanders */
        [data-testid="stExpander"] details {
            background: #161B27 !important;
            border: 1px solid #2D3748 !important;
            border-radius: 10px !important;
            overflow: hidden;
        }
        [data-testid="stExpander"] summary {
            color: #FFFFFF !important;
            font-weight: 500;
        }
        [data-testid="stExpander"] summary:hover { color: #4F8EF7 !important; }

        /* Checkbox & Radio */
        div[data-testid="stCheckbox"] label p,
        div[data-testid="stRadio"] label p { color: #8892A4 !important; }
        div[data-testid="stCheckbox"] label p:hover,
        div[data-testid="stRadio"] label p:hover { color: #FFFFFF !important; }

        /* Toggle */
        div[data-testid="stToggle"] label { color: #8892A4 !important; }

        /* Dataframe / Table */
        [data-testid="stDataFrame"] {
            border: 1px solid #2D3748 !important;
            border-radius: 10px !important;
            overflow: hidden;
        }
        iframe[data-testid="stDataFrameResizable"] { border-radius: 10px !important; }

        /* Progress / Slider */
        [data-baseweb="slider"] [data-testid="stSlider"] { color: #4F8EF7 !important; }

        /* Caption / small text */
        .stCaption, [data-testid="stCaptionContainer"] { color: #8892A4 !important; }

        /* ── Alert / notification boxes ──────────────────────────────── */
        [data-testid="stAlert"] {
            border-radius: 10px !important;
            border-left-width: 3px !important;
        }
        div[data-testid="stInfo"] {
            background: rgba(79,142,247,0.08) !important;
            border-color: rgba(79,142,247,0.35) !important;
            color: #FFFFFF !important;
        }
        div[data-testid="stSuccess"] {
            background: rgba(16,185,129,0.08) !important;
            border-color: rgba(16,185,129,0.35) !important;
            color: #FFFFFF !important;
        }
        div[data-testid="stWarning"] {
            background: rgba(245,158,11,0.08) !important;
            border-color: rgba(245,158,11,0.35) !important;
            color: #FFFFFF !important;
        }
        div[data-testid="stError"] {
            background: rgba(239,68,68,0.08) !important;
            border-color: rgba(239,68,68,0.35) !important;
            color: #FFFFFF !important;
        }
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] span { color: #FFFFFF !important; }

        /* ── Form containers ──────────────────────────────────────────── */
        [data-testid="stForm"] {
            background: #161B27 !important;
            border: 1px solid #2D3748 !important;
            border-radius: 12px !important;
        }

        /* ── Number input ─────────────────────────────────────────────── */
        [data-testid="stNumberInput"] > div {
            background: #1E2533 !important;
            border: 1px solid #2D3748 !important;
            border-radius: 8px !important;
        }
        [data-testid="stNumberInput"] button {
            background: #2D3748 !important;
            border: none !important;
            color: #FFFFFF !important;
        }
        [data-testid="stNumberInput"] button:hover { background: #4F8EF7 !important; }

        /* ── Date / time inputs ───────────────────────────────────────── */
        [data-testid="stDateInput"] input,
        [data-testid="stTimeInput"] input {
            background: #1E2533 !important;
            border: 1px solid #2D3748 !important;
            border-radius: 8px !important;
            color: #FFFFFF !important;
        }
        [data-testid="stDateInput"] > div,
        [data-testid="stTimeInput"] > div {
            background: transparent !important;
            border: none !important;
        }

        /* ── Code blocks ──────────────────────────────────────────────── */
        [data-testid="stCode"],
        [data-testid="stCode"] > div,
        .stCode { background: #161B27 !important; border-radius: 8px !important; }
        code {
            background: rgba(79,142,247,0.1) !important;
            color: #4F8EF7 !important;
            border-radius: 4px !important;
            padding: 0.1em 0.3em !important;
        }
        pre code { background: transparent !important; color: #FFFFFF !important; padding: 0 !important; }

        /* ── JSON display ─────────────────────────────────────────────── */
        [data-testid="stJson"] { background: #161B27 !important; border-radius: 8px !important; }

        /* ── Tab content panel ────────────────────────────────────────── */
        [role="tabpanel"] { background: transparent !important; }

        /* ── General wrappers that can bleed white ────────────────────── */
        .element-container,
        .stColumn,
        [data-testid="column"],
        [data-testid="stHorizontalBlock"] { background: transparent !important; }
        .main, [data-testid="stAppViewContainer"],
        [data-testid="stMainBlockContainer"] { background: transparent !important; }

        /* ── Spinner ──────────────────────────────────────────────────── */
        [data-testid="stSpinner"] { color: #4F8EF7 !important; }

        /* ════════════════════════════════════════════════════════════════
           HERO SECTION
        ════════════════════════════════════════════════════════════════ */
        .hero-shell {
            background: linear-gradient(135deg, #111827 0%, #0D1117 100%);
            border: 1px solid #2D3748;
            border-radius: 14px;
            padding: 2.5rem 3rem;
            margin-bottom: 1.25rem;
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 3rem;
            align-items: center;
            box-shadow: 0 4px 24px rgba(0,0,0,0.4),
                        inset 0 1px 0 rgba(255,255,255,0.04);
        }
        .hero-badge {
            display: inline-block;
            padding: 0.2rem 0.65rem;
            background: rgba(79,142,247,0.15);
            border: 1px solid rgba(79,142,247,0.3);
            border-radius: 4px;
            color: #4F8EF7;
            font-size: 0.64rem; font-weight: 700;
            letter-spacing: 0.12em; text-transform: uppercase;
            margin-bottom: 0.65rem;
        }
        .hero-title {
            color: #FFFFFF !important;
            font-size: 2.4rem !important; font-weight: 700 !important;
            letter-spacing: -0.025em !important;
            margin: 0 0 0.4rem !important; line-height: 1.1 !important;
        }
        .hero-subtitle { color: #8892A4; font-size: 0.9rem; margin: 0 0 1.1rem; line-height: 1.55; }
        .hero-chip {
            display: inline-block; padding: 0.2rem 0.6rem;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 4px; color: #8892A4;
            font-size: 0.72rem; font-weight: 500;
            margin-right: 0.35rem; margin-bottom: 0.25rem;
            transition: border-color 0.2s ease, color 0.2s ease;
        }
        .hero-chip:hover { border-color: #4F8EF7; color: #4F8EF7; }
        .hero-stats { display: grid; grid-template-columns: repeat(2, 148px); gap: 0.65rem; }
        .hero-stat-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid #2D3748;
            border-radius: 8px; padding: 0.85rem 1rem;
            transition: border-color 0.2s ease;
        }
        .hero-stat-card:hover { border-color: rgba(79,142,247,0.4); }
        .hero-stat-icon { font-size: 1rem; margin-bottom: 0.25rem; display: block; }
        .hero-stat-label {
            color: #8892A4; font-size: 0.62rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.15rem;
        }
        .hero-stat-value { color: #FFFFFF; font-size: 0.85rem; font-weight: 600; word-break: break-word; }

        /* ════════════════════════════════════════════════════════════════
           FEATURE CARDS GRID
        ════════════════════════════════════════════════════════════════ */
        .feat-grid {
            display: grid; gap: 0.8rem;
            margin-bottom: 1.25rem; align-items: stretch;
        }
        .feat-card {
            background: #161B27;
            border: 1px solid #2D3748;
            border-top: 2px solid var(--feat-accent, #4F8EF7);
            border-radius: 12px; padding: 1.3rem 1.35rem;
            display: flex; flex-direction: column;
            transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
        }
        .feat-card:hover {
            border-color: var(--feat-accent, #4F8EF7);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        }
        .feat-icon { font-size: 1.5rem; margin-bottom: 0.6rem; display: block; }
        .feat-title { font-size: 0.9rem; font-weight: 600; color: #FFFFFF; margin: 0 0 0.4rem; }
        .feat-desc {
            font-size: 0.82rem; color: #8892A4;
            line-height: 1.55; margin: 0 0 auto; padding-bottom: 0.8rem;
        }
        .feat-io {
            font-size: 0.75rem; color: #4a5568;
            line-height: 1.5; border-top: 1px solid #2D3748;
            padding-top: 0.65rem; margin-top: 0.1rem;
        }
        .feat-io strong { color: #8892A4; }

        /* ════════════════════════════════════════════════════════════════
           WORKSPACE / ACTION STRIPS
        ════════════════════════════════════════════════════════════════ */
        .action-strip { display: grid; grid-template-columns: 1fr 1fr auto; gap: 0.75rem; align-items: center; margin-bottom: 1rem; }
        .action-meta { color: #8892A4; font-size: 0.85rem; text-align: right; }
        .ws-section-title {
            font-size: 0.7rem; font-weight: 700; color: #4a5568;
            text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.65rem;
        }
        .ws-pill {
            display: inline-flex; align-items: center; gap: 0.25rem;
            padding: 0.22rem 0.6rem;
            background: rgba(255,255,255,0.04);
            border: 1px solid #2D3748;
            border-radius: 4px; color: #8892A4;
            font-size: 0.73rem; font-weight: 500;
            margin-right: 0.35rem; margin-bottom: 0.3rem;
        }

        /* ════════════════════════════════════════════════════════════════
           MISC HELPERS
        ════════════════════════════════════════════════════════════════ */
        .section-copy { color: #8892A4; font-size: 0.875rem; margin-bottom: 0.35rem; }
        .chip-row { margin-top: 0.3rem; margin-bottom: 0.15rem; }
        .tag-chip {
            display: inline-block; margin-right: 0.35rem; margin-bottom: 0.35rem;
            padding: 0.2rem 0.6rem; border-radius: 4px;
            background: rgba(79,142,247,0.1); color: #4F8EF7;
            font-size: 0.73rem; font-weight: 500;
            border: 1px solid rgba(79,142,247,0.2);
        }
        .eyebrow { color: #8892A4; font-size: 0.74rem; font-weight: 600; margin-bottom: 0.15rem; }
        .soft-note {
            background: #161B27; border: 1px solid #2D3748; border-radius: 10px;
            padding: 0.9rem 1rem; margin-bottom: 0.75rem;
        }
        .soft-note h4 { margin: 0 0 0.2rem; color: #FFFFFF; font-size: 0.9rem; font-weight: 600; }
        .soft-note p, .soft-note li { color: #8892A4; font-size: 0.85rem; line-height: 1.45; }
        .soft-note ul { margin: 0.3rem 0 0; padding-left: 1rem; }
        .microcopy { color: #8892A4; font-size: 0.84rem; }
        .answer-proof {
            background: #161B27; border: 1px solid #2D3748;
            border-left: 3px solid #4F8EF7;
            padding: 0.8rem 1rem; border-radius: 8px;
            color: #FFFFFF; font-size: 0.88rem;
        }
        .module-tile { height: 100%; display: flex; flex-direction: column; gap: 0.5rem; }
        .module-tile-header { display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; }
        .module-tile-title { color: #FFFFFF; font-size: 0.9rem; font-weight: 600; }
        .module-pill {
            font-size: 0.7rem; padding: 0.18rem 0.48rem; border-radius: 4px;
            background: rgba(79,142,247,0.12); color: #4F8EF7;
            border: 1px solid rgba(79,142,247,0.25); font-weight: 500;
        }
        .module-copy { color: #8892A4; font-size: 0.875rem; line-height: 1.5; }
        .module-meta { color: #8892A4; font-size: 0.8rem; line-height: 1.45; }
        .workspace-actions-note { color: #8892A4; font-size: 0.8rem; margin-top: 0.15rem; }

        /* ════════════════════════════════════════════════════════════════
           BUSINESS IMPACT & VALUE FRAMING COMPONENTS
        ════════════════════════════════════════════════════════════════ */
        .why-matters {
            background: rgba(79,142,247,0.04); border: 1px solid rgba(79,142,247,0.12);
            border-left: 3px solid #4F8EF7; border-radius: 0 8px 8px 0;
            padding: 1rem 1.25rem; margin-bottom: 1rem;
        }
        .why-matters-title {
            font-size: 0.68rem; font-weight: 700; color: #4F8EF7;
            text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem;
        }
        .why-matters-text { font-size: 0.84rem; color: #8892A4; line-height: 1.65; }

        .impact-grid-3 {
            display: grid; grid-template-columns: repeat(3, 1fr);
            gap: 0.65rem; margin-bottom: 1rem;
        }
        .impact-card {
            background: #161B27; border: 1px solid #232D3F;
            border-radius: 8px; padding: 0.95rem 1.05rem;
            display: flex; flex-direction: column;
        }
        .impact-icon { font-size: 1.1rem; margin-bottom: 0.35rem; }
        .impact-title {
            font-size: 0.82rem; font-weight: 600; color: #F0F4F8; margin-bottom: 0.3rem;
        }
        .impact-desc { font-size: 0.75rem; color: #8892A4; line-height: 1.55; margin-bottom: 0.55rem; flex: 1; }
        .impact-metric {
            font-size: 0.63rem; font-weight: 700; color: #4F8EF7;
            text-transform: uppercase; letter-spacing: 0.07em;
            background: rgba(79,142,247,0.08); border: 1px solid rgba(79,142,247,0.15);
            border-radius: 4px; padding: 0.15rem 0.45rem; display: inline-block;
        }

        .ba-wrap { margin: 0.5rem 0 0.75rem; }
        .ba-row {
            display: grid; grid-template-columns: 80px 1fr;
            gap: 0; border: 1px solid #232D3F; border-radius: 6px; overflow: hidden;
            margin-bottom: 0.35rem;
        }
        .ba-label {
            font-size: 0.66rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
            padding: 0.55rem 0.7rem; background: #1C2333; display: flex; align-items: center;
        }
        .ba-before-lbl { color: #F59E0B; }
        .ba-after-lbl  { color: #10B981; }
        .ba-text {
            font-size: 0.78rem; color: #8892A4; line-height: 1.5;
            padding: 0.55rem 0.9rem; background: #161B27;
        }

        .measured-by { margin: 0.4rem 0 0.6rem; }
        .measured-by-title {
            font-size: 0.66rem; font-weight: 700; color: #556070;
            text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.35rem;
        }
        .measured-by-item {
            font-size: 0.77rem; color: #8892A4; line-height: 1.55;
            padding: 0.2rem 0 0.2rem 0.8rem; border-left: 2px solid #232D3F;
            margin-bottom: 0.2rem;
        }

        .vmap-wrap { overflow-x: auto; }
        .vmap-table { width: 100%; border-collapse: collapse; }
        .vmap-table th {
            font-size: 0.64rem; font-weight: 700; color: #556070;
            text-transform: uppercase; letter-spacing: 0.08em;
            padding: 0.5rem 0.85rem; background: #1C2333; border: 1px solid #232D3F; text-align: left;
        }
        .vmap-table td {
            font-size: 0.78rem; color: #8892A4; line-height: 1.5;
            padding: 0.6rem 0.85rem; border: 1px solid #232D3F; vertical-align: top;
        }
        .vmap-table td:first-child { color: #F0F4F8; font-weight: 500; font-size: 0.8rem; }
        .vmap-table td:nth-child(2) { color: #4F8EF7; font-size: 0.75rem; }
        .vmap-table tr:hover td { background: rgba(79,142,247,0.025); }

        .kpi-section-title {
            font-size: 0.64rem; font-weight: 700; color: #556070;
            text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.35rem;
        }
        .kpi-row {
            display: flex; justify-content: space-between; align-items: center;
            padding: 0.28rem 0; border-bottom: 1px solid #1A2232; font-size: 0.75rem;
        }
        .kpi-lbl { color: #8892A4; }
        .kpi-val { color: #F0F4F8; font-weight: 600; }

        .module-impact-copy {
            font-size: 0.8rem; color: #8892A4; line-height: 1.65;
            margin-bottom: 0.75rem; padding: 0.7rem 0.9rem;
            background: rgba(79,142,247,0.03); border: 1px solid rgba(79,142,247,0.08);
            border-radius: 6px;
        }

        @media (max-width: 768px) {
            .impact-grid-3 { grid-template-columns: 1fr 1fr; }
            .roadmap-grid  { grid-template-columns: 1fr 1fr; }
        }

        /* ════════════════════════════════════════════════════════════════
           PRODUCTION HARDENING ROADMAP
        ════════════════════════════════════════════════════════════════ */
        .roadmap-intro-copy {
            font-size: 0.8rem; color: #8892A4; line-height: 1.7;
            margin-bottom: 0.85rem; padding: 0.75rem 1rem;
            background: rgba(79,142,247,0.03); border: 1px solid rgba(79,142,247,0.1);
            border-radius: 8px;
        }
        .roadmap-grid {
            display: grid; grid-template-columns: repeat(3, 1fr);
            gap: 0.7rem; margin: 0.75rem 0 0.5rem;
        }
        .roadmap-card {
            background: #161B27; border: 1px solid #232D3F; border-radius: 10px;
            padding: 1rem 1.1rem; display: flex; flex-direction: column; gap: 0.4rem;
        }
        .roadmap-card-header {
            display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.1rem;
        }
        .roadmap-card-icon { font-size: 1.1rem; }
        .roadmap-card-title {
            font-size: 0.82rem; font-weight: 700; color: #F0F4F8;
        }
        .roadmap-chip {
            display: inline-block; padding: 0.13rem 0.55rem; border-radius: 20px;
            font-size: 0.59rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.08em; width: fit-content;
        }
        .chip-next {
            background: rgba(245,158,11,0.1); color: #F59E0B;
            border: 1px solid rgba(245,158,11,0.22);
        }
        .chip-production {
            background: rgba(16,185,129,0.08); color: #10B981;
            border: 1px solid rgba(16,185,129,0.2);
        }
        .roadmap-card-desc {
            font-size: 0.75rem; color: #8892A4; line-height: 1.6; margin-top: 0.15rem;
        }
        .roadmap-bullet {
            font-size: 0.72rem; color: #6B7585; padding: 0.1rem 0 0.1rem 0.6rem;
            border-left: 2px solid #232D3F; margin-bottom: 0.08rem; line-height: 1.5;
        }
        .roadmap-section-title {
            font-size: 0.64rem; font-weight: 700; color: #556070;
            text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem;
        }
        /* Prototype → Production table */
        .proto-table { width: 100%; border-collapse: collapse; font-size: 0.75rem; }
        .proto-table th {
            background: #161B27; color: #4F8EF7;
            font-size: 0.63rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.09em; padding: 0.5rem 0.75rem;
            border: 1px solid #232D3F; text-align: left;
        }
        .proto-table td {
            padding: 0.42rem 0.75rem; border: 1px solid #1A2232;
            color: #8892A4; line-height: 1.55;
        }
        .proto-table td:first-child { color: #F0F4F8; font-weight: 500; }
        .proto-table td:nth-child(2) { color: #4F8EF7; }
        .proto-table tr:hover td { background: rgba(79,142,247,0.02); }
        /* 90-day plan phases */
        .plan-phase {
            background: #161B27; border: 1px solid #232D3F; border-radius: 8px;
            padding: 0.9rem 1rem; margin-bottom: 0.5rem;
        }
        .plan-phase-label {
            font-size: 0.62rem; font-weight: 700; color: #4F8EF7;
            text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 0.2rem;
        }
        .plan-phase-header {
            font-size: 0.82rem; font-weight: 700; color: #F0F4F8; margin-bottom: 0.45rem;
        }
        .plan-bullet {
            font-size: 0.74rem; color: #8892A4; padding: 0.1rem 0 0.1rem 0.6rem;
            border-left: 2px solid #232D3F; margin-bottom: 0.12rem; line-height: 1.5;
        }
        /* Module next-improvements expander */
        .next-improvements-card {
            background: rgba(79,142,247,0.025); border: 1px solid rgba(79,142,247,0.1);
            border-radius: 8px; padding: 0.85rem 1rem; margin-top: 0.3rem;
        }
        .next-imp-copy {
            font-size: 0.78rem; color: #8892A4; line-height: 1.65; margin-bottom: 0.6rem;
        }
        .next-imp-bullet {
            font-size: 0.73rem; color: #6B7585;
            border-left: 2px solid rgba(79,142,247,0.18);
            padding-left: 0.6rem; margin-bottom: 0.14rem; line-height: 1.5;
        }

        /* ════════════════════════════════════════════════════════════════
           INTERNAL AI BRAIN — enterprise source-citation UI
        ════════════════════════════════════════════════════════════════ */
        .brain-source-badge {
            display: inline-flex; align-items: center; gap: 0.35rem;
            padding: 0.2rem 0.6rem;
            background: rgba(79,142,247,0.08); border: 1px solid rgba(79,142,247,0.18);
            border-radius: 4px; color: #4F8EF7;
            font-size: 0.66rem; font-weight: 700;
            letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.5rem;
        }
        .brain-helper-line {
            color: #556070; font-size: 0.76rem; margin-bottom: 0.85rem; line-height: 1.5;
        }
        .brain-trust-panel {
            background: rgba(255,255,255,0.015); border: 1px solid #232D3F;
            border-radius: 8px; padding: 0.8rem 1rem; margin: 0.85rem 0;
        }
        .trust-row {
            display: flex; align-items: baseline; gap: 0.6rem; margin-bottom: 0.3rem;
        }
        .trust-row:last-child { margin-bottom: 0; }
        .trust-lbl {
            font-size: 0.66rem; font-weight: 700; color: #556070;
            text-transform: uppercase; letter-spacing: 0.08em;
            min-width: 120px; flex-shrink: 0;
        }
        .trust-val { font-size: 0.82rem; font-weight: 600; color: #F0F4F8; }
        .trust-reason { font-size: 0.78rem; color: #8892A4; line-height: 1.5; }
        .conf-high   { color: #10B981; }
        .conf-medium { color: #F59E0B; }
        .conf-low    { color: #EF4444; }
        .review-yes  { color: #F59E0B; }
        .review-no   { color: #10B981; }
        .brain-sources-header {
            font-size: 0.66rem; font-weight: 700; color: #556070;
            text-transform: uppercase; letter-spacing: 0.1em;
            margin: 1.1rem 0 0.55rem; border-top: 1px solid #232D3F; padding-top: 0.9rem;
        }
        .brain-src-card {
            background: #161B27; border: 1px solid #232D3F;
            border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.5rem;
        }
        .brain-src-name {
            font-size: 0.86rem; font-weight: 600; color: #F0F4F8; margin-bottom: 0.3rem;
        }
        .brain-src-meta {
            font-size: 0.73rem; color: #8892A4;
            display: flex; flex-wrap: wrap; gap: 0.65rem; margin-bottom: 0.45rem;
        }
        .brain-src-meta span { white-space: nowrap; }
        .rel-pill {
            font-size: 0.68rem; font-weight: 600;
            padding: 0.12rem 0.45rem; border-radius: 4px;
        }
        .rel-high   { background: rgba(16,185,129,0.1); color: #10B981; border: 1px solid rgba(16,185,129,0.2); }
        .rel-medium { background: rgba(245,158,11,0.1); color: #F59E0B; border: 1px solid rgba(245,158,11,0.2); }
        .rel-low    { background: rgba(239,68,68,0.1);  color: #EF4444; border: 1px solid rgba(239,68,68,0.2); }
        .brain-excerpt-label {
            font-size: 0.64rem; font-weight: 700; color: #556070;
            text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem;
        }
        .brain-excerpt-box {
            background: rgba(0,0,0,0.22); border: 1px solid #1A2232;
            border-left: 2px solid rgba(79,142,247,0.45);
            border-radius: 0 6px 6px 0; padding: 0.65rem 0.9rem;
            font-size: 0.79rem; color: #8892A4; line-height: 1.65;
            font-style: italic; max-height: 110px; overflow-y: auto;
        }
        .brain-excerpt-helper {
            font-size: 0.66rem; color: #3D4D60; margin-top: 0.3rem;
        }
        .brain-followup {
            background: rgba(79,142,247,0.04); border: 1px solid rgba(79,142,247,0.1);
            border-radius: 8px; padding: 0.65rem 0.9rem; margin-top: 0.9rem;
            font-size: 0.78rem; color: #8892A4; line-height: 1.55;
        }
        .brain-fallback-box {
            background: rgba(245,158,11,0.04); border: 1px solid rgba(245,158,11,0.15);
            border-radius: 10px; padding: 1.2rem 1.4rem; margin-bottom: 0.75rem;
        }
        .brain-fallback-title {
            font-size: 0.92rem; font-weight: 600; color: #F0F4F8; margin-bottom: 0.4rem;
        }
        .brain-fallback-sub { font-size: 0.82rem; color: #8892A4; line-height: 1.6; }
        .brain-fallback-tip {
            font-size: 0.78rem; color: #556070; margin-top: 0.6rem; padding-top: 0.6rem;
            border-top: 1px solid rgba(245,158,11,0.12);
        }
        .brain-kb-status {
            background: rgba(16,185,129,0.04); border: 1px solid rgba(16,185,129,0.18);
            border-radius: 8px; padding: 0.85rem 1rem; margin-top: 0.75rem;
        }
        .kb-status-title {
            font-size: 0.74rem; font-weight: 700; color: #10B981;
            letter-spacing: 0.04em; margin-bottom: 0.45rem;
        }
        .kb-status-row { font-size: 0.76rem; color: #8892A4; margin-bottom: 0.18rem; }
        .kb-status-row strong { color: #F0F4F8; font-weight: 600; }

        /* ── User badge (top-right) ──────────────────────────────────── */
        .user-top-badge {
            position: fixed; top: 0.7rem; right: 1rem; z-index: 9999;
            background: rgba(79,142,247,0.1);
            border: 1px solid rgba(79,142,247,0.25);
            border-radius: 20px; padding: 0.25rem 0.8rem;
            color: #4F8EF7; font-size: 0.78rem; font-weight: 600;
            pointer-events: none;
        }

        /* ════════════════════════════════════════════════════════════════
           RESPONSIVE
        ════════════════════════════════════════════════════════════════ */
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


def _render_business_impact_overview():
    """Render the Business Impact Overview section on the main dashboard."""
    st.markdown(
        '<div class="why-matters">'
        '<div class="why-matters-title">Why This Matters</div>'
        '<div class="why-matters-text">'
        "Your Tax Coach's long-term AI opportunity is not just generating content or answering questions. "
        "It is turning the firm's knowledge, workflows, and client communication patterns into repeatable systems. "
        "These tools are designed to support that direction by improving knowledge access, reducing manual review, "
        "speeding up communication, and making client value more visible."
        "</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size:0.68rem;font-weight:700;color:#556070;text-transform:uppercase;"
        "letter-spacing:0.1em;margin-bottom:0.55rem'>Business Impact Overview</div>"
        "<div style='font-size:0.78rem;color:#8892A4;margin-bottom:0.75rem'>"
        "TaxCopilot is designed to help Your Tax Coach reduce repetitive internal work, improve access to knowledge, "
        "support bookkeeping review, speed up client communication, and make client value easier to communicate."
        "</div>",
        unsafe_allow_html=True,
    )

    IMPACT_CARDS = [
        ("🔍", "Faster Knowledge Access",
         "Helps team members find SOPs, strategy guidance, and internal documentation without searching folders or interrupting senior staff.",
         "Example KPI: repeated questions reduced"),
        ("📊", "Bookkeeping Review Support",
         "Flags duplicates, unusual transactions, missing categories, and potential cleanup issues before review.",
         "Example KPI: transactions reviewed per hour"),
        ("💬", "Client Communication Speed",
         "Turns internal notes into advisor-reviewed client email drafts and summaries.",
         "Example KPI: recap drafting time reduced"),
        ("✍️", "Content Production Scale",
         "Turns one tax tip or strategy into multi-platform content drafts for social, email, LinkedIn, Reels, and X.",
         "Example KPI: content pieces drafted per session"),
        ("📈", "Client Value Visibility",
         "Helps convert financial activity and tax work into simple dashboard insights clients can understand.",
         "Example KPI: client recap consistency"),
        ("⚡", "Workflow Automation Readiness",
         "Maps repeatable workflows that can later connect to QuickBooks, CRM, client portal, Zapier, Make, or n8n.",
         "Example KPI: manual handoffs reduced"),
    ]

    cards_html = "".join(
        f'<div class="impact-card">'
        f'<div class="impact-icon">{icon}</div>'
        f'<div class="impact-title">{title}</div>'
        f'<div class="impact-desc">{desc}</div>'
        f'<div class="impact-metric">{metric}</div>'
        f'</div>'
        for icon, title, desc, metric in IMPACT_CARDS
    )
    st.markdown(f'<div class="impact-grid-3">{cards_html}</div>', unsafe_allow_html=True)

    with st.expander("📋 How This Maps to Your Tax Coach's Role Priorities"):
        st.markdown(
            '<div class="vmap-wrap"><table class="vmap-table">'
            "<thead><tr>"
            "<th>Job Priority</th><th>Related Module</th><th>Business Value</th>"
            "</tr></thead>"
            "<tbody>"
            "<tr><td>Internal AI Brain</td><td>TaxCopilot — Internal AI Brain</td>"
            "<td>Faster SOP and strategy retrieval with source-based, cited answers</td></tr>"
            "<tr><td>Bookkeeping Automation</td><td>TaxCopilot — Bookkeeping Copilot</td>"
            "<td>Flags cleanup issues and suggests categories for bookkeeper review before import</td></tr>"
            "<tr><td>Client Experience</td><td>TaxCopilot — Client Communication + Dashboard</td>"
            "<td>Creates clearer client recaps and makes financial progress visible</td></tr>"
            "<tr><td>Content & Messaging</td><td>Strategy Content Studio + Marketing Engine</td>"
            "<td>Turns tax ideas into client-friendly content drafts across multiple channels</td></tr>"
            "<tr><td>Workflow Automation</td><td>TaxCopilot — Automations</td>"
            "<td>Maps repeatable workflows for future n8n / Zapier / Make / API integrations</td></tr>"
            "<tr><td>Scalability</td><td>All modules</td>"
            "<td>Reduces repeated manual work and creates reusable internal systems</td></tr>"
            "</tbody></table></div>",
            unsafe_allow_html=True,
        )


_MODULE_IMPACT = {
    "brain": {
        "copy": (
            "This module is designed to reduce time spent searching internal documentation and asking repeat questions. "
            "Newer team members can retrieve SOP-based answers faster, while senior staff spend less time "
            "answering the same operational questions."
        ),
        "measured": [
            "Number of questions answered by the AI Brain",
            "Reduction in repeated team questions over time",
            "Time to find SOP or strategy guidance",
            "Documentation gaps discovered through unanswered questions",
            "Team feedback on answer usefulness",
        ],
        "before": "Team searches folders or asks senior staff for process answers.",
        "after": "Team asks the Internal AI Brain and receives a source-based answer with the relevant document excerpt.",
    },
    "bookkeeping": {
        "copy": (
            "This module is designed to reduce manual transaction cleanup by pre-flagging duplicates, "
            "unusual amounts, missing categories, and likely categorization issues before bookkeeper review."
        ),
        "measured": [
            "Transactions reviewed per hour",
            "Number of duplicates flagged",
            "Number of missing categories detected",
            "Percentage of transactions requiring manual review",
            "Bookkeeper approval rate for suggested categories",
        ],
        "before": "Bookkeeper manually scans every row for duplicates, missing categories, and unusual transactions.",
        "after": "AI highlights likely issues first, suggests categories, and lets the bookkeeper focus on exceptions.",
    },
    "communication": {
        "copy": (
            "This module helps convert internal meeting notes and strategy notes into clear client-facing drafts, "
            "reducing writing time and improving consistency across client communication."
        ),
        "measured": [
            "Drafts generated per session",
            "Average recap drafting time",
            "Advisor edit rate before sending",
            "Client action items captured",
            "Consistency of recap structure",
        ],
        "before": "Advisor starts from a blank page after a meeting.",
        "after": "Advisor reviews an AI-generated structured draft with summary, recommendations, and action items.",
    },
    "strategy": {
        "copy": (
            "This module helps turn internal tax strategies into clear client education assets, "
            "improving the firm's ability to explain complex topics in simple language."
        ),
        "measured": [
            "Content drafts generated",
            "Number of formats created from one strategy note",
            "Review edits required before approval",
            "Approved content pieces",
            "Content turnaround time",
        ],
        "before": "Team manually rewrites each tax idea for every channel or format.",
        "after": "AI creates draft explainers, reports, newsletters, or emails from one approved strategy note.",
    },
    "automations": {
        "copy": (
            "This module maps repeatable workflows that can later be connected through n8n, Zapier, Make, "
            "APIs, QuickBooks, CRM, and client portal tools."
        ),
        "measured": [
            "Manual handoffs reduced",
            "Steps mapped for automation",
            "Workflow errors potentially reduced",
            "Time from trigger to completed task",
            "Number of processes documented",
        ],
        "before": "Team manually moves information between systems.",
        "after": "Validated workflows trigger the next step automatically after team approval.",
    },
    "dashboard": {
        "copy": (
            "This module helps make client value visible by turning transaction data into simple financial "
            "summaries, directional insights, and client-friendly dashboard views."
        ),
        "measured": [
            "Dashboards generated",
            "Client summaries created",
            "Categories flagged for review",
            "Advisor-approved insights",
            "Client-facing recap consistency",
        ],
        "before": "Financial progress and advisory value may be hidden inside spreadsheets or internal notes.",
        "after": "Client-facing dashboards show income, expenses, savings, trends, and discussion points in a clear format.",
    },
}

_MODULE_ROADMAP = {
    "brain": {
        "copy": (
            "The current version demonstrates source-based internal search. "
            "The next step is to make it production-ready with document governance, role-based access, "
            "stronger retrieval evaluation, and feedback loops so the knowledge base improves over time."
        ),
        "items": [
            "Add admin document approval before indexing",
            "Add document version control",
            "Add stronger metadata: document owner, department, upload date, review date",
            "Add golden question test set for common SOP questions",
            "Add retrieval evaluation dashboard",
            "Add 'I don't know' threshold tuning",
            "Add feedback loop for helpful / not helpful answers",
            "Add role-based access to sensitive documents",
            "Add audit logs for user queries and source documents used",
        ],
    },
    "bookkeeping": {
        "copy": (
            "The current version flags likely issues and suggests categories from CSV uploads. "
            "The next step is to connect it to the firm's chart of accounts, add approval workflows, "
            "and integrate with QuickBooks Online after the process is validated."
        ),
        "items": [
            "QuickBooks Online API integration",
            "Human approval queue before export/import",
            "Confidence threshold settings",
            "Vendor rule memory",
            "Bookkeeper correction learning",
            "Category mapping based on firm chart of accounts",
            "Duplicate detection tuning",
            "Audit log of suggested vs approved categories",
            "Exception reporting dashboard",
        ],
    },
    "communication": {
        "copy": (
            "The current version converts internal notes into structured client communication drafts. "
            "The next step is to add advisor approval, reusable templates, CRM integration, and review "
            "tracking before any client-facing message is sent."
        ),
        "items": [
            "Advisor approval workflow",
            "Saved client communication templates",
            "CRM/client portal integration",
            "Tone and format presets",
            "Compliance review checklist",
            "Before-send review status",
            "Version history for generated drafts",
            "Client-specific context fields",
            "Email draft export or Gmail/Outlook integration",
        ],
    },
    "strategy": {
        "copy": (
            "The current version converts internal strategy notes into client-facing drafts. "
            "The next step is to ground outputs in approved strategy documents and add a review process "
            "for accuracy, brand alignment, and compliance."
        ),
        "items": [
            "Connect to approved tax strategy knowledge base",
            "Add content review workflow",
            "Add brand voice examples approved by the team",
            "Add compliance checklist",
            "Add content version history",
            "Add reusable prompt templates",
            "Add export to Google Docs or CMS",
            "Add campaign tagging",
        ],
    },
    "automations": {
        "copy": (
            "The current version shows workflow blueprints and automation architecture. "
            "The next step is to implement validated workflows with logs, alerts, manual approval "
            "checkpoints, and integrations into the firm's existing tools."
        ),
        "items": [
            "Convert workflow blueprints into live n8n workflows",
            "Add Zapier/Make integration options",
            "Add trigger logs",
            "Add failed automation alerts",
            "Add manual approval checkpoints",
            "Add workflow testing sandbox",
            "Add CRM/client portal triggers",
            "Add QuickBooks event triggers",
            "Add team notification routing",
        ],
    },
    "dashboard": {
        "copy": (
            "The current version demonstrates dashboard insights from transaction data. "
            "The next step is to define the scoring model with the advisory team, add verified client "
            "metrics, and connect to real reporting systems."
        ),
        "items": [
            "Define scoring logic with advisory/bookkeeping team",
            "Add client-specific benchmarks",
            "Add tax savings tracking",
            "Add net worth trend tracking",
            "Add recurring monthly dashboard snapshots",
            "Add advisor notes",
            "Add client-facing PDF export",
            "Add data validation checks",
            "Add QuickBooks or reporting integration",
        ],
    },
}


def _render_module_impact(module_key: str):
    """Render a Business Impact expander for a specific module."""
    data = _MODULE_IMPACT.get(module_key)
    if not data:
        return

    measured_html = "".join(
        f'<div class="measured-by-item">· {item}</div>' for item in data["measured"]
    )

    with st.expander("📊 Business Impact — Potential Value for Your Tax Coach"):
        st.markdown(
            f'<div class="module-impact-copy">{data["copy"]}</div>'
            f'<div class="measured-by">'
            f'<div class="measured-by-title">Measured By (Prototype KPI Suggestions)</div>'
            f'{measured_html}'
            f'</div>'
            f'<div style="margin-top:0.75rem">'
            f'<div class="measured-by-title">Before vs After</div>'
            f'<div class="ba-wrap">'
            f'<div class="ba-row">'
            f'<div class="ba-label ba-before-lbl">Before</div>'
            f'<div class="ba-text">{data["before"]}</div>'
            f'</div>'
            f'<div class="ba-row">'
            f'<div class="ba-label ba-after-lbl">After</div>'
            f'<div class="ba-text">{data["after"]}</div>'
            f'</div></div></div>',
            unsafe_allow_html=True,
        )


def _render_module_roadmap(module_key: str):
    """Render a Next Improvements expander for a specific module."""
    data = _MODULE_ROADMAP.get(module_key)
    if not data:
        return
    bullets_html = "".join(
        f'<div class="next-imp-bullet">· {item}</div>' for item in data["items"]
    )
    with st.expander("🗺️ Next Improvements — Production Hardening Roadmap"):
        st.markdown(
            f'<div class="next-improvements-card">'
            f'<div class="next-imp-copy">{data["copy"]}</div>'
            f'{bullets_html}'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_production_roadmap():
    """Render the top-level Production Hardening Roadmap section on the dashboard."""
    st.markdown(
        '<div class="roadmap-section-title" style="margin-top:1.5rem">Production Hardening Roadmap</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="roadmap-intro-copy">'
        "TaxCopilot is a deployed prototype designed to demonstrate the architecture, workflow, and business "
        "value of an internal AI command center for a tax advisory firm. The next step would be production "
        "hardening with real team workflows, approved internal documentation, role-based access, audit logs, "
        "feedback loops, and deeper integrations."
        "</div>",
        unsafe_allow_html=True,
    )

    ROADMAP_CARDS = [
        {
            "icon": "🔐",
            "title": "Security & Access Control",
            "chip": "next",
            "desc": "Add role-based access so tax, bookkeeping, operations, and marketing users only see the tools and data relevant to their work.",
            "items": [
                "Role-based permissions",
                "Admin/user roles",
                "Client-data access controls",
                "Secure session management",
                "Stronger authentication options",
            ],
        },
        {
            "icon": "📂",
            "title": "Document Governance",
            "chip": "next",
            "desc": "Improve how SOPs, strategy guides, and internal documentation are uploaded, reviewed, versioned, and retired.",
            "items": [
                "Document approval workflow",
                "Source versioning",
                "Expiration dates for outdated SOPs",
                "Admin review queue",
                "Document owner metadata",
            ],
        },
        {
            "icon": "🧪",
            "title": "AI Evaluation & Quality Control",
            "chip": "next",
            "desc": "Create structured test sets and feedback loops to measure answer quality, retrieval accuracy, and user trust.",
            "items": [
                "Golden test questions",
                "Expected answer comparisons",
                "Retrieval quality scoring",
                "Human feedback buttons",
                "Hallucination risk tracking",
                "Prompt/version testing",
            ],
        },
        {
            "icon": "✅",
            "title": "Human Review Workflows",
            "chip": "next",
            "desc": "Add approval queues for tax-sensitive answers, client communication drafts, bookkeeping exports, and marketing content.",
            "items": [
                "Advisor approval queue",
                "Bookkeeper approval queue",
                "Marketing review queue",
                "Review status badges",
                "Approved/rejected history",
                "Comments and revision notes",
            ],
        },
        {
            "icon": "🔌",
            "title": "System Integrations",
            "chip": "production",
            "desc": "Connect TaxCopilot to the firm's existing tools after process validation.",
            "items": [
                "QuickBooks Online API",
                "CRM integration",
                "Client portal integration",
                "Google Drive or Dropbox document sync",
                "Email draft integration",
                "Slack or team chat notifications",
                "Zapier, Make, or n8n workflows",
            ],
        },
        {
            "icon": "📡",
            "title": "Monitoring & Business Metrics",
            "chip": "production",
            "desc": "Track adoption, usage, accuracy, and operational impact over time.",
            "items": [
                "Questions answered",
                "Documents searched",
                "Transactions reviewed",
                "Issues flagged",
                "Drafts generated",
                "Time saved estimates",
                "User feedback trends",
                "Error and fallback tracking",
            ],
        },
    ]

    CHIP_HTML = {
        "next":       '<span class="roadmap-chip chip-next">Next Step</span>',
        "production": '<span class="roadmap-chip chip-production">Production Upgrade</span>',
    }

    cards_html = ""
    for card in ROADMAP_CARDS:
        bullets = "".join(
            f'<div class="roadmap-bullet">· {item}</div>' for item in card["items"]
        )
        cards_html += (
            f'<div class="roadmap-card">'
            f'<div class="roadmap-card-header">'
            f'<span class="roadmap-card-icon">{card["icon"]}</span>'
            f'<span class="roadmap-card-title">{card["title"]}</span>'
            f'</div>'
            f'{CHIP_HTML[card["chip"]]}'
            f'<div class="roadmap-card-desc">{card["desc"]}</div>'
            f'{bullets}'
            f'</div>'
        )
    st.markdown(f'<div class="roadmap-grid">{cards_html}</div>', unsafe_allow_html=True)

    with st.expander("📊 From Prototype to Production — Capability Map"):
        PROTO_ROWS = [
            ("Internal document search",
             "Document governance, versioning, role-based access",
             "Keeps internal answers accurate and permission-safe"),
            ("RAG answers with sources",
             "Retrieval evaluation, feedback loops, fallback thresholds",
             "Improves trust and reduces unsupported answers"),
            ("CSV bookkeeping review",
             "QuickBooks integration, approval queue, chart of accounts mapping",
             "Fits real bookkeeping workflows safely"),
            ("Client email drafting",
             "Advisor approval, CRM/client portal integration, version history",
             "Keeps client communication accurate and reviewable"),
            ("Content generation",
             "Approved brand voice library, review workflow, analytics",
             "Keeps marketing consistent, accurate, and measurable"),
            ("Dashboard insights",
             "Real client metrics, advisory-defined scoring, recurring reports",
             "Makes client value visible using trusted data"),
            ("Workflow diagrams",
             "Live n8n/Zapier/Make automations with logs and alerts",
             "Reduces manual handoffs while preserving control"),
        ]
        rows_html = "".join(
            f"<tr><td>{cur}</td><td>{upg}</td><td>{why}</td></tr>"
            for cur, upg, why in PROTO_ROWS
        )
        st.markdown(
            '<table class="proto-table"><thead><tr>'
            "<th>Current Prototype Capability</th>"
            "<th>Production Upgrade</th>"
            "<th>Why It Matters</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table>",
            unsafe_allow_html=True,
        )

    with st.expander("📅 First 90 Days Implementation Plan"):
        PHASES = [
            {
                "label": "Days 1–15",
                "title": "Discovery & Process Mapping",
                "items": [
                    "Meet tax, bookkeeping, operations, and marketing teams",
                    "Review SOPs, strategy guides, and documentation structure",
                    "Identify repeated internal questions",
                    "Map bookkeeping cleanup process",
                    "Identify highest-impact automations",
                ],
            },
            {
                "label": "Days 16–45",
                "title": "Internal AI Brain v1",
                "items": [
                    "Load approved SOPs and strategy guides",
                    "Build searchable source-based internal assistant",
                    "Add review flags for tax-sensitive questions",
                    "Test with a small team",
                    "Track unanswered questions and documentation gaps",
                ],
            },
            {
                "label": "Days 46–70",
                "title": "Workflow Tools",
                "items": [
                    "Improve bookkeeping review assistant",
                    "Add client recap generator",
                    "Add draft approval process",
                    "Build first automation workflow",
                    "Begin measuring time saved and review rate",
                ],
            },
            {
                "label": "Days 71–90",
                "title": "Adoption & Iteration",
                "items": [
                    "Train team on tools and gather feedback",
                    "Improve retrieval and prompts based on usage",
                    "Add integrations based on team workflow",
                    "Prepare roadmap for next AI systems",
                ],
            },
        ]
        phases_html = ""
        for phase in PHASES:
            bullets = "".join(
                f'<div class="plan-bullet">· {item}</div>' for item in phase["items"]
            )
            phases_html += (
                f'<div class="plan-phase">'
                f'<div class="plan-phase-label">{phase["label"]}</div>'
                f'<div class="plan-phase-header">{phase["title"]}</div>'
                f'{bullets}'
                f'</div>'
            )
        st.markdown(phases_html, unsafe_allow_html=True)


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
                <div class="hero-badge">Internal AI Command Center — Human-in-the-Loop Workflow</div>
                <h1 class="hero-title">{APP_TITLE}</h1>
                <p class="hero-subtitle">Internal AI assistant that supports the team — retrieves knowledge, drafts communications, flags issues, and surfaces insights for human review.</p>
                <span class="hero-chip">Source-Based Retrieval</span>
                <span class="hero-chip">Bookkeeping Review</span>
                <span class="hero-chip">Draft Communications</span>
                <span class="hero-chip">Advisor-in-the-Loop</span>
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

    # Business impact overview + value map
    _render_business_impact_overview()
    _render_production_roadmap()

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
    """Render a compact sidebar with user profile at top and sign-out at bottom."""
    with st.sidebar:
        # ── User profile (top) ────────────────────────────────────────────
        user_name  = st.session_state.get("user_name", "")
        user_email = st.session_state.get("user_email", "")
        if user_name:
            st.markdown(
                f"""<div style="
                    background:rgba(79,142,247,0.08);
                    border:1px solid rgba(79,142,247,0.18);
                    border-radius:10px;padding:0.75rem 0.9rem;margin-bottom:0.75rem;">
                    <div style="font-weight:700;color:#ffffff;font-size:0.9rem;">👤 {user_name}</div>
                    <div style="color:#64748b;font-size:0.75rem;margin-top:0.15rem;">{user_email}</div>
                </div>""",
                unsafe_allow_html=True,
            )

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

        # ── Prototype Usage Metrics ───────────────────────────────────────
        questions    = st.session_state.get("kpi_questions_asked", 0)
        docs_indexed = len(st.session_state.get("workspace_file_names", []))
        chunks       = len(st.session_state.get("workspace_chunks", []))
        txn          = st.session_state.get("kpi_transactions_analyzed", 0)
        issues       = st.session_state.get("kpi_issues_flagged", 0)
        drafts       = st.session_state.get("kpi_drafts_generated", 0)
        content      = st.session_state.get("kpi_content_pieces", 0)

        kpi_rows = [
            ("Questions asked", questions),
            ("Documents indexed", docs_indexed),
            ("Chunks created", chunks),
            ("Transactions analyzed", txn),
            ("Issues flagged", issues),
            ("Drafts generated", drafts),
            ("Content pieces", content),
        ]
        rows_html = "".join(
            f'<div class="kpi-row"><span class="kpi-lbl">{lbl}</span>'
            f'<span class="kpi-val">{val}</span></div>'
            for lbl, val in kpi_rows
        )
        st.markdown(
            f'<div style="margin-top:0.5rem">'
            f'<div class="kpi-section-title">Prototype Usage Metrics</div>'
            f'{rows_html}'
            f'<div style="font-size:0.62rem;color:#3D4D60;margin-top:0.35rem">'
            f'Session-level only. Not company-wide production metrics.'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Status")
        st.caption(f"Ollama model: `{DEFAULT_OLLAMA_MODEL}`")
        st.toggle("Show debug panels", key="show_debug_panels")
        if st.session_state.get("retrieval_mode"):
            st.caption(f"Retrieval mode: {st.session_state['retrieval_mode']}")

        # ── Sign Out (bottom of sidebar) ──────────────────────────────────
        if st.session_state.get("user_name"):
            st.markdown("---")
            if st.button("🚪  Sign Out", use_container_width=True, key="sidebar_signout"):
                st.session_state.update({
                    "authenticated": False,
                    "user_name":     "",
                    "user_email":    "",
                    "login_error":   False,
                    "login_view":    "signin",
                })
                st.rerun()


def _brain_review_required(result: dict) -> tuple:
    """Return (review_required: bool, reason: str) using confidence, style, and question content."""
    confidence    = result.get("confidence", 0.0)
    question_style = result.get("question_style", "")
    answer        = result.get("answer", "")
    retrieval_mode = result.get("retrieval_mode", "")
    question      = st.session_state.get("last_brain_question", "").lower()

    TAX_SENSITIVE = {
        "tax", "taxes", "irs", "deduction", "deductions", "filing", "audit",
        "accounting", "bookkeeping", "client", "strategy", "advice", "penalty",
        "compliance", "entity", "s-corp", "llc", "profit", "loss", "income",
        "expense", "quarterly", "estimated", "write-off", "depreciation",
        "return", "1099", "w-2", "schedule", "basis", "capital gains",
    }
    is_tax_sensitive = any(kw in question for kw in TAX_SENSITIVE)

    if answer in NO_PROOF_ANSWERS:
        return True, "No supporting source found in uploaded documents."
    if confidence < 0.40:
        return True, "Source support is weak — low confidence retrieval. Consult source documents directly."
    if retrieval_mode == "Keyword fallback":
        return True, "Keyword-only retrieval mode active (semantic embeddings unavailable). Verify manually."
    if question_style in {"assessment", "complex"}:
        return True, "Answer involves professional judgment — a qualified team member should review."
    if question_style == "workflow" and is_tax_sensitive:
        return True, "Workflow involves tax-sensitive steps — review the source SOP directly."
    if is_tax_sensitive and confidence < 0.75:
        return True, "Tax-sensitive topic with moderate source confidence — advisor review recommended."
    return False, "Source support is strong and the question is factual in nature."


def _brain_render_source_card(source: dict, index: int):
    """Render a structured source document card with metadata and excerpt."""
    raw_score   = source.get("score", 0.0)
    relevance   = min(int(raw_score * 100), 99)
    doc_name    = source.get("document_name", "Unknown")
    doc_type    = source.get("document_type", "General Document")
    section     = source.get("section_title") or source.get("section_label", "—")
    page_num    = source.get("page_number")
    chunk_id    = source.get("chunk_id")
    excerpt     = source.get("chunk_text", "")

    if relevance >= 70:
        rel_class = "rel-high"
    elif relevance >= 45:
        rel_class = "rel-medium"
    else:
        rel_class = "rel-low"

    location_str = f"Page {page_num}" if page_num else f"Chunk {chunk_id}"

    meta_html = (
        f"<span>📄 {doc_type}</span>"
        f"<span>Section: {section}</span>"
        f"<span>{location_str}</span>"
    )

    st.markdown(
        f"""<div class="brain-src-card">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.3rem;">
                <div class="brain-src-name">{index}. {doc_name}</div>
                <span class="rel-pill {rel_class}">Relevance {relevance}%</span>
            </div>
            <div class="brain-src-meta">{meta_html}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    if excerpt:
        with st.expander("View relevant excerpt", expanded=(index == 1)):
            st.markdown(
                '<div class="brain-excerpt-label">Relevant Excerpt</div>'
                f'<div class="brain-excerpt-box">"{excerpt[:600]}{"…" if len(excerpt) > 600 else ""}"</div>'
                '<div class="brain-excerpt-helper">This excerpt is the retrieved context used to generate the answer.</div>',
                unsafe_allow_html=True,
            )


def _render_citation_blocks(result):
    """Legacy citation renderer — kept for compound results. Main path uses _brain_render_source_card."""
    citations = result.get("citations") or []
    if not citations and result.get("citation"):
        citations = [result["citation"]]
    if not citations or result.get("answer") in NO_PROOF_ANSWERS:
        return
    for citation in citations:
        with st.container(border=True):
            st.markdown("**Source Document**")
            st.write(
                f"{citation['document_name']} — {citation['section_label']} — Chunk {citation['chunk_id']}"
            )
            st.caption(f"Document type: {citation['document_type']}")
            if citation.get("proof_snippet"):
                st.markdown("**Relevant Excerpt**")
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
    st.session_state["kpi_questions_asked"] = st.session_state.get("kpi_questions_asked", 0) + 1
    _record_recent_query(question, result.get("answer") or "")


def _render_brain_result():
    """Enterprise source-citation answer UI for the Internal AI Brain."""
    result = st.session_state.get("last_brain_result")

    # ── Empty state: no query run yet ────────────────────────────────────────
    if not result:
        with st.container(border=True):
            st.markdown(
                "<div style='text-align:center;padding:2rem 1rem'>"
                "<div style='font-size:1.6rem;margin-bottom:0.6rem'>🧠</div>"
                "<div style='font-size:0.92rem;font-weight:600;color:#F0F4F8;margin-bottom:0.35rem'>"
                "Ask a question to search your knowledge base</div>"
                "<div style='font-size:0.8rem;color:#8892A4;max-width:340px;margin:0 auto;line-height:1.6'>"
                "The assistant will retrieve relevant source material from your uploaded documents "
                "and provide a grounded, cited answer.</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        return

    ts = _format_timestamp(st.session_state.get("last_brain_timestamp"))
    confidence   = result.get("confidence", 0.0)
    sources      = result.get("sources", [])
    answer       = result.get("answer", "")
    question_style = result.get("question_style", "")

    review_req, review_reason = _brain_review_required(result)

    # ── Compound question path ────────────────────────────────────────────────
    if result.get("compound_results"):
        st.caption(f"Source-based answer · Generated {ts}")
        for claim_result in result["compound_results"]:
            cr_conf = claim_result.get("confidence", 0.0)
            cr_review, cr_reason = _brain_review_required(claim_result)
            with st.container(border=True):
                st.markdown(
                    '<div class="brain-source-badge">🔍 Source-Based Answer</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**{claim_result['label']}**")
                st.markdown(claim_result["answer"])

                if cr_conf >= 0.70:
                    conf_cls, conf_lbl = "conf-high", "High"
                elif cr_conf >= 0.40:
                    conf_cls, conf_lbl = "conf-medium", "Medium"
                else:
                    conf_cls, conf_lbl = "conf-low", "Low"

                rev_cls = "review-yes" if cr_review else "review-no"
                rev_lbl = "Yes" if cr_review else "No"

                st.markdown(
                    f"""<div class="brain-trust-panel">
                        <div class="trust-row">
                            <span class="trust-lbl">Confidence</span>
                            <span class="trust-val {conf_cls}">{conf_lbl} · {cr_conf:.0%}</span>
                        </div>
                        <div class="trust-row">
                            <span class="trust-lbl">Review Required</span>
                            <span class="trust-val {rev_cls}">{rev_lbl}</span>
                        </div>
                        <div class="trust-row">
                            <span class="trust-lbl">Reason</span>
                            <span class="trust-reason">{cr_reason}</span>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

                if claim_result.get("sources"):
                    st.markdown('<div class="brain-sources-header">Sources Used</div>', unsafe_allow_html=True)
                    for idx, src in enumerate(claim_result["sources"][:3], start=1):
                        _brain_render_source_card(src, idx)

        _render_document_debug_panels(result)
        return

    # ── Single-question path ──────────────────────────────────────────────────
    with st.container(border=True):
        # Header badge + timestamp
        hdr_col, ts_col = st.columns([3, 1])
        with hdr_col:
            st.markdown(
                '<div class="brain-source-badge">🔍 Source-Based Answer</div>'
                '<div class="brain-helper-line">'
                "Generated from retrieved internal documents. "
                "Review required for tax-sensitive decisions."
                "</div>",
                unsafe_allow_html=True,
            )
        with ts_col:
            st.caption(ts)

        # ── Fallback: no source found ─────────────────────────────────────────
        if answer in NO_PROOF_ANSWERS:
            st.markdown(
                '<div class="brain-fallback-box">'
                '<div class="brain-fallback-title">⚠️ Insufficient Source Coverage</div>'
                '<div class="brain-fallback-sub">'
                "I could not find enough support in the uploaded documents to answer confidently. "
                "Please review the source materials directly or consult a qualified team member."
                "</div>"
                '<div class="brain-fallback-tip">'
                "💡 Suggested next step: Upload a relevant SOP, strategy guide, or internal note "
                "to improve knowledge base coverage for this topic."
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            if sources:
                st.markdown('<div class="brain-sources-header">Top Attempted Matches</div>', unsafe_allow_html=True)
                for idx, src in enumerate(sources[:2], start=1):
                    _brain_render_source_card(src, idx)
            _render_document_debug_panels(result)
            return

        # ── Answer text ───────────────────────────────────────────────────────
        st.markdown(answer)

        # ── Confidence / review trust panel ──────────────────────────────────
        if confidence >= 0.70:
            conf_cls, conf_lbl = "conf-high", "High"
        elif confidence >= 0.40:
            conf_cls, conf_lbl = "conf-medium", "Medium"
        elif confidence > 0:
            conf_cls, conf_lbl = "conf-low", "Low"
        else:
            conf_cls, conf_lbl = "conf-low", "—"

        rev_cls = "review-yes" if review_req else "review-no"
        rev_lbl = "Yes" if review_req else "No"

        st.markdown(
            f"""<div class="brain-trust-panel">
                <div class="trust-row">
                    <span class="trust-lbl">Confidence</span>
                    <span class="trust-val {conf_cls}">{conf_lbl} · {confidence:.0%}</span>
                </div>
                <div class="trust-row">
                    <span class="trust-lbl">Review Required</span>
                    <span class="trust-val {rev_cls}">{rev_lbl}</span>
                </div>
                <div class="trust-row">
                    <span class="trust-lbl">Reason</span>
                    <span class="trust-reason">{review_reason}</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # ── Feedback buttons ──────────────────────────────────────────────────
        fb1, fb2, _ = st.columns([1, 1.3, 4])
        ts_key = st.session_state.get("last_brain_timestamp", "")
        with fb1:
            if st.button("👍 Helpful", key=f"feedback_up_{ts_key}"):
                st.session_state.setdefault("feedback_log", []).append({
                    "question": st.session_state.get("last_brain_question", ""),
                    "feedback": "positive",
                    "timestamp": ts_key,
                })
                st.toast("Thanks — this feedback helps improve the knowledge base.")
        with fb2:
            if st.button("👎 Needs Improvement", key=f"feedback_down_{ts_key}"):
                st.session_state.setdefault("feedback_log", []).append({
                    "question": st.session_state.get("last_brain_question", ""),
                    "feedback": "negative",
                    "timestamp": ts_key,
                })
                st.toast("Thanks — noted. Consider uploading more specific source documents.")

        # ── Sources used ──────────────────────────────────────────────────────
        top_sources = sources[:3]
        if top_sources:
            st.markdown('<div class="brain-sources-header">Sources Used</div>', unsafe_allow_html=True)
            for idx, src in enumerate(top_sources, start=1):
                _brain_render_source_card(src, idx)

        # ── Follow-up guidance ────────────────────────────────────────────────
        st.markdown(
            '<div class="brain-followup">'
            "💬 <strong>Need more detail?</strong> Try a follow-up like: "
            "<em>'Show me the step-by-step SOP'</em> or "
            "<em>'What documents are required for this process?'</em>"
            "</div>",
            unsafe_allow_html=True,
        )

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
        "<p class='section-copy'>Retrieve source-grounded answers from uploaded internal documents. "
        "Suggested answers are grounded in retrieved source material to reduce hallucination risk — "
        "review for tax-sensitive decisions before relying on them.</p>",
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1, 1.25], gap="large")

    with left_col:
        with st.container(border=True):
            _render_step_header("Knowledge Base", "Upload SOPs, guides, client notes, and internal documents")

            if not st.session_state.get("workspace_documents"):
                st.markdown(
                    "<div style='background:rgba(79,142,247,0.04);border:1px solid rgba(79,142,247,0.12);"
                    "border-radius:8px;padding:0.9rem 1rem;margin-bottom:0.75rem'>"
                    "<div style='font-size:0.82rem;font-weight:600;color:#F0F4F8;margin-bottom:0.25rem'>"
                    "Upload internal documents to activate the AI Brain</div>"
                    "<div style='font-size:0.76rem;color:#8892A4;line-height:1.6'>"
                    "Add SOPs, strategy guides, client communication templates, or internal notes. "
                    "The assistant will use these documents to answer team questions with source references."
                    "</div></div>",
                    unsafe_allow_html=True,
                )

            if st.button("Load Sample Workspace", type="primary", use_container_width=True):
                with st.spinner("Loading the sample workspace..."):
                    _load_demo_workspace()
                st.rerun()

            uploaded_files = st.file_uploader(
                "Upload PDF, TXT, or CSV files",
                type=SUPPORTED_UPLOAD_TYPES,
                accept_multiple_files=True,
                key="brain_workspace_uploader",
                label_visibility="collapsed",
                help="Upload text-based PDFs, TXT notes, or CSV exports. CSV files are converted into searchable transaction summaries for retrieval.",
            )
            _render_selected_files_preview(uploaded_files)

            if st.button("Index Uploaded Files", use_container_width=True):
                try:
                    with st.spinner("Building knowledge base — chunking and embedding documents..."):
                        _index_uploaded_workspace(uploaded_files)
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not build the workspace: {error}")

            # Knowledge base status panel
            if st.session_state.get("workspace_documents"):
                doc_count   = len(st.session_state.get("workspace_file_names", []))
                chunk_count = len(st.session_state.get("workspace_chunks", []))
                ret_mode    = st.session_state.get("retrieval_mode", "—")
                vs_label    = "FAISS (semantic)" if "Semantic" in ret_mode else "Keyword index"
                loaded_at   = _format_timestamp(st.session_state.get("workspace_loaded_at"))
                st.markdown(
                    f'<div class="brain-kb-status">'
                    f'<div class="kb-status-title">✓ Knowledge Base Ready</div>'
                    f'<div class="kb-status-row">Documents indexed: <strong>{doc_count}</strong></div>'
                    f'<div class="kb-status-row">Chunks created: <strong>{chunk_count}</strong></div>'
                    f'<div class="kb-status-row">Embedding model: <strong>all-MiniLM-L6-v2</strong></div>'
                    f'<div class="kb-status-row">Vector store: <strong>{vs_label}</strong></div>'
                    f'<div class="kb-status-row">Status: <strong>Ready for source-based search</strong></div>'
                    f'<div class="kb-status-row">Indexed: <strong>{loaded_at}</strong></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            elif st.session_state.get("retrieval_mode") == "Keyword fallback":
                st.caption("🔍 Using keyword-based retrieval — semantic embeddings unavailable")

        if st.session_state.get("workspace_documents"):
            _render_loaded_document_cards()
        _render_workspace_inventory()

    with right_col:
        with st.container(border=True):
            _render_step_header("Ask", "Search your internal documents with a natural language question")

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

    _render_module_impact("brain")
    _render_module_roadmap("brain")


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
        "<p class='section-copy'>Flag transactions that need attention and suggest categories for bookkeeper review. "
        "A bookkeeper should approve all outputs before import into QuickBooks or any accounting system.</p>",
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
                label_visibility="collapsed",
            )

            demo_a_col, demo_b_col = st.columns(2)
            if demo_a_col.button("Use Sample Transactions: ABC", use_container_width=True):
                st.session_state["bookkeeping_demo_choice"] = "transactions_abc.csv"
                st.rerun()
            if demo_b_col.button("Use Sample Transactions: XYZ", use_container_width=True):
                st.session_state["bookkeeping_demo_choice"] = "transactions_xyz.csv"
                st.rerun()

            st.caption("⚠️ Suggested categories are assistive only. A bookkeeper should review and approve before import into any accounting system.")

        raw_df, source_name = _load_bookkeeping_dataframe(uploaded_csv)
        if raw_df is None:
            _render_empty_state(
                title="Upload a bookkeeping CSV to begin",
                description="CSV only",
            )
            return

        with st.spinner("Reviewing the transaction file and preparing suggestions..."):
            cleaned_df, report = process_dataframe(raw_df)
        # Track prototype usage KPIs (session-level)
        st.session_state["kpi_transactions_analyzed"] = report.get("total_rows", 0)
        st.session_state["kpi_issues_flagged"] = (
            report.get("duplicate_count", 0)
            + report.get("missing_category_count", 0)
            + report.get("anomaly_count", 0)
        )

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
            st.markdown("#### Vendor Normalization Suggestions — Review Before Applying")
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

    _render_module_impact("bookkeeping")
    _render_module_roadmap("bookkeeping")


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
        "<p class='section-copy'>Generate draft client-facing outputs from internal documents. "
        "All drafts should be reviewed by an advisor for accuracy, tone, and client-specific context before sending.</p>",
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
                        st.session_state["kpi_drafts_generated"] = st.session_state.get("kpi_drafts_generated", 0) + 1
                except RuntimeError as error:
                    st.error(str(error))

        report_output = st.session_state.get("client_report_output")
        if report_output:
            st.markdown("#### Draft Output — Review Before Sending")
            st.caption("ℹ️ This is a draft generated from internal documents. An advisor should review for accuracy, tone, and client-specific context before sending.")
            _REPORT_SECTIONS = [
                ("Draft Summary", "summary", None, None),
                ("Key Issues", "key_issues", None, None),
                ("Recommendations", "recommendations", None, None),
                ("Suggested Action Items", "action_items", None, None),
                ("Draft Client Email — Advisor Review Required", "client_email", 240, "client_report_email_output"),
            ]
            for label, key, height, card_key in _REPORT_SECTIONS:
                value = report_output.get(key, "")
                if value:
                    _render_output_card(label, value, height=height, key=card_key)
        elif not st.session_state.get("client_explanation_output"):
            with st.container(border=True):
                _render_step_header("Output", "Generated draft output appears here")

    _render_module_impact("communication")
    _render_module_roadmap("communication")


def _render_strategy_content_studio_tab():
    """Render the strategy-to-content workflow."""
    st.subheader("Strategy Content Studio")
    st.markdown(
        "<p class='section-copy'>Create draft client-facing content from internal strategy notes. "
        "Review all drafts for accuracy and firm alignment before publishing.</p>",
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
                    st.session_state["kpi_content_pieces"] = st.session_state.get("kpi_content_pieces", 0) + 1

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

    _render_module_impact("strategy")
    _render_module_roadmap("strategy")


def _render_automations_tab():
    """Render the Automations tab — pre-built workflow triggers and integration roadmap."""
    st.subheader("Automations")
    st.markdown(
        "<p class='section-copy'>Workflow blueprints showing how internal systems can be connected. "
        "These are implementation-ready designs — each automation should be validated by the team before deployment.</p>",
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
                "Suggest categories for bookkeeper review",
                "Flag duplicates and missing data",
                "Generate review summary for bookkeeper approval",
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
                "Run category suggestions for bookkeeper review",
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

    _render_module_impact("automations")
    _render_module_roadmap("automations")


_LOGIN_CSS = """
<style>
/* ── Hide all Streamlit chrome on login page ───────────────────────────── */
#MainMenu, footer, header                 { visibility: hidden !important; }
[data-testid="stToolbar"]                 { display: none !important; }
[data-testid="stDecoration"]              { display: none !important; }
[data-testid="stStatusWidget"]            { display: none !important; }
[data-testid="collapsedControl"]          { display: none !important; }
section[data-testid="stSidebar"]          { display: none !important; }

/* ── Full-screen dark background with subtle gradient ──────────────────── */
.stApp {
    background: #0A0F1E !important;
    background-image:
        radial-gradient(ellipse at 20% 55%, rgba(79,142,247,0.09) 0%, transparent 50%),
        radial-gradient(ellipse at 78% 18%, rgba(123,92,247,0.09) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 95%, rgba(79,142,247,0.05) 0%, transparent 45%);
    min-height: 100vh;
}
.block-container {
    padding-top: 3.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* ── Login card — styled bordered container ────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #111827 !important;
    border: 1px solid #2D3748 !important;
    border-radius: 16px !important;
    padding: 2.5rem 2.25rem !important;
    box-shadow:
        0 0 0 1px rgba(79,142,247,0.06),
        0 25px 60px rgba(0,0,0,0.55),
        0 0 90px rgba(79,142,247,0.07) !important;
}

/* ── Logo ──────────────────────────────────────────────────────────────── */
.ytc-logo { text-align: center; margin-bottom: 2rem; }
.ytc-logo-text {
    font-size: 3rem; font-weight: 900; color: #4F8EF7;
    letter-spacing: -0.03em; line-height: 1;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    text-shadow: 0 0 48px rgba(79,142,247,0.45);
}
.ytc-company {
    font-size: 1.05rem; color: #ffffff; font-weight: 700;
    margin-top: 0.45rem;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.ytc-subtitle {
    font-size: 0.7rem; color: #8892A4; font-weight: 500;
    margin-top: 0.2rem; letter-spacing: 0.13em; text-transform: uppercase;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ── Divider ──────────────────────────────────────────────────────────── */
.login-divider {
    display: flex; align-items: center; gap: 0.75rem;
    margin: 1rem 0;
}
.login-divider::before, .login-divider::after {
    content: ''; flex: 1; height: 1px; background: rgba(255,255,255,0.08);
}
.login-divider span {
    color: #4a5568; font-size: 0.73rem; font-weight: 500;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ── Error box ─────────────────────────────────────────────────────────── */
.login-error {
    background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.28);
    border-radius: 8px; padding: 0.65rem 0.9rem; color: #f87171;
    font-size: 0.84rem; margin: 0.6rem 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.signup-error {
    background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.28);
    border-radius: 8px; padding: 0.65rem 0.9rem; color: #f87171;
    font-size: 0.84rem; margin: 0.6rem 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.signup-success {
    background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.28);
    border-radius: 8px; padding: 0.65rem 0.9rem; color: #34d399;
    font-size: 0.84rem; margin: 0.6rem 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ── Forgot password / bottom links ───────────────────────────────────── */
.login-forgot {
    text-align: right; margin-top: 0.15rem; margin-bottom: 0.7rem;
}
.login-forgot a {
    color: #4a5568; font-size: 0.76rem; text-decoration: none;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    transition: color 0.15s;
}
.login-forgot a:hover { color: #4F8EF7; }
.login-bottom {
    text-align: center; color: #4a5568; font-size: 0.82rem;
    margin-top: 1.1rem;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.login-bottom-link { color: #4F8EF7; font-weight: 600; cursor: pointer; }

/* ── Password strength bar ─────────────────────────────────────────────── */
.pw-strength-wrap { margin-top: -0.3rem; margin-bottom: 0.6rem; }
.pw-strength-bar {
    height: 3px; border-radius: 3px; transition: width 0.3s ease;
}
.pw-strength-label {
    font-size: 0.72rem; margin-top: 0.15rem;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ── ALL text inputs ───────────────────────────────────────────────────── */
div[data-testid="stTextInput"] input {
    background: #1E2533 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-size: 14.5px !important;
    padding: 0.7rem 0.9rem !important;
}
div[data-testid="stTextInput"] input::placeholder { color: rgba(255,255,255,0.22) !important; }
div[data-testid="stTextInput"] input:focus {
    border-color: #4F8EF7 !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,0.14) !important;
    outline: none !important;
}
div[data-testid="stTextInput"] > div { border: none !important; background: transparent !important; }
div[data-testid="stTextInput"] label { display: none !important; }

/* ── Checkbox (show/hide password) ────────────────────────────────────── */
div[data-testid="stCheckbox"] { margin-top: -0.5rem; margin-bottom: 0.4rem; }
div[data-testid="stCheckbox"] label p { color: #4a5568 !important; font-size: 0.77rem !important; }

/* ── Primary button: Sign In / Create Account ──────────────────────────── */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4F8EF7 0%, #7B5CF7 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    height: 48px !important;
    letter-spacing: 0.02em;
    box-shadow: 0 4px 20px rgba(79,142,247,0.32) !important;
}
div.stButton > button[kind="primary"]:hover {
    opacity: 0.91 !important;
    box-shadow: 0 6px 28px rgba(79,142,247,0.48) !important;
}

/* ── Secondary buttons (link-style: Sign up / Sign in links) ───────────── */
div.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: none !important;
    color: #4F8EF7 !important;
    font-size: 0.83rem !important;
    font-weight: 600 !important;
    padding: 0 !important;
    height: auto !important;
    min-height: 0 !important;
}

/* ── Google button — white background, grey border ─────────────────────── */
:has(> [data-testid="stButton"] > button[key="google_btn"]) div.stButton > button,
div[data-testid="column"]:first-child div.stButton > button:not([kind="primary"]):not([kind="secondary"]) {
    background: #ffffff !important;
    border: 1px solid #dadce0 !important;
    border-radius: 10px !important;
    color: #3c4043 !important;
    font-weight: 500 !important;
    font-size: 13.5px !important;
    height: 46px !important;
}
/* ── Apple button — black background, white text ───────────────────────── */
div[data-testid="column"]:last-child div.stButton > button:not([kind="primary"]):not([kind="secondary"]) {
    background: #050505 !important;
    border: 1px solid #111 !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    font-size: 13.5px !important;
    height: 46px !important;
}

/* ── User badge (top-right, shown when logged in) ─────────────────────── */
.user-top-badge {
    position: fixed; top: 0.7rem; right: 1rem; z-index: 9999;
    background: rgba(79,142,247,0.1);
    border: 1px solid rgba(79,142,247,0.22);
    border-radius: 20px; padding: 0.25rem 0.75rem;
    color: #4F8EF7; font-size: 0.78rem; font-weight: 600;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    pointer-events: none;
}
</style>
"""


def _pw_strength(password: str) -> tuple[int, str, str]:
    """Return (score 0-4, label, bar-color) for a password."""
    if not password:
        return 0, "", "#374151"
    score = 0
    if len(password) >= 8:
        score += 1
    if any(c.isupper() for c in password):
        score += 1
    if any(c.isdigit() for c in password):
        score += 1
    if any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?" for c in password):
        score += 1
    labels = ["", "Weak", "Fair", "Good", "Strong"]
    colors = ["#374151", "#ef4444", "#f59e0b", "#3b82f6", "#10b981"]
    return score, labels[score], colors[score]


def _render_login_page():
    """Full-screen corporate login page — appears before the main app loads."""
    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)

    DEMO_EMAIL    = "demo@yourtaxcoach.com"
    DEMO_PASSWORD = "TaxCoach2025"

    view = st.session_state.get("login_view", "signin")

    # Three columns: narrow | card | narrow
    _, center, _ = st.columns([1, 1.35, 1])

    with center:
        # ── Logo ─────────────────────────────────────────────────────────
        st.markdown("""
        <div class="ytc-logo">
            <div class="ytc-logo-text">YTC</div>
            <div class="ytc-company">Your Tax Coach</div>
            <div class="ytc-subtitle">AI Command Center</div>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):

            # ══════════════════════════════════════════════════════════════
            if view == "signin":

                # Social login buttons
                g_col, a_col = st.columns(2, gap="small")
                with g_col:
                    if st.button("G   Continue with Google",
                                 use_container_width=True, key="google_btn"):
                        st.toast(
                            "Coming Soon — Google login launching next week!",
                            icon="ℹ️",
                        )
                with a_col:
                    if st.button("🍎  Continue with Apple",
                                 use_container_width=True, key="apple_btn"):
                        st.toast(
                            "Coming Soon — Google login launching next week!",
                            icon="ℹ️",
                        )

                st.markdown(
                    '<div class="login-divider"><span>or</span></div>',
                    unsafe_allow_html=True,
                )

                email = st.text_input(
                    "e", placeholder="Work email address",
                    key="login_email", label_visibility="collapsed",
                )

                show_pw = st.checkbox("👁  Show password", key="login_show_pw")
                password = st.text_input(
                    "p", placeholder="Password",
                    type="default" if show_pw else "password",
                    key=f"login_pw_{show_pw}",
                    label_visibility="collapsed",
                )

                st.markdown(
                    '<div class="login-forgot"><a href="#">Forgot password?</a></div>',
                    unsafe_allow_html=True,
                )

                if st.session_state.get("login_error"):
                    st.markdown(
                        '<div class="login-error">'
                        "❌ Invalid email or password. Please try again."
                        "</div>",
                        unsafe_allow_html=True,
                    )

                if st.button("Sign In", type="primary",
                             use_container_width=True, key="signin_btn"):
                    # Demo account — always works
                    if email.strip() == DEMO_EMAIL and password == DEMO_PASSWORD:
                        st.session_state.update({
                            "authenticated": True,
                            "user_name":     "Demo User",
                            "user_email":    email.strip(),
                            "login_error":   False,
                        })
                        st.rerun()
                    else:
                        # Try Supabase
                        try:
                            from supabase_config import login_user, is_configured
                            if is_configured():
                                result = login_user(email.strip(), password)
                                if result["success"]:
                                    u = result["user"]
                                    st.session_state.update({
                                        "authenticated": True,
                                        "user_name":     u["full_name"],
                                        "user_email":    u["email"],
                                        "login_error":   False,
                                    })
                                    st.rerun()
                                else:
                                    st.session_state["login_error"] = True
                                    st.rerun()
                            else:
                                st.session_state["login_error"] = True
                                st.rerun()
                        except Exception:
                            st.session_state["login_error"] = True
                            st.rerun()

                st.markdown(
                    '<div class="login-bottom">'
                    "Don't have an account?&nbsp;"
                    '<span class="login-bottom-link">Sign Up</span>'
                    "</div>",
                    unsafe_allow_html=True,
                )
                if st.button("Sign Up →", key="go_signup"):
                    st.session_state["login_view"]  = "signup"
                    st.session_state["login_error"] = False
                    st.session_state.pop("signup_error", None)
                    st.rerun()

            # ══════════════════════════════════════════════════════════════
            else:  # signup view

                full_name = st.text_input(
                    "fn", placeholder="Full Name",
                    key="signup_name", label_visibility="collapsed",
                )
                email_su = st.text_input(
                    "em", placeholder="Work email address",
                    key="signup_email", label_visibility="collapsed",
                )
                password_su = st.text_input(
                    "pw", placeholder="Password (min 8 characters)",
                    type="password", key="signup_pw",
                    label_visibility="collapsed",
                )

                # Password strength indicator
                if password_su:
                    score, label, color = _pw_strength(password_su)
                    pct = score * 25
                    st.markdown(
                        f"""<div class="pw-strength-wrap">
                            <div style="background:#1e2533;border-radius:3px;height:3px;">
                                <div class="pw-strength-bar"
                                     style="width:{pct}%;background:{color};"></div>
                            </div>
                            <div class="pw-strength-label" style="color:{color};">
                                {label}
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                company = st.text_input(
                    "co", placeholder="Company Name",
                    key="signup_company", label_visibility="collapsed",
                )

                if st.session_state.get("signup_error"):
                    st.markdown(
                        f'<div class="signup-error">❌ {st.session_state["signup_error"]}</div>',
                        unsafe_allow_html=True,
                    )

                if st.button("Create Account", type="primary",
                             use_container_width=True, key="create_btn"):
                    if not all([full_name, email_su, password_su, company]):
                        st.session_state["signup_error"] = "Please fill in all fields."
                        st.rerun()
                    elif len(password_su) < 8:
                        st.session_state["signup_error"] = "Password must be at least 8 characters."
                        st.rerun()
                    else:
                        try:
                            from supabase_config import register_user, is_configured
                            if is_configured():
                                result = register_user(
                                    full_name.strip(), email_su.strip(),
                                    password_su, company.strip(),
                                )
                                if result["success"]:
                                    st.session_state.update({
                                        "authenticated": True,
                                        "user_name":     full_name.strip(),
                                        "user_email":    email_su.strip(),
                                        "login_error":   False,
                                    })
                                    st.session_state.pop("signup_error", None)
                                    st.rerun()
                                else:
                                    st.session_state["signup_error"] = result["error"]
                                    st.rerun()
                            else:
                                st.session_state["signup_error"] = (
                                    "Supabase is not configured. "
                                    "Add SUPABASE_URL and SUPABASE_ANON_KEY to your .env file."
                                )
                                st.rerun()
                        except Exception as exc:
                            st.session_state["signup_error"] = f"Registration error: {exc}"
                            st.rerun()

                st.markdown(
                    '<div class="login-bottom">'
                    "Already have an account?&nbsp;"
                    '<span class="login-bottom-link">Sign In</span>'
                    "</div>",
                    unsafe_allow_html=True,
                )
                if st.button("← Sign In", key="go_signin"):
                    st.session_state["login_view"] = "signin"
                    st.session_state.pop("signup_error", None)
                    st.rerun()


def _render_client_dashboard_tab():
    """Client Finance Dashboard — adapted from Personal Finance Dashboard.

    Uses session state instead of SQLite so it works on Streamlit Cloud with
    no extra setup. CSV format is identical to the original app:
    columns: type, amount, category, date  (note is optional).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datetime import date as _date

    DEFAULT_CATEGORIES = [
        "Food", "Rent", "Transport", "Entertainment",
        "Utilities", "Shopping", "Health", "Salary", "Other",
    ]

    # ── Built-in sample data (pre-loads so the tab is never empty) ────────
    def _sample_df():
        return pd.DataFrame([
            {"type": "Income",  "amount": 8500.0, "category": "Salary",        "date": "2026-01-01", "note": "January"},
            {"type": "Expense", "amount": 1200.0, "category": "Rent",          "date": "2026-01-05", "note": ""},
            {"type": "Expense", "amount": 350.0,  "category": "Food",          "date": "2026-01-10", "note": ""},
            {"type": "Expense", "amount": 180.0,  "category": "Transport",     "date": "2026-01-15", "note": ""},
            {"type": "Expense", "amount": 120.0,  "category": "Utilities",     "date": "2026-01-20", "note": ""},
            {"type": "Income",  "amount": 8500.0, "category": "Salary",        "date": "2026-02-01", "note": "February"},
            {"type": "Expense", "amount": 1200.0, "category": "Rent",          "date": "2026-02-05", "note": ""},
            {"type": "Expense", "amount": 420.0,  "category": "Food",          "date": "2026-02-12", "note": ""},
            {"type": "Expense", "amount": 200.0,  "category": "Entertainment", "date": "2026-02-18", "note": ""},
            {"type": "Expense", "amount": 150.0,  "category": "Health",        "date": "2026-02-22", "note": ""},
            {"type": "Income",  "amount": 8500.0, "category": "Salary",        "date": "2026-03-01", "note": "March"},
            {"type": "Expense", "amount": 1200.0, "category": "Rent",          "date": "2026-03-05", "note": ""},
            {"type": "Expense", "amount": 380.0,  "category": "Food",          "date": "2026-03-10", "note": ""},
            {"type": "Expense", "amount": 250.0,  "category": "Shopping",      "date": "2026-03-16", "note": ""},
            {"type": "Expense", "amount": 90.0,   "category": "Utilities",     "date": "2026-03-20", "note": ""},
        ])

    # ── Session state init ────────────────────────────────────────────────
    st.session_state.setdefault("cd_df", _sample_df())
    st.session_state.setdefault("cd_budget", 5000.0)
    st.session_state.setdefault("cd_last_file", "")

    st.subheader("Client Dashboard")
    st.markdown(
        "<p class='section-copy'>Track income, expenses, and financial health with directional insights. "
        "Upload a CSV to replace data (columns: type, amount, category, date). "
        "Dashboard insights are for internal review and client discussion — review with the advisory team before sharing with clients.</p>",
        unsafe_allow_html=True,
    )

    # ── Controls row ──────────────────────────────────────────────────────
    ctrl_col, upload_col, budget_col = st.columns([1.4, 2, 1.2], gap="large")

    with ctrl_col:
        with st.container(border=True):
            _render_step_header("Add Entry", "")
            with st.form("cd_entry_form", clear_on_submit=True):
                type_ = st.selectbox("Type", ["Income", "Expense"])
                amount = st.number_input("Amount ($)", min_value=0.0, step=100.0)
                category = st.selectbox("Category", DEFAULT_CATEGORIES)
                date_in = st.date_input("Date", value=_date.today())
                note = st.text_input("Note (optional)")
                if st.form_submit_button("Add Entry", type="primary", use_container_width=True):
                    if amount <= 0:
                        st.error("Enter a valid amount")
                    else:
                        new_row = pd.DataFrame([{
                            "type": type_, "amount": float(amount),
                            "category": category, "date": date_in.isoformat(), "note": note,
                        }])
                        st.session_state["cd_df"] = pd.concat(
                            [st.session_state["cd_df"], new_row], ignore_index=True
                        )
                        st.success("Entry added")

    with upload_col:
        with st.container(border=True):
            _render_step_header("Import / Export", "")
            uploaded = st.file_uploader(
                "Upload transactions CSV (replaces all data)",
                type=["csv"],
                key="cd_uploader",
                label_visibility="collapsed",
            )
            if uploaded is not None:
                file_key = f"{uploaded.name}_{uploaded.size}"
                if file_key != st.session_state["cd_last_file"]:
                    try:
                        imp_df = pd.read_csv(uploaded)
                        imp_df.columns = imp_df.columns.str.lower()
                        required = {"type", "amount", "category", "date"}
                        if not required.issubset(set(imp_df.columns)):
                            st.error(f"CSV must have columns: {', '.join(sorted(required))}")
                        else:
                            if "note" not in imp_df.columns:
                                imp_df["note"] = ""
                            imp_df = imp_df[["type", "amount", "category", "date", "note"]].copy()
                            imp_df["type"] = imp_df["type"].str.capitalize()
                            imp_df["amount"] = pd.to_numeric(imp_df["amount"], errors="coerce").fillna(0)
                            # Replace — never append
                            st.session_state["cd_df"] = imp_df.reset_index(drop=True)
                            st.session_state["cd_last_file"] = file_key
                            st.success(f"Replaced all data with {len(imp_df)} rows from '{uploaded.name}'")
                            st.rerun()
                    except Exception as err:
                        st.error(f"Import failed: {err}")
                else:
                    st.caption(f"'{uploaded.name}' already loaded. Upload a different file to replace.")

            if not st.session_state["cd_df"].empty:
                st.download_button(
                    "Download CSV",
                    data=st.session_state["cd_df"].to_csv(index=False).encode("utf-8"),
                    file_name="client_transactions.csv",
                    mime="text/csv",
                )

    with budget_col:
        with st.container(border=True):
            _render_step_header("Budget", "")
            new_budget = st.number_input(
                "Monthly budget ($)",
                value=float(st.session_state["cd_budget"]),
                step=500.0,
            )
            if st.button("Save Budget", use_container_width=True):
                st.session_state["cd_budget"] = new_budget
                st.success("Saved")
            if st.button("Reset Sample Data", use_container_width=True):
                st.session_state["cd_df"] = _sample_df()
                st.session_state["cd_last_file"] = ""
                st.rerun()

    # ── Compute summary ───────────────────────────────────────────────────
    df = st.session_state["cd_df"].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    income   = df[df["type"] == "Income"]["amount"].sum()
    expenses = df[df["type"] == "Expense"]["amount"].sum()
    savings  = income - expenses
    budget   = float(st.session_state["cd_budget"])
    score    = int(min(max((income - expenses) / income * 100 if income > 0 else 0, 0), 100))

    # ── Key metrics ───────────────────────────────────────────────────────
    st.markdown("#### Key Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Income",   f"${income:,.0f}")
    m2.metric("Total Expenses", f"${expenses:,.0f}")
    m3.metric("Savings",        f"${savings:,.0f}")
    m4.metric("Health Score",   f"{score}/100")

    # ── Charts ────────────────────────────────────────────────────────────
    if df.empty:
        st.info("No data — add entries or upload a CSV.")
        return

    st.markdown("#### Visualizations")
    chart_col1, chart_col2 = st.columns(2)

    exp_df = df[df["type"] == "Expense"].groupby("category")["amount"].sum()
    exp_df = exp_df[exp_df > 0]

    with chart_col1:
        st.markdown("**Expenses by Category**")
        if not exp_df.empty:
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            wedges, _, autotexts = ax1.pie(
                exp_df.values, autopct="%1.0f%%", pctdistance=0.82, startangle=140
            )
            ax1.legend(
                wedges, exp_df.index,
                title="Categories", loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8,
            )
            ax1.set_aspect("equal")
            fig1.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)
        else:
            st.info("No expense data")

    with chart_col2:
        st.markdown("**Savings Over Time**")
        df_sorted = df.sort_values("date").copy()
        df_sorted["net"] = df_sorted.apply(
            lambda r: r["amount"] if r["type"] == "Income" else -r["amount"], axis=1
        )
        trend = df_sorted.groupby("date")["net"].sum().cumsum()
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        x_pos = list(range(len(trend)))
        ax2.plot(x_pos, trend.values, marker="o", markersize=4, linewidth=1.5, color="#2563eb")
        ax2.fill_between(x_pos, trend.values, alpha=0.1, color="#2563eb")
        labels = [d.strftime("%b %d") for d in trend.index]
        step = max(1, len(x_pos) // 8)
        ax2.set_xticks(x_pos[::step])
        ax2.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Cumulative Savings ($)")
        ax2.yaxis.get_major_formatter().set_scientific(False)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown("#### More Insights")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Expenses by Month**")
        df["month"] = df["date"].dt.to_period("M")
        monthly = df[df["type"] == "Expense"].groupby("month")["amount"].sum()
        if not monthly.empty:
            fig3, ax3 = plt.subplots(figsize=(5, 3.5))
            bars = ax3.bar(
                [str(m) for m in monthly.index], monthly.values,
                color="#2563eb", edgecolor="white",
            )
            ax3.set_ylabel("Expenses ($)")
            ax3.set_ylim(bottom=0)
            for bar in bars:
                h = bar.get_height()
                ax3.annotate(
                    f"${h:,.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8,
                )
            plt.xticks(rotation=45, ha="right")
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
        else:
            st.info("No expense data by month")

    with colB:
        st.markdown("**Spending by Weekday**")
        df["weekday"] = df["date"].dt.day_name()
        heat = df[df["type"] == "Expense"].groupby("weekday")["amount"].sum().reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        ).fillna(0)
        fig4, ax4 = plt.subplots(figsize=(6, 2))
        im = ax4.imshow(heat.values.reshape(1, -1), aspect="auto", cmap="YlOrRd")
        ax4.set_xticks(range(7))
        ax4.set_xticklabels(heat.index, rotation=30, ha="right", fontsize=8)
        ax4.set_yticks([])
        for j, val in enumerate(heat.values):
            ax4.text(
                j, 0, f"${val:,.0f}", ha="center", va="center",
                fontsize=7, color="black" if val < heat.max() * 0.6 else "white",
            )
        fig4.colorbar(im, ax=ax4, orientation="vertical", pad=0.02).set_label("Amount ($)", fontsize=8)
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    # ── Insights ──────────────────────────────────────────────────────────
    st.markdown("#### Insights")
    insights = []
    if expenses > 0:
        cat_totals = df[df["type"] == "Expense"].groupby("category")["amount"].sum()
        top_cat = cat_totals.idxmax()
        top_pct = cat_totals.max() / expenses
        if top_pct > 0.4:
            insights.append(f"⚠️ {top_pct:.0%} of expenses are in **{top_cat}**. Consider reviewing.")
        else:
            insights.append(f"✅ Highest expense category: **{top_cat}** ({top_pct:.0%} of expenses).")
    if budget > 0:
        this_month = pd.Timestamp.today().replace(day=1).normalize()
        monthly_exp = df[(df["type"] == "Expense") & (df["date"] >= this_month)]["amount"].sum()
        if monthly_exp > budget:
            insights.append(f"🔴 Monthly budget of ${budget:,.0f} exceeded — spent ${monthly_exp:,.0f} this month.")
        else:
            pct = monthly_exp / budget
            insights.append(f"🟢 Used {pct:.0%} of monthly budget (${monthly_exp:,.0f} of ${budget:,.0f}).")
    if income > 0 and savings / income < 0.05:
        insights.append("💡 Savings rate is very low. Consider reducing non-essential spending.")
    if not insights:
        insights.append("No major issues detected. Keep tracking!")
    for ins in insights:
        st.markdown(f"- {ins}")

    # ── Transaction table ─────────────────────────────────────────────────
    st.markdown("#### Transactions")
    display = df[["date", "type", "amount", "category", "note"]].copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display.sort_values("date", ascending=False), use_container_width=True, hide_index=True)

    _render_module_impact("dashboard")
    _render_module_roadmap("dashboard")


def main():
    _initialize_session_state()

    # ── Login gate — show login page until authenticated ──────────────────
    if not st.session_state.get("authenticated"):
        _render_login_page()
        return

    # ── Show user badge fixed top-right ───────────────────────────────────
    user_name = st.session_state.get("user_name", "")
    if user_name:
        st.markdown(
            f'<div class="user-top-badge">👤 {user_name}</div>',
            unsafe_allow_html=True,
        )

    _apply_page_style()
    _render_sidebar()
    _render_dashboard()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Internal AI Brain",
            "Bookkeeping Copilot",
            "Client Communication",
            "Strategy Content Studio",
            "Automations",
            "Client Dashboard",
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

    with tab6:
        _render_client_dashboard_tab()


if __name__ == "__main__":
    main()
