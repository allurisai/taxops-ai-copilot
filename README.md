# TaxCopilot

TaxCopilot is an internal AI assistant for tax and bookkeeping teams.

It helps teams:

- Search SOPs, client notes, strategy guides, and financial documents with source-backed answers
- Review bookkeeping exports and flag transactions needing attention
- Generate client-ready summaries, action items, and email drafts
- Convert internal strategy notes into polished client-facing content
- Automate repeatable workflows across departments

## Architecture

- **Streamlit** — Web interface accessible to non-technical team members
- **Sentence-Transformers + FAISS** — Semantic document retrieval with hybrid keyword scoring
- **Ollama (local) / Cloud API (fallback)** — Flexible LLM backend
- **Pandas** — Bookkeeping CSV analysis and cleanup

## Modules

### Internal AI Brain
Search internal knowledge and get answers with confidence scoring, source citations, and proof snippets. Supports factual, yes/no, list, compound, assessment, and workflow questions.

### Bookkeeping Copilot
Review transaction CSVs with automated duplicate detection, missing value flagging, vendor normalization, and category suggestions. Export cleaned data for accounting review.

### Client Communication
Generate structured client reports with summaries, key issues, recommendations, action items, and professional email drafts — all grounded in actual internal documents.

### Strategy Content Studio
Convert internal tax strategy notes into client emails, educational explainers, newsletter drafts, and social media posts.

### Automations
Pre-built workflow triggers connecting document intake, bookkeeping review, client communication, and content generation into automated pipelines.

## Setup

```bash
python3 -m venv .venv-local
source .venv-local/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b
ollama serve
streamlit run app.py
```

## Cloud Deployment

Set `ANTHROPIC_API_KEY` as an environment variable to enable cloud LLM fallback when Ollama is not available. This allows deployment to Streamlit Cloud or any hosted environment.

## Project Structure

```
taxops-ai-copilot/
├── app.py                        # Main Streamlit app — UI, routing, tab rendering
├── requirements.txt
├── utils/
│   ├── ollama_client.py          # LLM interface: local Ollama + cloud API fallback
│   ├── qa_chain.py               # RAG pipeline: question routing, retrieval, answer generation
│   ├── vector_store.py           # FAISS index build and hybrid semantic/keyword search
│   ├── document_loader.py        # PDF, TXT, CSV ingestion and chunking
│   ├── summarizer.py             # Format-specific content generators (email, newsletter, report)
│   └── data_cleaner.py           # Bookkeeping CSV processing and flagging
└── data/
    └── demo_tax_firm/            # Sample workspace for testing
        ├── clients/              # Client profiles and meeting notes
        ├── communications/       # Sample email and communication notes
        ├── financials/           # Financial summaries and year-end notes
        ├── sops/                 # Standard operating procedures
        ├── strategies/           # Tax strategy guides and checklists
        └── transactions/         # Sample bookkeeping CSVs (ABC, XYZ)
```
