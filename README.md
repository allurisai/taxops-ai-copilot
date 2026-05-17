# TaxCopilot — Internal AI Command Center

**Current Status:** Deployed prototype demonstrating architecture, workflows, and business use cases for an internal AI assistant at a tax advisory firm.

---

## What It Does

TaxCopilot is a Streamlit-based internal AI tool with six integrated modules:

| Module | What It Does |
|---|---|
| **Internal AI Brain** | Source-based internal search across SOPs, strategy guides, and client documents |
| **Bookkeeping Copilot** | Flags transactions needing attention and suggests categories for bookkeeper review |
| **Client Communication** | Converts internal notes into advisor-reviewed client email drafts |
| **Strategy Content Studio** | Turns tax strategy notes into draft explainers, newsletters, social posts, and emails |
| **Automations** | Workflow blueprints for document intake, bookkeeping, and client communication |
| **Client Dashboard** | Financial health summaries from transaction data for advisor review |

---

## Human-in-the-Loop Design

All AI outputs are positioned as drafts for human review — not final outputs.

- **Internal AI Brain** — answers cite source documents and flag tax-sensitive questions for advisor review
- **Bookkeeping Copilot** — suggests categories; a bookkeeper approves before any import
- **Client Communication** — generates drafts; an advisor reviews before sending
- **Strategy Content Studio** — creates content drafts; team reviews for accuracy and brand alignment
- **Automations** — workflow blueprints; validated before any live automation
- **Client Dashboard** — directional insights reviewed by advisor before client use

---

## Architecture

- **Frontend:** Streamlit
- **AI:** Ollama (local, llama3.2:3b) with Groq cloud fallback (llama-3.1-8b-instant)
- **Retrieval:** FAISS vector store + Sentence Transformers (`all-MiniLM-L6-v2`), hybrid cosine + keyword scoring
- **Auth:** Supabase (PostgreSQL) + bcrypt password hashing
- **Data:** pandas, pypdf, matplotlib
- **Env:** python-dotenv

---

## Project Structure

```
taxops-ai-copilot/
├── app.py                        # Main Streamlit app — UI, routing, tab rendering
├── requirements.txt
├── supabase_config.py            # Auth helpers: register, login, bcrypt hashing
├── utils/
│   ├── ollama_client.py          # LLM interface: local Ollama + cloud API fallback
│   ├── qa_chain.py               # RAG pipeline: question routing, retrieval, answer generation
│   ├── vector_store.py           # FAISS index build and hybrid semantic/keyword search
│   ├── document_loader.py        # PDF, TXT, CSV ingestion and chunking
│   ├── summarizer.py             # Format-specific content generators (email, newsletter, report)
│   └── data_cleaner.py           # Bookkeeping CSV processing and flagging
└── data/
    └── demo_tax_firm/            # Sample workspace for testing
        ├── clients/
        ├── communications/
        ├── financials/
        ├── sops/
        ├── strategies/
        └── transactions/
```

---

## Setup

```bash
# 1. Clone and create virtual environment
python3 -m venv .venv-local
source .venv-local/bin/activate   # .venv-local\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# Edit .env — add SUPABASE_URL and SUPABASE_ANON_KEY
# Add GROQ_API_KEY for cloud LLM fallback (optional but recommended for deployment)

# 4. Run
streamlit run app.py
```

### Supabase Database Setup

Run this SQL in your Supabase SQL editor:

```sql
CREATE TABLE IF NOT EXISTS public.users (
    id            UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    full_name     TEXT        NOT NULL,
    email         TEXT        UNIQUE NOT NULL,
    password_hash TEXT        NOT NULL,
    company_name  TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

CREATE POLICY "allow_register" ON public.users
    FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "allow_login_lookup" ON public.users
    FOR SELECT TO anon USING (true);
```

---

## Production Hardening Roadmap

TaxCopilot is a deployed prototype designed for extension. Next implementation steps:

### Security & Access Control — Next Step
- Role-based permissions (tax, bookkeeping, operations, marketing)
- Admin/user roles with client-data access controls
- Secure session management and stronger authentication options

### Document Governance — Next Step
- Document approval workflow before indexing
- Source versioning and expiration dates for outdated SOPs
- Admin review queue and document owner metadata

### AI Evaluation & Quality Control — Next Step
- Golden test questions with expected answer comparisons
- Retrieval quality scoring and fallback threshold tuning
- Human feedback loops (helpful / not helpful) per answer
- Hallucination risk tracking and prompt/version testing

### Human Review Workflows — Next Step
- Advisor approval queue for tax-sensitive answers
- Bookkeeper approval queue before any export/import
- Marketing review queue for all content drafts
- Review status badges, history, and revision notes

### System Integrations — Production Upgrade
- QuickBooks Online API
- CRM and client portal integration
- Google Drive / Dropbox document sync
- Gmail/Outlook email draft integration
- Slack or Teams notifications
- Zapier, Make, or n8n automation workflows

### Monitoring & Business Metrics — Production Upgrade
- Questions answered, documents searched, transactions reviewed
- Issues flagged, drafts generated
- User feedback trends and fallback tracking

---

## First 90 Days Implementation Plan

**Days 1–15 — Discovery & Process Mapping**
Meet teams, review documentation structure, identify repeated internal questions and highest-impact automations.

**Days 16–45 — Internal AI Brain v1**
Load approved SOPs and strategy guides, build source-based search with review flags, test with a small team.

**Days 46–70 — Workflow Tools**
Improve bookkeeping review assistant, add client recap generator, add draft approval process, build first automation workflow.

**Days 71–90 — Adoption & Iteration**
Train team, collect feedback, improve retrieval and prompts, add integrations, prepare next roadmap.

---

## Evaluation Plan

| Signal | How to Measure |
|---|---|
| Answer quality | Golden test questions with expected answers |
| Retrieval accuracy | Cosine similarity scores + manual review sample |
| Human feedback | Helpful / not helpful per answer |
| Approval rate | Percentage of drafts approved without major edits |
| Usage | Questions asked, documents searched, drafts generated per session |
| Fallback tracking | Ollama vs cloud fallback rate |

---

## Integration Roadmap

QuickBooks Online · CRM · Client portal · Google Drive / Dropbox · Gmail / Outlook · Slack / Teams · Zapier / Make / n8n

---

## Security Notes

- Passwords are hashed with bcrypt — plain-text passwords are never stored
- Supabase credentials are loaded from `.env` — never hardcoded
- `.env` is included in `.gitignore`
- All AI outputs are drafts for human review — no automatic sending or publishing
