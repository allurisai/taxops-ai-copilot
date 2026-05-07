# TaxCopilot Demo Flow

This walkthrough is designed for a 60 to 90 second recruiter demo.

## Goal

Show that TaxCopilot is a usable local product for:

- internal knowledge access
- bookkeeping review support
- client communication support
- strategy content generation

## Demo Setup

Before the demo:

1. Start Ollama
2. Start Streamlit
3. Open the app in the browser
4. Click `Load Demo Workspace`

## 60–90 Second Recruiter Walkthrough

### 1. Open With Product Framing

Say:

`This is TaxCopilot, a fully local AI workflow assistant for tax and bookkeeping teams. It searches internal documents, supports bookkeeping review, and generates client-ready outputs without using paid APIs.`

### 2. Show The Internal AI Brain

Open the `Internal AI Brain` tab.

Call out:

- the workspace snapshot
- the loaded document types
- the synthetic tax firm demo files

Use one of these questions:

- `What is the client onboarding process?`
- `How should uncategorized transactions be handled?`
- `What issues does this client have?`
- `Does this client use QuickBooks?`
- `What strategy applies when profits exceed $100,000?`

What to highlight:

- the answer is short and usable
- the app shows exact `Source`
- the app shows `Proof`
- the full chunk text is available in expanders

### 3. Show Bookkeeping Copilot

Open the `Bookkeeping Copilot` tab.

Click:

- `Use Demo Transactions: ABC`

What to highlight:

- rows needing review
- missing category matches
- duplicates found
- unusual amount detection
- suggested categories
- downloadable cleaned CSV

Say:

`This is positioned as AI-assisted bookkeeping triage, not automated accounting.`

### 4. Show Client Communication

Open the `Client Communication` tab.

Use the default selected docs and click:

- `Generate Client Report`

What to highlight:

- summary
- key issues
- recommendations
- action items
- client email draft

Optional:

- click `Explain for Client`

### 5. Show Strategy Content Studio

Open the `Strategy Content Studio` tab.

Choose:

- `educational explainer`

Click:

- `Generate Content`

What to highlight:

- the content stays grounded in the selected internal docs
- the tone is business-friendly
- the module converts internal strategy into client-facing language

## Suggested Script

`The first module is the Internal AI Brain, where a team can search SOPs and client notes with proof-backed answers. The second module is a bookkeeping review workflow that flags duplicate imports, missing categories, and unusual transactions. The third module turns internal notes into client-ready communication, and the fourth turns strategy notes into external-facing content. Everything here runs locally with Ollama and sentence-transformers.`

## Best Demo Questions

- `What is the client onboarding process?`
- `How should uncategorized transactions be handled?`
- `What issues does this client have?`
- `Does this client use QuickBooks?`
- `What strategy applies when profits exceed $100,000?`
- `Summarize the financial risks in this note.`

## If You Need A Shorter Demo

Use only these three moves:

1. Load demo workspace
2. Ask `How should uncategorized transactions be handled?`
3. Open `Bookkeeping Copilot` and show `Use Demo Transactions: ABC`
