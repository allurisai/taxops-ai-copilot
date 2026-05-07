import re

from utils.ollama_client import query_ollama


# ── Pattern helpers ──────────────────────────────────────────────────────────

# Matches "N lines" / "N line" anywhere in the instruction
_LINE_COUNT_RE = re.compile(r'\b(\d+)\s*lines?\b', re.IGNORECASE)

# Matches instructions that impose a length/format constraint tight enough to
# conflict with the rigid 5-section report template.
_FORMAT_OVERRIDE_RE = re.compile(
    r'\b(\d+\s*lines?|short|brief|bullet|one\s+paragraph|single\s+paragraph|concise)\b',
    re.IGNORECASE,
)

# "X only" → which report dict key(s) to keep.
# Evaluated in order; first match wins.
_SECTION_ONLY_RULES = [
    (re.compile(r'\bsummary\s+only\b|\bonly\s+(the\s+)?summary\b', re.IGNORECASE), ["summary"]),
    (re.compile(r'\bkey\s+issues?\s+only\b|\bissues?\s+only\b|\bonly\s+(the\s+)?issues?\b', re.IGNORECASE), ["key_issues"]),
    (re.compile(r'\brecommendations?\s+only\b|\bonly\s+(the\s+)?recommendations?\b', re.IGNORECASE), ["recommendations"]),
    (re.compile(r'\baction\s+items?\s+only\b|\bonly\s+(the\s+)?action\s+items?\b', re.IGNORECASE), ["action_items"]),
    (re.compile(r'\bemail\s+(draft\s+)?only\b|\bonly\s+(the\s+)?email\b', re.IGNORECASE), ["client_email"]),
]


# ── Utility functions ────────────────────────────────────────────────────────

def _is_format_override(instruction: str) -> bool:
    return bool(_FORMAT_OVERRIDE_RE.search(instruction))


def _get_section_filter(instruction: str):
    """Return a list of dict keys to keep, or None if no 'X only' instruction."""
    if not instruction:
        return None
    for pattern, keys in _SECTION_ONLY_RULES:
        if pattern.search(instruction):
            return keys
    return None


def _enforce_length(text: str, user_instruction: str) -> str:
    """Post-generation hard trim: if user asked for N lines, keep only N lines.
    This runs AFTER the model generates, guaranteeing the constraint is met."""
    if not user_instruction or not text:
        return text
    match = _LINE_COUNT_RE.search(user_instruction)
    if match:
        n = int(match.group(1))
        non_empty = [line for line in text.splitlines() if line.strip()]
        if len(non_empty) > n:
            return '\n'.join(non_empty[:n])
    return text


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting that small models add despite instructions.
    Keeps the text content, strips the markers."""
    # **bold** and *italic* → plain text
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\*(.+?)\*', r'\1', text, flags=re.DOTALL)
    # ## Headings → plain text
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    return text


def _parse_sections(response_text, section_names):
    """Parse a model response that uses fixed uppercase section headings."""
    parsed_sections = {}
    for index, section_name in enumerate(section_names):
        next_section = section_names[index + 1] if index + 1 < len(section_names) else None
        pattern = (
            rf"{section_name}:\s*(.*?)\s*(?={next_section}:)"
            if next_section
            else rf"{section_name}:\s*(.*)"
        )
        match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
        parsed_sections[section_name] = match.group(1).strip() if match else ""
    return parsed_sections


# ── Format-specific prompt builders ─────────────────────────────────────────
# Each format has its own prompt so the model knows exactly what structure to
# produce. User instruction is placed at the top AND repeated immediately
# before Output: — small local models forget constraints that appear far from
# the generation trigger.

def _ensure_email_structure(text: str) -> str:
    """Post-processing guarantee: strip markdown then add Subject / Dear /
    closing if the model omitted any of them."""
    text = _strip_markdown(text).strip()

    # ── Subject line ─────────────────────────────────────────────────────────
    if not re.search(r'^Subject:', text, re.IGNORECASE | re.MULTILINE):
        # Try to derive a subject from the first non-empty line
        first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")
        # If first line looks like a heading (short, no period) use it as subject
        if first_line and len(first_line) < 80 and not first_line.endswith('.'):
            text = text[len(first_line):].lstrip('\n')
            subject = first_line
        else:
            subject = "Update from Your Tax & Bookkeeping Team"
        text = f"Subject: {subject}\n\n{text}"

    # ── Salutation ───────────────────────────────────────────────────────────
    if not re.search(r'^Dear\b', text, re.IGNORECASE | re.MULTILINE):
        # Insert "Dear Client," after the subject line
        text = re.sub(
            r'(Subject:[^\n]*\n+)',
            r'\1Dear Client,\n\n',
            text,
            count=1,
            flags=re.IGNORECASE,
        )

    # ── Closing ──────────────────────────────────────────────────────────────
    closings = ("best regards", "sincerely", "kind regards", "regards,", "warm regards")
    if not any(c in text.lower() for c in closings):
        text = text.rstrip() + "\n\nBest regards,\nYour Tax & Bookkeeping Team"

    return text


def _email_prompt(context: str, tone: str, user_instruction: str) -> str:
    """Build a prompt for client email draft.

    Shows the model an exact fill-in-the-blank template so it knows every part
    required. Ends with 'Subject:' so the first generated token is the subject
    line. _ensure_email_structure() + _strip_markdown() clean up afterwards.
    """
    length_hint = f"{user_instruction}. " if user_instruction else ""
    return f"""Write a short professional client email. {length_hint}Tone: {tone}.
Use only the facts from the context below. Do not invent anything.
Write in plain sentences. Do NOT use bullet points, numbered lists, or markdown like ** or ##.

Context:
{context}

Fill in this email template exactly. Replace each [...] with real content:

Subject: [short subject line]
Dear Client,
[2-3 plain sentences summarising the key point from the context]
[1 sentence about next steps if relevant]
Best regards,
Your Tax & Bookkeeping Team

Your completed email:
Subject:"""


def _newsletter_prompt(context: str, tone: str, user_instruction: str) -> str:
    instr = f"\nUser Instruction (FOLLOW EXACTLY — overrides everything): {user_instruction}" if user_instruction else ""
    reminder = f"\nReminder — follow this instruction EXACTLY: {user_instruction}" if user_instruction else ""
    return f"""You are writing a short newsletter.

Output type: short newsletter draft
Tone: {tone}{instr}

Context (use only this — do not invent facts):
{context}

Rules:
- NO Subject line
- NO "Dear" greeting
- NO email-style closing or signature
- Start with a short heading (one line, plain text — NOT "Subject:")
- Write 2–4 short paragraphs OR use bullet points if asked
- Informative and engaging
- No markdown (** or ##)
- Follow user instruction EXACTLY for length and format
- Do not add content not supported by the context
{reminder}

Output:"""


def _explainer_prompt(context: str, tone: str, user_instruction: str) -> str:
    instr = f"\nUser Instruction (FOLLOW EXACTLY — overrides everything): {user_instruction}" if user_instruction else ""
    reminder = f"\nReminder — follow this instruction EXACTLY: {user_instruction}" if user_instruction else ""
    return f"""You are writing an educational explainer.

Output type: educational explainer
Tone: {tone}{instr}

Context (use only this — do not invent facts):
{context}

Rules:
- NO email structure (no Subject, no Dear, no closing)
- NO conversational greeting
- Structured explanation with clear logical flow
- Plain language — minimize jargon
- Educational and informative
- No markdown (** or ##)
- Follow user instruction EXACTLY for length and format
- Do not add content not supported by the context
{reminder}

Output:"""


def _social_post_prompt(context: str, tone: str, user_instruction: str) -> str:
    instr = f"\nUser Instruction (FOLLOW EXACTLY — overrides everything): {user_instruction}" if user_instruction else ""
    reminder = f"\nReminder — follow this instruction EXACTLY: {user_instruction}" if user_instruction else ""
    return f"""You are writing a social media post.

Output type: social post draft
Tone: {tone}{instr}

Context (use only this — do not invent facts):
{context}

Rules:
- NO email structure (no Subject, no Dear, no closing)
- Short and punchy — 1 to 4 sentences unless user instructs otherwise
- Engaging and direct
- No markdown (** or ##)
- Follow user instruction EXACTLY
- Do not add content not supported by the context
{reminder}

Output:"""


def _generic_content_prompt(context: str, output_type: str, tone: str, user_instruction: str) -> str:
    instr = f"\nUser Instruction (FOLLOW EXACTLY — overrides everything): {user_instruction}" if user_instruction else ""
    reminder = f"\nReminder — follow this instruction EXACTLY: {user_instruction}" if user_instruction else ""
    return f"""You are an AI assistant that must strictly follow user instructions.

Output type: {output_type}
Tone: {tone}{instr}

Context (use only this — do not invent facts):
{context}

Rules:
- Follow user instruction EXACTLY
- Do not add extra content or sections
- Do not exceed requested length
- Use the correct format for the selected output type
- Keep output clean and professional
- No markdown (** or ##)
{reminder}

Output:"""





# ── Fallbacks ────────────────────────────────────────────────────────────────

def _fallback_report_output(context_preview):
    return {
        "summary": f"Working summary based on the selected documents: {context_preview}",
        "key_issues": "- Review the selected documents for client-specific risks and unresolved bookkeeping questions.",
        "recommendations": "- Confirm supporting records\n- Review flagged bookkeeping items\n- Prepare client follow-up on open questions",
        "action_items": "- Validate the latest numbers\n- Confirm missing documentation\n- Send a concise client update",
        "client_email": (
            "Subject: Update on Your Tax and Bookkeeping Review\n\n"
            "Dear Client,\n\n"
            "We reviewed the available documents and prepared a short internal summary. "
            "We are validating a few items and will follow up with any action items that need your input.\n\n"
            "Best regards,\nTaxCopilot Team"
        ),
    }


def _fallback_communication_output(notes):
    preview = " ".join(notes.split())[:500]
    return {
        "email": (
            "Subject: Update on Your Tax and Bookkeeping Review\n\n"
            "Dear Client,\n\n"
            f"We reviewed the notes available to our team. Key points include: {preview}\n\n"
            "We will follow up with any additional questions or action items.\n\n"
            "Best regards,\nTaxCopilot Team"
        ),
        "explanation": (
            "We reviewed the internal notes and translated them into a simpler client-ready explanation. "
            "The main purpose is to share the key points clearly without adding unsupported advice."
        ),
        "summary": (
            "Client-ready summary:\n"
            f"- Main update: {preview}\n"
            "- Next step: confirm open items and keep documentation ready."
        ),
    }


# ── Explicit public generators (one per format) ───────────────────────────────
# app.py routes directly to these — no hidden internal dispatch.

def generate_email(context: str, guidance: str = "", tone: str = "Professional") -> str:
    """Generate a client email. Guarantees Subject / Dear / closing via post-processing."""
    guidance = guidance.strip()
    prompt = _email_prompt(context, tone, guidance)
    output = query_ollama(prompt).strip()
    output = _ensure_email_structure(output)   # strips markdown + injects email structure
    return _enforce_length(output, guidance)


def generate_newsletter(context: str, guidance: str = "", tone: str = "Professional") -> str:
    """Generate a short newsletter — no email structure."""
    guidance = guidance.strip()
    prompt = _newsletter_prompt(context, tone, guidance)
    output = _strip_markdown(query_ollama(prompt).strip())
    return _enforce_length(output, guidance)


def generate_explainer(context: str, guidance: str = "", tone: str = "Professional") -> str:
    """Generate an educational explainer — no email structure."""
    guidance = guidance.strip()
    prompt = _explainer_prompt(context, tone, guidance)
    output = _strip_markdown(query_ollama(prompt).strip())
    return _enforce_length(output, guidance)


def generate_social_post(context: str, guidance: str = "", tone: str = "Professional") -> str:
    """Generate a social media post — short, punchy, no email structure."""
    guidance = guidance.strip()
    prompt = _social_post_prompt(context, tone, guidance)
    output = _strip_markdown(query_ollama(prompt).strip())
    return _enforce_length(output, guidance)


def generate_strategy_content(context, output_type, tone="Professional", extra_instruction=""):
    """Backward-compat wrapper — routes to the explicit named generators above."""
    key = output_type.lower().strip()
    guidance = extra_instruction.strip() if extra_instruction else ""
    if "email" in key:
        return generate_email(context, guidance, tone)
    if "newsletter" in key:
        return generate_newsletter(context, guidance, tone)
    if "explainer" in key:
        return generate_explainer(context, guidance, tone)
    if "social" in key:
        return generate_social_post(context, guidance, tone)
    return generate_explainer(context, guidance, tone)


def generate_client_report(context, extra_instruction=""):
    """Generate a structured client report.

    Priority order:
    1. Section filter ("summary only", "email only", etc.) → return only that section
    2. Format override ("3 lines", "short") → bypass 5-section template, free-form output
    3. Default → full 5-section structured report

    _enforce_length() is applied as a hard post-processing trim in all paths.
    """
    user_instruction = extra_instruction.strip() if extra_instruction else ""

    # Priority 1 — section-only filter (generate full report, then slice)
    section_filter = _get_section_filter(user_instruction)

    # Priority 2 — format override with no section filter: bypass the template
    if user_instruction and _is_format_override(user_instruction) and not section_filter:
        prompt = _generic_content_prompt(context, "client report", "professional", user_instruction)
        output = _enforce_length(query_ollama(prompt).strip(), user_instruction)
        return {
            "summary": output,
            "key_issues": "",
            "recommendations": "",
            "action_items": "",
            "client_email": "",
        }

    # Priority 3 — structured 5-section report
    instruction_block = (
        f"User instruction (apply to ALL sections — FOLLOW EXACTLY):\n{user_instruction}\n\n"
        if user_instruction else ""
    )

    prompt = f"""You are a professional report writer.

{instruction_block}Generate a structured client report using exactly this format:

SUMMARY:
<1-2 sentences>

KEY_ISSUES:
<bullet list>

RECOMMENDATIONS:
<bullet list>

ACTION_ITEMS:
<bullet list>

CLIENT_EMAIL:
<professional email — must include Subject:, Dear [Client],, and a closing>

Rules:
- Apply user instruction to every section for length, tone, and format.
- No markdown like ** or ##.
- Base everything only on the context below.
- Note unclear items as things to verify rather than inventing facts.

Context:
{context}
"""

    raw_response = query_ollama(prompt)
    sections = _parse_sections(
        raw_response,
        ["SUMMARY", "KEY_ISSUES", "RECOMMENDATIONS", "ACTION_ITEMS", "CLIENT_EMAIL"],
    )

    if not any(sections.values()):
        result = _fallback_report_output(context[:500])
    else:
        fallback = _fallback_report_output(context[:500])
        result = {
            "summary": sections["SUMMARY"] or fallback["summary"],
            "key_issues": sections["KEY_ISSUES"] or "- Review the selected documents for unresolved issues.",
            "recommendations": sections["RECOMMENDATIONS"] or "- Validate supporting records",
            "action_items": sections["ACTION_ITEMS"] or "- Confirm the next internal follow-up step",
            "client_email": sections["CLIENT_EMAIL"] or fallback["client_email"],
        }

    # Apply section filter — keep only the requested section(s)
    if section_filter:
        filtered = {k: "" for k in result}
        for key in section_filter:
            filtered[key] = result.get(key, "")
        return filtered

    # Apply length enforcement to each section if user specified a line count
    if user_instruction:
        result = {k: _enforce_length(v, user_instruction) for k, v in result.items()}

    return result


def explain_for_client(context, extra_instruction=""):
    """Explain the selected material in plain English for a client."""
    user_instruction = extra_instruction.strip() if extra_instruction else ""
    prompt = _explainer_prompt(context, "client-friendly", user_instruction)
    output = query_ollama(prompt).strip()
    return _enforce_length(output, user_instruction)


def generate_client_communication(notes):
    """Generate three communication outputs (backward-compatibility function)."""
    prompt = f"""Write three short outputs from the notes below. Use exactly this format:

EMAIL:
<professional client email with Subject:, Dear [Client],, and closing>

EXPLANATION:
<plain-language explanation>

SUMMARY:
<short summary>

Rules:
- Keep each section short and factual.
- No markdown like ** or ##.
- Base everything only on the notes provided.
- Do not invent tax advice.

Notes:
{notes}
"""
    raw_response = query_ollama(prompt)
    sections = _parse_sections(raw_response, ["EMAIL", "EXPLANATION", "SUMMARY"])

    if not any(sections.values()):
        return _fallback_communication_output(notes)

    fallback = _fallback_communication_output(notes)
    return {
        "email": sections["EMAIL"] or fallback["email"],
        "explanation": sections["EXPLANATION"] or fallback["explanation"],
        "summary": sections["SUMMARY"] or fallback["summary"],
    }
