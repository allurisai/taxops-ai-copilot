"""
Supabase connection and authentication helpers for TaxCopilot.

Credentials are loaded from a .env file (never hardcoded).
Passwords are hashed with bcrypt — plain-text passwords are never stored.

Supabase table required (run this SQL in your Supabase SQL editor):

    CREATE TABLE IF NOT EXISTS public.users (
        id          UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
        full_name   TEXT        NOT NULL,
        email       TEXT        UNIQUE NOT NULL,
        password_hash TEXT      NOT NULL,
        company_name  TEXT,
        created_at  TIMESTAMPTZ DEFAULT NOW()
    );

    ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

    -- Allow anonymous inserts (registration)
    CREATE POLICY "allow_register" ON public.users
        FOR INSERT TO anon WITH CHECK (true);

    -- Allow anonymous select by email (login lookup)
    CREATE POLICY "allow_login_lookup" ON public.users
        FOR SELECT TO anon USING (true);
"""

import os
import bcrypt
from dotenv import load_dotenv

# ── Load credentials from .env ─────────────────────────────────────────────
load_dotenv()

SUPABASE_URL      = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

# ── Initialise Supabase client (lazy — only when credentials exist) ────────
_client = None


def get_client():
    """Return a cached Supabase client, or None if credentials are missing."""
    global _client
    if _client is not None:
        return _client
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return None
    try:
        from supabase import create_client
        _client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        return _client
    except Exception:
        return None


def is_configured() -> bool:
    """Return True when Supabase credentials are present in the environment."""
    return bool(SUPABASE_URL and SUPABASE_ANON_KEY)


# ── Password helpers ───────────────────────────────────────────────────────

def hash_password(plain_text: str) -> str:
    """Hash a plain-text password with bcrypt. Returns the hash as a string."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain_text.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_text: str, stored_hash: str) -> bool:
    """Return True when plain_text matches the bcrypt stored_hash."""
    try:
        return bcrypt.checkpw(plain_text.encode("utf-8"), stored_hash.encode("utf-8"))
    except Exception:
        return False


# ── Auth helpers ───────────────────────────────────────────────────────────

def register_user(full_name: str, email: str, password: str, company_name: str = "") -> dict:
    """
    Insert a new user into the users table.

    Returns:
        {"success": True, "user": {...}}          on success
        {"success": False, "error": "message"}    on failure
    """
    client = get_client()
    if client is None:
        return {"success": False, "error": "Supabase is not configured. Check your .env file."}

    email = email.strip().lower()

    # Check if email is already registered
    existing = get_user_by_email(email)
    if existing.get("user"):
        return {"success": False, "error": "An account with this email already exists."}

    pw_hash = hash_password(password)

    try:
        response = (
            client.table("users")
            .insert({
                "full_name":     full_name.strip(),
                "email":         email,
                "password_hash": pw_hash,
                "company_name":  company_name.strip(),
            })
            .execute()
        )
        if response.data:
            return {"success": True, "user": response.data[0]}
        return {"success": False, "error": "Registration failed — no data returned from database."}
    except Exception as exc:
        error_msg = str(exc)
        if "duplicate" in error_msg.lower() or "unique" in error_msg.lower():
            return {"success": False, "error": "An account with this email already exists."}
        return {"success": False, "error": f"Database error: {error_msg}"}


def login_user(email: str, password: str) -> dict:
    """
    Verify email + password against the users table.

    Returns:
        {"success": True, "user": {...}}          on success
        {"success": False, "error": "message"}    on failure
    """
    client = get_client()
    if client is None:
        return {"success": False, "error": "Supabase is not configured. Check your .env file."}

    email = email.strip().lower()
    result = get_user_by_email(email)

    if not result.get("user"):
        # Deliberately vague to prevent email enumeration
        return {"success": False, "error": "Invalid email or password. Please try again."}

    user = result["user"]
    if not verify_password(password, user["password_hash"]):
        return {"success": False, "error": "Invalid email or password. Please try again."}

    return {"success": True, "user": user}


def get_user_by_email(email: str) -> dict:
    """
    Fetch a single user row by email address.

    Returns:
        {"user": {...}}   if found
        {"user": None}    if not found
        {"error": "..."}  on database error
    """
    client = get_client()
    if client is None:
        return {"user": None, "error": "Supabase not configured."}

    try:
        response = (
            client.table("users")
            .select("id, full_name, email, password_hash, company_name, created_at")
            .eq("email", email.strip().lower())
            .limit(1)
            .execute()
        )
        if response.data:
            return {"user": response.data[0]}
        return {"user": None}
    except Exception as exc:
        return {"user": None, "error": str(exc)}
