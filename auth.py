import sqlite3
import hashlib
import secrets
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

DB_PATH = "statify.db"

# ─────────────────────────────────────────────
# Database setup
# ─────────────────────────────────────────────

def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            firstname TEXT    NOT NULL,
            email     TEXT    NOT NULL UNIQUE,
            password  TEXT    NOT NULL,
            created   TEXT    NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS reset_tokens (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            email      TEXT NOT NULL,
            token      TEXT NOT NULL UNIQUE,
            expires_at TEXT NOT NULL,
            used       INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# Password hashing
# ─────────────────────────────────────────────

def hash_password(password: str) -> str:
    """SHA-256 hash with a random salt, stored as salt:hash."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, stored: str) -> bool:
    """Verify a plain password against the stored salt:hash."""
    try:
        salt, hashed = stored.split(":", 1)
        return hashlib.sha256((salt + password).encode()).hexdigest() == hashed
    except ValueError:
        return False


# ─────────────────────────────────────────────
# User management
# ─────────────────────────────────────────────

def create_user(firstname: str, email: str, password: str):
    """
    Insert a new user.
    Returns (True, None) on success or (False, error_message) on failure.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (firstname, email, password, created) VALUES (?, ?, ?, ?)",
            (firstname, email.lower(), hash_password(password), datetime.utcnow().isoformat())
        )
        conn.commit()
        return True, None
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    finally:
        conn.close()


def authenticate_user(email: str, password: str):
    """
    Returns (user_row_dict, None) on success or (None, error_message) on failure.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email = ?", (email.lower(),))
    row = c.fetchone()
    conn.close()

    if not row:
        return None, "No account found with that email address."

    # row: (id, firstname, email, password, created)
    stored_hash = row[3]
    if not verify_password(password, stored_hash):
        return None, "Incorrect password. Please try again."

    return {"id": row[0], "firstname": row[1], "email": row[2]}, None


def email_exists(email: str) -> bool:
    """Check whether an email is registered."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE email = ?", (email.lower(),))
    result = c.fetchone()
    conn.close()
    return result is not None


# ─────────────────────────────────────────────
# Password-reset tokens
# ─────────────────────────────────────────────

def create_reset_token(email: str) -> str:
    """Generate a secure token valid for 1 hour and store it."""
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.utcnow() + timedelta(hours=1)).isoformat()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Invalidate any previous tokens for this email
    c.execute("DELETE FROM reset_tokens WHERE email = ?", (email.lower(),))
    c.execute(
        "INSERT INTO reset_tokens (email, token, expires_at) VALUES (?, ?, ?)",
        (email.lower(), token, expires_at)
    )
    conn.commit()
    conn.close()
    return token


def validate_reset_token(token: str):
    """
    Returns (email, None) if valid, or (None, error_message) if not.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT email, expires_at, used FROM reset_tokens WHERE token = ?",
        (token,)
    )
    row = c.fetchone()
    conn.close()

    if not row:
        return None, "Invalid or expired reset link."

    email, expires_at, used = row

    if used:
        return None, "This reset link has already been used."

    if datetime.utcnow() > datetime.fromisoformat(expires_at):
        return None, "This reset link has expired. Please request a new one."

    return email, None


def mark_token_used(token: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE reset_tokens SET used = 1 WHERE token = ?", (token,))
    conn.commit()
    conn.close()


def update_password(email: str, new_password: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE users SET password = ? WHERE email = ?",
        (hash_password(new_password), email.lower())
    )
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# Email sending
# ─────────────────────────────────────────────

def send_reset_email(to_email: str, token: str, base_url: str) -> bool:
    """
    Send a password-reset email.
    Configure via environment variables:
        MAIL_SERVER   (default: smtp.gmail.com)
        MAIL_PORT     (default: 587)
        MAIL_USERNAME
        MAIL_PASSWORD
        MAIL_FROM     (default: MAIL_USERNAME)
    Returns True on success, False on failure.
    """
    mail_server   = os.environ.get("MAIL_SERVER",   "smtp.gmail.com")
    mail_port     = int(os.environ.get("MAIL_PORT", 587))
    mail_username = os.environ.get("MAIL_USERNAME", "")
    mail_password = os.environ.get("MAIL_PASSWORD", "")
    mail_from     = os.environ.get("MAIL_FROM", mail_username)

    reset_link = f"{base_url}/reset-password/{token}"

    # ── HTML email body ──────────────────────────────────────────────────────
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #05070a; color: #e2e8f0; margin: 0; padding: 0; }}
        .container {{ max-width: 520px; margin: 40px auto; background: linear-gradient(135deg, #1a1c2e, #0d0f1a);
                      border: 1px solid rgba(0,242,255,0.15); border-radius: 20px; overflow: hidden; }}
        .header {{ background: linear-gradient(90deg, #00f2ff22, #0072ff22); padding: 36px 40px 28px;
                   border-bottom: 1px solid rgba(0,242,255,0.1); text-align: center; }}
        .logo {{ display: inline-flex; align-items: center; justify-content: center;
                 width: 48px; height: 48px; border: 2px solid #00f2ff;
                 border-radius: 14px; font-size: 22px; font-weight: 800; color: #00f2ff;
                 margin-bottom: 16px; box-shadow: 0 0 20px rgba(0,242,255,0.3); }}
        h1 {{ margin: 0; font-size: 22px; font-weight: 700;
              background: linear-gradient(90deg, #fff, #7dd3fc); -webkit-background-clip: text;
              -webkit-text-fill-color: transparent; }}
        .body {{ padding: 36px 40px; }}
        p {{ color: #94a3b8; line-height: 1.7; margin: 0 0 20px; font-size: 15px; }}
        .btn {{ display: inline-block; padding: 14px 36px;
                background: linear-gradient(90deg, #00f2ff, #0072ff);
                color: white !important; font-weight: 700; text-decoration: none;
                border-radius: 50px; font-size: 14px; letter-spacing: 0.05em;
                box-shadow: 0 0 20px rgba(0,242,255,0.35); }}
        .link-box {{ margin-top: 24px; padding: 14px 18px;
                     background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
                     border-radius: 10px; word-break: break-all; font-size: 12px; color: #64748b; }}
        .footer {{ padding: 20px 40px; border-top: 1px solid rgba(255,255,255,0.05);
                   text-align: center; font-size: 12px; color: #475569; }}
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <div class="logo">S</div>
          <h1>Reset Your Password</h1>
        </div>
        <div class="body">
          <p>We received a request to reset the password for your <strong style="color:#e2e8f0">Statify</strong> account. Click the button below to choose a new password.</p>
          <p style="text-align:center; margin: 32px 0;">
            <a href="{reset_link}" class="btn">Reset Password</a>
          </p>
          <p>This link will expire in <strong style="color:#e2e8f0">1 hour</strong>. If you didn't request a password reset, you can safely ignore this email.</p>
          <div class="link-box">
            Or copy this link: {reset_link}
          </div>
        </div>
        <div class="footer">© Statify · This is an automated message, please do not reply.</div>
      </div>
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Reset your Statify password"
    msg["From"]    = mail_from
    msg["To"]      = to_email
    msg.attach(MIMEText(html_body, "html"))

    # If no credentials set, just print the link (dev mode)
    if not mail_username or not mail_password:
        print(f"\n[DEV MODE] Password reset link for {to_email}:\n{reset_link}\n")
        return True

    try:
        with smtplib.SMTP(mail_server, mail_port) as server:
            server.ehlo()
            server.starttls()
            server.login(mail_username, mail_password)
            server.sendmail(mail_from, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")
        return False
