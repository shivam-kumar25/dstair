"""
Symmetric encryption utilities for at-rest secret storage.
Uses Fernet (AES-128-CBC + HMAC-SHA256) with a key derived from the app SECRET_KEY.

Important: If SECRET_KEY changes, previously encrypted values will be unreadable.
The key is derived fresh on every call so SECRET_KEY rotations take effect immediately
without requiring a server restart.
"""
import base64
import hashlib
import logging

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

# Cache Fernet instances keyed by the SECRET_KEY value they were derived from.
# This means a SECRET_KEY rotation is picked up immediately (old key → new cache
# entry, old entry is no longer used) while avoiding repeated SHA-256 + Fernet
# object construction on every decrypt/encrypt call (e.g. masked_key rendering).
_fernet_cache: dict[str, "Fernet"] = {}


def _get_fernet() -> "Fernet":
    """Return a Fernet cipher for the current app's SECRET_KEY.

    Results are cached per unique SECRET_KEY value so rotation is transparent
    (a new key produces a new cache entry) but the common case (same key) pays
    the construction cost only once.
    """
    from flask import current_app
    secret = current_app.config["SECRET_KEY"]
    if secret not in _fernet_cache:
        derived = hashlib.sha256(secret.encode()).digest()
        key = base64.urlsafe_b64encode(derived)
        _fernet_cache[secret] = Fernet(key)
    return _fernet_cache[secret]


def encrypt_value(plaintext: str) -> str:
    """Encrypt a plaintext string, returning a Fernet token string."""
    if not plaintext:
        return plaintext
    fernet = _get_fernet()
    return fernet.encrypt(plaintext.encode()).decode()


def decrypt_value(ciphertext: str) -> str:
    """
    Decrypt a Fernet token string back to plaintext.
    If the value isn't a valid Fernet token (e.g., legacy plaintext),
    return it as-is for backward compatibility.
    """
    if not ciphertext:
        return ciphertext
    fernet = _get_fernet()
    try:
        return fernet.decrypt(ciphertext.encode()).decode()
    except (InvalidToken, Exception):
        # Value is likely legacy plaintext — return as-is
        logger.debug("Decryption failed — treating value as legacy plaintext.")
        return ciphertext


def is_encrypted(value: str) -> bool:
    """Check if a value looks like a Fernet token (starts with 'gAAAAA')."""
    return bool(value) and value.startswith('gAAAAA')
