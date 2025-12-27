"""Unit tests for GitHub security (T005)."""

import pytest
import hmac
import hashlib
from src.supervisor.github import verify_signature

def test_verify_signature_valid():
    secret = "test_secret"
    body = b'{"action": "opened"}'
    computed_hash = hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha256
    ).hexdigest()
    signature_header = f"sha256={computed_hash}"
    
    assert verify_signature(body, signature_header, secret) is True

def test_verify_signature_invalid_secret():
    secret = "test_secret"
    body = b'{"action": "opened"}'
    signature_header = "sha256=wrong_hash"
    
    assert verify_signature(body, signature_header, secret) is False

def test_verify_signature_missing_header():
    assert verify_signature(b"body", "", "secret") is False

def test_verify_signature_wrong_format():
    assert verify_signature(b"body", "sha1=abcd", "secret") is False

def test_verify_signature_empty_secret():
    assert verify_signature(b"body", "sha256=abcd", "") is False
