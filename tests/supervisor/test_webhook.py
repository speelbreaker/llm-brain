"""Tests for webhook signature validation."""

import hashlib
import hmac

import pytest

from src.supervisor.github import verify_signature


class TestWebhookSignature:
    """Tests for GitHub webhook signature validation."""
    
    def test_valid_signature_accepted(self):
        """Test that valid HMAC SHA-256 signature is accepted."""
        secret = "test_secret_key"
        payload = b'{"action": "opened", "pull_request": {}}'
        
        expected = hmac.new(
            secret.encode("utf-8"),
            payload,
            hashlib.sha256
        ).hexdigest()
        signature = f"sha256={expected}"
        
        assert verify_signature(payload, signature, secret) is True
    
    def test_invalid_signature_rejected(self):
        """Test that invalid signature is rejected."""
        secret = "test_secret_key"
        payload = b'{"action": "opened", "pull_request": {}}'
        
        wrong_signature = "sha256=0000000000000000000000000000000000000000000000000000000000000000"
        
        assert verify_signature(payload, wrong_signature, secret) is False
    
    def test_missing_signature_rejected(self):
        """Test that missing signature is rejected."""
        secret = "test_secret_key"
        payload = b'{"action": "opened"}'
        
        assert verify_signature(payload, "", secret) is False
    
    def test_wrong_prefix_rejected(self):
        """Test that signature with wrong prefix is rejected."""
        secret = "test_secret_key"
        payload = b'{"action": "opened"}'
        
        expected = hmac.new(
            secret.encode("utf-8"),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        assert verify_signature(payload, f"md5={expected}", secret) is False
        assert verify_signature(payload, expected, secret) is False
    
    def test_modified_payload_rejected(self):
        """Test that modified payload fails validation."""
        secret = "test_secret_key"
        original_payload = b'{"action": "opened"}'
        modified_payload = b'{"action": "closed"}'
        
        signature = hmac.new(
            secret.encode("utf-8"),
            original_payload,
            hashlib.sha256
        ).hexdigest()
        
        assert verify_signature(modified_payload, f"sha256={signature}", secret) is False
    
    def test_wrong_secret_rejected(self):
        """Test that wrong secret fails validation."""
        correct_secret = "correct_secret"
        wrong_secret = "wrong_secret"
        payload = b'{"action": "opened"}'
        
        signature = hmac.new(
            correct_secret.encode("utf-8"),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        assert verify_signature(payload, f"sha256={signature}", wrong_secret) is False
