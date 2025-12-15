"""Tests for Telegram message safety (HTML escaping, plaintext fallback)."""

import pytest

from src.supervisor.telegram_notify import (
    escape_html,
    strip_html_tags,
    safe_truncate,
    TelegramStatusCard,
    MessageRegistry,
)


class TestHtmlEscaping:
    """Tests for HTML escaping."""
    
    def test_escapes_angle_brackets(self):
        """Test escaping of < and > characters."""
        text = "<script>alert('xss')</script>"
        result = escape_html(text)
        
        assert "<script>" not in result
        assert "&lt;script&gt;" in result
    
    def test_escapes_ampersand(self):
        """Test escaping of & character."""
        text = "foo & bar"
        result = escape_html(text)
        
        assert "& " not in result or "&amp;" in result
    
    def test_escapes_quotes(self):
        """Test escaping of quote characters."""
        text = 'He said "hello"'
        result = escape_html(text)
        
        assert "&quot;" in result
    
    def test_empty_string(self):
        """Test handling of empty string."""
        assert escape_html("") == ""
        assert escape_html(None) == ""
    
    def test_no_special_chars(self):
        """Test text without special characters."""
        text = "Normal text here"
        assert escape_html(text) == text


class TestStripHtmlTags:
    """Tests for HTML tag stripping."""
    
    def test_strips_bold_tags(self):
        """Test stripping of <b> tags."""
        text = "<b>Bold text</b>"
        result = strip_html_tags(text)
        
        assert "<b>" not in result
        assert "</b>" not in result
        assert "Bold text" in result
    
    def test_strips_multiple_tags(self):
        """Test stripping of multiple different tags."""
        text = "<b>Bold</b> and <i>italic</i> and <code>code</code>"
        result = strip_html_tags(text)
        
        assert "<" not in result
        assert ">" not in result
        assert "Bold" in result
        assert "italic" in result
        assert "code" in result
    
    def test_converts_html_entities(self):
        """Test conversion of HTML entities back to characters."""
        text = "&lt;script&gt; &amp; stuff"
        result = strip_html_tags(text)
        
        assert "<script>" in result
        assert "&" in result


class TestSafeTruncate:
    """Tests for safe truncation."""
    
    def test_truncates_long_text(self):
        """Test truncation of text exceeding max_chars."""
        text = "A" * 100
        result = safe_truncate(text, 50)
        
        assert len(result) == 50
        assert result.endswith("...")
    
    def test_preserves_short_text(self):
        """Test that short text is not modified."""
        text = "Short text"
        result = safe_truncate(text, 50)
        
        assert result == text
    
    def test_handles_empty_string(self):
        """Test handling of empty string."""
        assert safe_truncate("", 50) == ""
    
    def test_handles_none(self):
        """Test handling of None."""
        assert safe_truncate(None, 50) == ""
    
    def test_custom_suffix(self):
        """Test custom truncation suffix."""
        text = "A" * 100
        result = safe_truncate(text, 50, suffix="[cut]")
        
        assert result.endswith("[cut]")
        assert len(result) == 50


class TestTelegramStatusCard:
    """Tests for TelegramStatusCard."""
    
    def test_card_text_escapes_dynamic_content(self):
        """Test that card text escapes dynamic content."""
        from unittest.mock import MagicMock
        
        settings = MagicMock()
        settings.telegram_enabled = True
        settings.telegram_bot_token = "token"
        settings.telegram_chat_id = "123"
        settings.telegram_max_chars = 4000
        settings.telegram_debounce_seconds = 0
        
        registry = MessageRegistry()
        card = TelegramStatusCard(
            settings=settings,
            repo="owner/repo",
            pr_number=42,
            message_registry=registry,
        )
        
        card.pr_title = "<script>alert('xss')</script>"
        card.current_phase = "CHECKS"
        
        html_text = card._build_card_text(use_html=True)
        
        assert "<script>" not in html_text
        assert "&lt;script&gt;" in html_text
    
    def test_card_plaintext_no_html(self):
        """Test that plaintext mode contains no HTML tags."""
        from unittest.mock import MagicMock
        
        settings = MagicMock()
        settings.telegram_enabled = True
        settings.telegram_bot_token = "token"
        settings.telegram_chat_id = "123"
        settings.telegram_max_chars = 4000
        settings.telegram_debounce_seconds = 0
        
        registry = MessageRegistry()
        card = TelegramStatusCard(
            settings=settings,
            repo="owner/repo",
            pr_number=42,
            message_registry=registry,
        )
        
        card.pr_title = "Test PR"
        card.current_phase = "DONE"
        
        plain_text = card._build_card_text(use_html=False)
        
        assert "<b>" not in plain_text
        assert "</b>" not in plain_text
        assert "<a href" not in plain_text
        assert "<code>" not in plain_text


class TestMessageRegistry:
    """Tests for MessageRegistry."""
    
    def test_set_and_get_message_id(self):
        """Test setting and getting message ID."""
        registry = MessageRegistry()
        
        registry.set_message_id(("owner/repo", 42), 12345)
        
        assert registry.get_message_id(("owner/repo", 42)) == 12345
    
    def test_get_nonexistent_returns_none(self):
        """Test getting nonexistent key returns None."""
        registry = MessageRegistry()
        
        assert registry.get_message_id(("owner/repo", 999)) is None
    
    def test_clear_message_id(self):
        """Test clearing message ID."""
        registry = MessageRegistry()
        
        registry.set_message_id(("owner/repo", 42), 12345)
        registry.clear(("owner/repo", 42))
        
        assert registry.get_message_id(("owner/repo", 42)) is None
    
    def test_export_and_load(self):
        """Test export and load functionality."""
        registry1 = MessageRegistry()
        registry1.set_message_id(("owner/repo", 42), 12345)
        registry1.set_message_id(("other/repo", 10), 67890)
        
        exported = registry1.export()
        
        registry2 = MessageRegistry()
        registry2.load(exported)
        
        assert registry2.get_message_id(("owner/repo", 42)) == 12345
        assert registry2.get_message_id(("other/repo", 10)) == 67890
