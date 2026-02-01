import pytest

from src.execution_ledger import make_label


def test_make_label_format_and_invariants():
    intent_id = "a1b2c3d4e5f6a7b8c9d0" + "1234567890abcdef"  # 32+ chars fine
    assert make_label(intent_id, "CLOSE", 0) == "cc|a1b2c3d4e5f6a7b8c9d0|c|a0"
    assert make_label(intent_id, "OPEN", 2) == "cc|a1b2c3d4e5f6a7b8c9d0|o|a2"

    lbl = make_label(intent_id, "OPEN", 999)
    assert len(lbl) <= 64


def test_make_label_rejects_bad_chars():
    with pytest.raises(ValueError):
        make_label("NOTASCII!!!!", "CLOSE", 0)
