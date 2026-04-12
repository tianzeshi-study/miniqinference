import miniqwen_python as mq
import pytest

def test_import():
    """Test that the module can be imported."""
    assert mq is not None

def test_version():
    """Test that the version is available (if defined)."""
    # Assuming version is dynamic or set in pyproject.toml
    pass
