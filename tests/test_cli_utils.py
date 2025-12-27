
import pytest
from cli_utils import progress_bar

def test_progress_bar_fallback():
    # Test that the progress bar yields all items
    items = list(range(5))
    output = list(progress_bar(items, description="Test"))
    assert output == items

def test_progress_bar_is_generator():
    items = [1, 2, 3]
    pb = progress_bar(items)
    # Ensure it's iterable
    assert iter(pb) is pb or hasattr(pb, '__iter__')
