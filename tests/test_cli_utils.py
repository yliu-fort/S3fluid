import io
import sys
import time
from cli_utils import progress_bar

def test_progress_bar_output():
    # Capture stdout
    captured_output = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = captured_output

    try:
        # Run progress bar
        items = list(range(10))
        for _ in progress_bar(items, desc="Test", unit="x"):
            time.sleep(0.01) # Small delay to ensure time diff if needed, though mocking time is better usually
            pass

        output = captured_output.getvalue()

        # Check for key elements
        assert "Test:" in output
        assert "100.0%" in output
        assert "\r" in output
        assert "[" in output and "]" in output # For bar and time
        # Check for time format like [00:00]
        assert ":00]" in output or ":01]" in output

        # Ensure it yielded all items
        result_list = []
        # Clear buffer for next test
        captured_output.truncate(0)
        captured_output.seek(0)

        for i in progress_bar(items, desc="Test"):
            result_list.append(i)
        assert result_list == items

    finally:
        # Restore stdout
        sys.stdout = original_stdout

if __name__ == "__main__":
    test_progress_bar_output()
    print("cli_utils tests passed!")
