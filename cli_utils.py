import sys
import os
from contextlib import contextmanager

def ensure_directory(path):
    """
    Ensures that the directory for the given file path exists.
    If path is a directory, ensures it exists.
    """
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
            print(f"Created directory: {dirname}")
        except OSError as e:
            print(f"Error creating directory {dirname}: {e}")

try:
    from rich.progress import track
    def progress_bar(sequence, description="Processing..."):
        return track(sequence, description=description)

except ImportError:
    try:
        from tqdm import tqdm
        def progress_bar(sequence, description="Processing..."):
            return tqdm(sequence, desc=description)
    except ImportError:
        def progress_bar(sequence, description="Processing..."):
            """
            Simple text-based progress bar fallback.
            """
            total = len(sequence)
            for i, item in enumerate(sequence):
                yield item
                percent = 100 * (i + 1) / total
                bar_length = 30
                filled_length = int(bar_length * (i + 1) // total)
                bar = '=' * filled_length + '-' * (bar_length - filled_length)
                sys.stdout.write(f'\r{description} |{bar}| {percent:.1f}%')
                sys.stdout.flush()
            sys.stdout.write('\n')
