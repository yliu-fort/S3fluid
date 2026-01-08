import os
import sys

def ensure_directory(path):
    """
    Ensures that the directory for the given file path exists.
    If 'path' is a directory (ends with separator), it creates it.
    If 'path' is a file, it creates the parent directory.
    """
    if not path:
        return

    dirname = os.path.dirname(path) if not path.endswith(os.sep) else path
    if dirname and not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
            print(f"Created directory: {dirname}")
        except OSError as e:
            print(f"Error creating directory {dirname}: {e}")

def progress_bar(iterable, total=None, description="Processing"):
    """
    Returns a progress bar iterator.
    Prioritizes 'rich.progress', then 'tqdm', then a simple text fallback.
    """

    # Try rich
    try:
        from rich.progress import track
        return track(iterable, description=description, total=total)
    except ImportError:
        pass

    # Try tqdm
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=description)
    except ImportError:
        pass

    # Fallback
    print(f"{description}...")
    return iterable
