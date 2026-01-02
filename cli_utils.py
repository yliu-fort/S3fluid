import os
import sys

def ensure_directory(path):
    """
    Ensures that the directory exists. If it doesn't, creates it.
    """
    if path and not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            # Handle race condition or permission error slightly gracefully
            if not os.path.exists(path):
                print(f"Error creating directory {path}: {e}", file=sys.stderr)

def progress_bar(iterable, total=None, desc="Processing"):
    """
    A progress bar that tries to use rich, then tqdm, then falls back to a simple text counter.

    Args:
        iterable: The iterable to wrap.
        total: Optional total number of items (if not len(iterable)).
        desc: Description text to show.
    """
    # Try rich
    try:
        from rich.progress import track
        return track(iterable, description=desc, total=total)
    except ImportError:
        pass

    # Try tqdm
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc)
    except ImportError:
        pass

    # Fallback
    def simple_generator():
        n = total
        if n is None:
            try:
                n = len(iterable)
            except TypeError:
                n = None

        # Determine width for formatting
        width = len(str(n)) if n else 0

        for i, item in enumerate(iterable):
            yield item
            # Update progress
            current = i + 1
            if n:
                percent = current * 100 / n
                # Use \r to overwrite line
                sys.stdout.write(f"\r{desc}: {current:>{width}}/{n} ({percent:5.1f}%)")
            else:
                sys.stdout.write(f"\r{desc}: {current}")
            sys.stdout.flush()
        print() # Newline at end

    return simple_generator()
