import os
import sys

def ensure_directory(path: str):
    """
    Ensures that the directory exists.
    If it doesn't exist, it creates it.
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def progress_bar(iterable, total=None, desc="Processing"):
    """
    A progress bar that adapts to the available environment.
    Prioritizes 'rich', then 'tqdm', then a simple text fallback.

    Args:
        iterable: The iterable to wrap.
        total: Total number of items (optional, inferred if iterable has len).
        desc: Description to show.
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

    # Fallback text-based progress bar
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    def simple_generator():
        # Only print progress if stdout is a terminal
        is_tty = sys.stdout.isatty()

        for i, item in enumerate(iterable):
            yield item
            if is_tty:
                if total:
                    percent = (i + 1) / total * 100
                    sys.stdout.write(f"\r{desc}: {percent:.1f}% ({i+1}/{total})")
                else:
                    sys.stdout.write(f"\r{desc}: {i+1} items")
                sys.stdout.flush()

        if is_tty:
            print() # Newline at completion

    return simple_generator()
