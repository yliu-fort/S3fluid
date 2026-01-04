import os
import sys

def ensure_directory(path):
    """Ensures that the directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)

def progress_bar(iterable, desc="Processing", total=None):
    """
    Returns a progress bar iterator.
    Prioritizes rich.progress, then tqdm, then a simple text fallback.
    """
    try:
        from rich.progress import track
        return track(iterable, description=desc, total=total)
    except ImportError:
        pass

    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total)
    except ImportError:
        pass

    # Simple text fallback
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    print(f"{desc}...")
    for i, item in enumerate(iterable):
        yield item
        if total:
            percent = (i + 1) / total * 100
            if (i + 1) % (total // 10 if total >= 10 else 1) == 0:
                 print(f"{percent:.0f}% complete", end='\r', file=sys.stderr)
    print("")
