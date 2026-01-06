import sys
import os

def ensure_directory(path):
    """Ensures that the directory for the given path exists."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory)

def progress_bar(iterable, total=None, desc="Processing"):
    """
    A wrapper that tries to use rich, then tqdm, then falls back to a simple text bar.
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
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    print(f"{desc}...")

    def simple_generator():
        for i, item in enumerate(iterable):
            yield item
            if total:
                percent = (i + 1) * 100 // total
                sys.stdout.write(f"\r[{'=' * (percent // 5)}{' ' * (20 - percent // 5)}] {percent}%")
                sys.stdout.flush()
        print() # Newline at end

    return simple_generator()
