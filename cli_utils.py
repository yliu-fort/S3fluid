import os
import sys

def ensure_directory(path):
    """Ensures that the directory for the given path exists."""
    if not path: return
    # If path has extension, assume it's a file and use parent dir
    if os.path.splitext(path)[1]: path = os.path.dirname(path)
    if path and not os.path.exists(path): os.makedirs(path, exist_ok=True)

def progress_bar(iterable, total=None, desc="Processing"):
    """Returns a progress bar iterator. Prioritizes 'rich', then 'tqdm', then text."""
    try:
        from rich.progress import track
        return track(iterable, description=desc, total=total)
    except ImportError: pass
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc)
    except ImportError: pass

    if total is None:
        try: total = len(iterable)
        except (TypeError, AttributeError): total = None

    def generator():
        for i, item in enumerate(iterable):
            yield item
            current = i + 1
            if total:
                pct = (current * 100) // total
                bar = '=' * int(15 * current // total)
                sys.stdout.write(f"\r{desc}: [{bar:<15}] {pct}%")
            else:
                sys.stdout.write(f"\r{desc}: {current}")
            sys.stdout.flush()
        sys.stdout.write("\n")
    return generator()
