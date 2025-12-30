import sys
import time
import os

def progress_bar(iterable, desc="Processing", total=None):
    """
    A progress bar that tries to use rich, then tqdm, then falls back to a simple text bar.
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
        return tqdm(iterable, desc=desc, total=total)
    except ImportError:
        pass

    # Fallback
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    print(f"{desc}...")

    start_time = time.time()
    for i, item in enumerate(iterable):
        yield item
        if total:
            # Update every 10% or at least every second, or at the end
            # This is a simple fallback, so we don't want to complicate it too much
            should_update = (i + 1) == total or (i % max(1, total // 10) == 0)
            if should_update:
                percent = (i + 1) / total * 100
                elapsed = time.time() - start_time
                sys.stdout.write(f"\r{desc}: {percent:.1f}% complete ({elapsed:.1f}s)")
                sys.stdout.flush()
    print("") # Newline at the end
