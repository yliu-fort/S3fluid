import sys
import time

def progress_bar(iterable, total=None, desc="Progress"):
    """
    A simple progress bar wrapper.

    It attempts to use `rich.progress.track` or `tqdm` if available.
    Otherwise, it falls back to a simple text-based progress indicator
    that updates on the same line to avoid console spam.

    Args:
        iterable (iterable): The iterable to iterate over.
        total (int, optional): The total number of items. If None, attempts to use len(iterable).
        desc (str, optional): Description to display.

    Returns:
        generator: Yields items from the iterable.
    """
    # Try rich (best UX)
    try:
        from rich.progress import track
        return track(iterable, description=desc, total=total)
    except ImportError:
        pass

    # Try tqdm (standard UX)
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc)
    except ImportError:
        pass

    # Fallback (custom UX)
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    def generator():
        start_time = time.time()
        # Print initial line
        sys.stdout.write(f"{desc}: [Initializing...]\r")
        sys.stdout.flush()

        for i, item in enumerate(iterable):
            yield item

            current = i + 1
            if total:
                percent = current / total * 100
                bar_length = 20
                filled_length = int(bar_length * current // total)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)

                # Update line
                sys.stdout.write(f"\r{desc}: |{bar}| {current}/{total} ({percent:.1f}%)")
            else:
                sys.stdout.write(f"\r{desc}: {current} steps")

            sys.stdout.flush()

        # New line after completion
        sys.stdout.write("\n")
        sys.stdout.flush()

    return generator()
