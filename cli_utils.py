import os
import sys

def ensure_directory(path):
    """
    Ensure that the directory for the given path exists.
    If the path is just a filename (no directory component), it does nothing.
    """
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

def progress_bar(iterable, total=None, desc="Processing"):
    """
    A progress bar that tries to use rich, then tqdm, then falls back to a simple text bar.

    Args:
        iterable: The iterable to iterate over.
        total: Total number of items (optional, used if iterable has no len()).
        desc: Description to show in the progress bar.
    """
    # Try rich first (prettier)
    try:
        from rich.progress import track
        # rich.progress.track automatically handles len(iterable) if available
        return track(iterable, description=desc, total=total)
    except ImportError:
        pass

    # Try tqdm second (standard in data science)
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc)
    except ImportError:
        pass

    # Fallback to simple text progress
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    # Generator to wrap the iterable
    def generator():
        for i, item in enumerate(iterable):
            yield item
            if total:
                percent = (i + 1) * 100 / total
                # Use carriage return \r to overwrite the line
                sys.stdout.write(f"\r{desc}: {percent:.1f}% ({i + 1}/{total})")
                sys.stdout.flush()
            else:
                sys.stdout.write(f"\r{desc}: {i + 1}")
                sys.stdout.flush()
        print() # Newline at the end to prevent next output from overwriting the bar

    return generator()
