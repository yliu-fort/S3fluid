import os

# Try to import rich, then tqdm, then fallback
try:
    from rich.progress import track
except ImportError:
    track = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def ensure_directory(path: str):
    """
    Ensures that the directory exists. If it doesn't, it creates it.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def progress_bar(iterable, description="Processing..."):
    """
    Wraps an iterable with a progress bar.
    Prioritizes 'rich', then 'tqdm', then falls back to a simple pass-through.
    """
    if track:
        return track(iterable, description=description)
    elif tqdm:
        return tqdm(iterable, desc=description)
    else:
        print(description)
        return iterable
