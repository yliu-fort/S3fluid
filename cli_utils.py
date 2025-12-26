import sys
import shutil
import time

def progress_bar(iterable, desc="Progress", unit="it"):
    """
    A simple zero-dependency progress bar for CLI applications.

    Args:
        iterable: The iterable to wrap.
        desc (str): Description text to show before the bar.
        unit (str): Optional unit string (e.g., 'it', 'steps').

    Yields:
        Items from the iterable.
    """
    try:
        total = len(iterable)
    except TypeError:
        total = None

    # Get terminal width, default to 80 if not available
    cols = shutil.get_terminal_size((80, 20)).columns

    start_time = time.time()

    def print_status(iteration):
        if total:
            percent = ("{0:.1f}").format(100 * (iteration / float(total)))
            elapsed = time.time() - start_time
            elapsed_str = f"[{int(elapsed // 60):02d}:{int(elapsed % 60):02d}]"

            prefix = f"{desc}: "
            suffix = f" {percent}% {elapsed_str}"

            # Dynamic bar length
            bar_len = cols - len(prefix) - len(suffix) - 5
            bar_len = max(10, min(bar_len, 50))

            filled_len = int(bar_len * iteration // total)
            bar = '█' * filled_len + '-' * (bar_len - filled_len)

            sys.stdout.write(f'\r{prefix}[{bar}]{suffix}')
        else:
            sys.stdout.write(f'\r{desc}: {iteration} {unit}')

        sys.stdout.flush()

    print_status(0)
    for i, item in enumerate(iterable):
        yield item
        print_status(i + 1)

    sys.stdout.write('\n')
    sys.stdout.flush()
