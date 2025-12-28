try:
    from rich.progress import track
    # Wrapper to match the usage if we had a custom class, but rich.track is already great
    def progress_bar(iterable, description="Processing..."):
        return track(iterable, description=description)
except ImportError:
    import sys
    import time

    # Fallback if rich is not available (though it should be)
    class ProgressBar:
        def __init__(self, iterable, prefix='Progress:', suffix='Complete', length=50, fill='█'):
            self.iterable = iterable
            self.prefix = prefix
            self.suffix = suffix
            self.length = length
            self.fill = fill
            self.total = len(iterable)
            self.start_time = time.time()

        def __iter__(self):
            for i, item in enumerate(self.iterable):
                yield item
                self.print_progress(i + 1)
            print()

        def print_progress(self, iteration):
            if self.total == 0:
                percent = "100.0"
                filledLength = self.length
            else:
                percent = ("{0:.1f}").format(100 * (iteration / float(self.total)))
                filledLength = int(self.length * iteration // self.total)
            bar = self.fill * filledLength + '-' * (self.length - filledLength)

            elapsed = time.time() - self.start_time
            if iteration > 0:
                rate = iteration / elapsed
                remaining = (self.total - iteration) / rate
                eta = f"ETA: {int(remaining)}s"
            else:
                eta = ""

            sys.stdout.write(f'\r{self.prefix} |{bar}| {percent}% {self.suffix} [{eta}]')
            sys.stdout.flush()

    def progress_bar(iterable, description="Processing..."):
        return ProgressBar(iterable, prefix=description)
