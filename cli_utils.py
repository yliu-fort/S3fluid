try:
    from rich.progress import track
    def progress_bar(sequence, description="Processing..."):
        """
        Wraps an iterable with a rich progress bar if available.
        Otherwise falls back to a simple print loop.
        """
        return track(sequence, description=description)
except ImportError:
    def progress_bar(sequence, description="Processing..."):
        """
        Fallback progress bar.
        """
        print(description)
        try:
            total = len(sequence)
            has_len = True
        except TypeError:
            has_len = False

        for i, item in enumerate(sequence):
            yield item
            if i % 10 == 0:
                if has_len:
                    print(f"Step {i}/{total}", end='\r')
                else:
                    print(f"Step {i}", end='\r')
        print()
