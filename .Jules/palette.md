## 2024-05-23 - CLI Progress Indicators
**Learning:** Simulation scripts with long-running loops often print step numbers to stdout, causing console spam and offering no ETA. This is a poor developer experience.
**Action:** Wrap loops in a `progress_bar` utility (using `rich` or `tqdm` with a fallback) to provide visual feedback and time estimates without cluttering the log.

## 2024-05-23 - Output Directory Handling
**Learning:** Scripts writing to subdirectories (e.g., `results/`) often fail if the directory doesn't exist, frustrating users who expect "it to just work".
**Action:** Always use `ensure_directory(path)` before opening a file for writing to automatically create the necessary folder structure.
