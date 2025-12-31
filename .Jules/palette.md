## 2024-05-23 - [CLI Progress Indicators]
**Learning:** Python scientific scripts often use `print(t)` inside loops, which causes console spam.
**Action:** Replace `print(t)` with `rich.progress` (or `tqdm`) for a clean, informative progress bar. Also ensure output directories exist to prevent `FileNotFoundError`.
