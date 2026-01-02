## 2024-05-22 - Missing CLI Utilities
**Learning:** The repository lacked shared CLI utilities for common tasks like directory creation and progress reporting, leading to code duplication and poor UX (console spam, crashes).
**Action:** Standardize on a `cli_utils.py` module that provides robust fallbacks for these common needs, ensuring consistent behavior across all scripts.
