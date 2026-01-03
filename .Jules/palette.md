## 2024-05-23 - Progressive Enhancement for CLI Tools
**Learning:** CLI tools often lack visual feedback or spam stdout, making them hard to use. Users may not have fancy libraries like `rich` installed.
**Action:** Implement "progressive enhancement" for CLI outputs: check for `rich`, then `tqdm`, then fallback to a simple text-based progress bar. This ensures a delightful experience where possible, but a functional one everywhere.
