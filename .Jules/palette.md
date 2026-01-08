# Palette's Journal

## 2025-01-28 - CLI UX Enhancements
**Learning:** Simulation scripts often neglect basic CLI UX like progress bars and output directory management, leading to console spam and "File not found" errors.
**Action:** Created `cli_utils` to standardise progress reporting (using `rich` or `tqdm`) and directory creation. This pattern should be applied to all future simulation scripts.
