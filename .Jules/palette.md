## 2024-12-25 - CLI Progress Indicators
**Learning:** UX applies to CLI tools too. Long-running simulations need progress feedback.
**Action:** When external dependencies like `rich` are not allowed, implementing a simple, zero-dependency text-based progress bar (using `\r` and `sys.stdout`) is a lightweight and effective way to improve the developer experience without bloating the project.
