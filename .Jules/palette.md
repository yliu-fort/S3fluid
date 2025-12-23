## 2024-05-22 - CLI Progress Indicators
**Learning:** Users running long simulation scripts (100+ steps) have no visibility into progress or ETA, leading to uncertainty. Printing raw step numbers floods the console and hides important warnings.
**Action:** Use `rich.progress.track` to wrap the main simulation loop. It provides an immediate visual upgrade (progress bar, percentage, ETA) with zero configuration and minimal code changes. This is a high-impact, low-effort "micro-UX" win for CLI tools.
