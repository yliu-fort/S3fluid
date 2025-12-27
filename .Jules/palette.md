## 2024-05-23 - [CLI Progress Indicators]
**Learning:** CLI applications that flood stdout with raw numbers cause "terminal fatigue" and make it hard to gauge time remaining. Users respond positively to structured progress bars (`rich.progress`) that show percentage and time estimates.
**Action:** Use `rich.progress` for any long-running loop (>1s). Fallback to structured text updates if dependencies are restricted, but always prefer visual indicators.
