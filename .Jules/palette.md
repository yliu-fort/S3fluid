## 2024-02-12 - [CLI Progress Bars]
**Learning:** Python CLI tools benefit significantly from visual progress indicators. The `rich` library is a standard and powerful tool for this, and it's often already present as a transitive dependency (e.g. via `meshio`), allowing for "free" UX upgrades without adding new direct dependencies.
**Action:** When working on CLI scripts, check if `rich` or `tqdm` are available in the environment to replace raw print loops with progress bars.
