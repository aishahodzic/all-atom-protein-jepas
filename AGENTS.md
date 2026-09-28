# Temporary files

- Never create datasets, worktrees, agent dumps, caches, or other temporary artifacts in the system `/tmp` directory.
- Use `$HOME/tmp` (normally provided through `$TMPDIR`) for temporary work. Give each task a clearly named subdirectory so it can be inspected and cleaned safely.
- When a tool ignores `$TMPDIR`, pass it an explicit path under `$HOME/tmp`.
- Keep durable project outputs in their appropriate repository directories; `$HOME/tmp` is still disposable scratch space.
- Before moving or deleting existing temporary data, verify that no active process is using it.

# Figure exports

- Export plots and images as PNG only. Do not generate PDF copies of figure assets.
