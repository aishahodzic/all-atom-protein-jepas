# Temporary files

- Never create datasets, worktrees, session dumps, caches, or other temporary artifacts in the system `/tmp` directory.
- Use `$HOME/tmp` for all temporary work. `CLAUDE_CODE_TMPDIR` and `TMPDIR` should both point there.
- Give each task a clearly named subdirectory under `$HOME/tmp`, and pass an explicit path when a tool ignores the environment variables.
- Keep durable project outputs in their appropriate repository directories; `$HOME/tmp` is still disposable scratch space.
- Before moving or deleting existing temporary data, verify that no active process is using it.
