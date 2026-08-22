# exp_logs -- version-control workflow

<!-- Canonical home of the version-control rules for plan and log files; CLAUDE.md points here. -->

Task-owned PLAN*.md and phase logs live here as tracked files, one directory per campaign
(stable campaign prefix as the directory name, e.g. SAE/). Workspace paths elsewhere are
relative symlinks into this tree; keep editing through the workspace path.

Commit cadence: after each completed round -- the planner after a verification round, the
implementer together with the tested in-scope code of that round.

Staging: explicit paths only, never `git add -A` -- other people and sessions edit these
checkouts live. Inspect `git status` and the diff first; rebase before push.
