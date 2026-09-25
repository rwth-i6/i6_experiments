# impl: w2vu env build, bounded numpy-uninstall loop (2026-09-25)

Status: DONE. Not committed; no job submitted; partial env `/work/asr4/hwu/conda/envs/w2vu` untouched
(only its pip sources were read).

## Cause (confirmed)
Build 4364831 hung at step 3 on `while "$PY" -m pip uninstall -y numpy ...; do :; done`. In pip 23.3.2,
`req_install.py:720-721` logs "Skipping numpy as it is not installed." and returns None, and
`commands/uninstall.py` still returns SUCCESS, so the loop never ends once numpy is gone. Reproduced
in a scratch venv (T0 below: the old loop is killed by a 20 s timeout).

## Change (one file)
`recipe/i6_experiments/users/wu/experiments/unsupervised_asr/env/build_w2vu_env.sh`, step 3 only:
the loop condition is now `pip show numpy` (pip 23.3.2 `commands/show.py`: `print_results` returns
False when nothing is found, and `run` returns ERROR = exit 1), capped at 10 uninstalls, `die` with the
remaining Version/Location if numpy is still listed at the cap. The `rm -rf` cleanup and the
`numpy==1.23.5` install after it are unchanged.

Material assumptions (for the reviewer):
- A failing `pip uninstall` (e.g. a dist-info without RECORD) ends the loop with `break`, which is the
  old loop's behaviour on a nonzero exit; the unchanged `rm -rf` then clears what is left.
- `PYTHONNOUSERSITE=1` on the loop's `pip show`/`pip uninstall`: pip otherwise sees
  `~/.local/lib/python3.9/site-packages` (the step-3 comment says so), and a user-site numpy, which
  pip refuses to uninstall as outside the env, would keep `pip show` at exit 0 and trip the cap on
  every build. No `~/.local/lib/python3.9` exists today, so this is only a guard. Drop it if unwanted.

## Diff
```diff
diff --git a/users/wu/experiments/unsupervised_asr/env/build_w2vu_env.sh b/users/wu/experiments/unsupervised_asr/env/build_w2vu_env.sh
index fd3f4c38e..a77945d0b 100755
--- a/users/wu/experiments/unsupervised_asr/env/build_w2vu_env.sh
+++ b/users/wu/experiments/unsupervised_asr/env/build_w2vu_env.sh
@@ -88,8 +88,17 @@ export PATH="$PREFIX/bin:$PATH"
 
 # fairseq 0.12.2 needs numpy<1.24 -- fairseq/data/data_utils.py:488 still uses `np.int`.
 # Uninstall in a loop: a previous --ignore-installed run can leave several dist-infos stacked, and
-# each `pip uninstall` removes only one.
-while "$PY" -m pip uninstall -y numpy >/dev/null 2>&1; do :; done
+# each `pip uninstall` removes only one. The loop runs while `pip show numpy` finds it (pip 23.3.2
+# exits 1 when absent), not on uninstall's status: uninstalling an absent package logs "Skipping ...
+# as it is not installed" and still exits 0, so looping on it never ends. A failing uninstall ends the
+# loop (the rm -rf below clears what it left); still present after 10 uninstalls is an error.
+# PYTHONNOUSERSITE=1: only this env's numpy counts (a user-site numpy is outside the env).
+n_uninst=0
+while PYTHONNOUSERSITE=1 "$PY" -m pip show numpy >/dev/null 2>&1; do
+    (( n_uninst < 10 )) || die "numpy still installed after 10 pip uninstalls: $(PYTHONNOUSERSITE=1 "$PY" -m pip show numpy 2>&1 | grep -E '^(Version|Location):' | tr '\n' ' ')"
+    PYTHONNOUSERSITE=1 "$PY" -m pip uninstall -y numpy >/dev/null 2>&1 || break
+    n_uninst=$((n_uninst + 1))
+done
 rm -rf "$PREFIX"/lib/python3.*/site-packages/numpy "$PREFIX"/lib/python3.*/site-packages/numpy-*.dist-info \
        "$PREFIX"/lib/python3.*/site-packages/numpy.libs
 "$PY" -m pip install --no-cache-dir "numpy==1.23.5"
```

## Scan for other unbounded loops / static exit statuses
The script has one shell loop (the one fixed); line ~129 `for n in (...)` is inside the Python heredoc
(two items). No other command's exit status is assumed to change between calls. Nothing else fixed.
Not the same defect, just noted: pip installs/downloads have no timeout (a network stall would hang
too), and line 51 refuses an existing prefix, so the rebuild needs the partial env removed first
(not done here, per the dispatch).

## Checks
- `bash -n build_w2vu_env.sh`: OK.
- Loop tests: the exact loop text was extracted from the script with
  `sed -n '/^n_uninst=0$/,/^done$/p'` and sourced by a harness (`set -euo pipefail`, same `die`)
  against a scratch venv (`/usr/bin/python3` 3.10.13, `pip==23.3.2`), in
  `/var/tmp/claude-2764/.../scratchpad/numpy_loop/`. Fake dist-infos (T4, T5) are METADATA+INSTALLER
  (+RECORD) dirs named `numpy-1.<i>.0.dist-info`. `pip show numpy` with numpy absent: rc=1.

Test output:
```
== T0 old loop, numpy absent (timeout 20 s)
rc=124 (124 = killed by timeout, i.e. hangs)
== T1 new loop, numpy absent
loop exited normally after n_uninst=0 uninstalls
rc=0 secs=1
== T2 one real copy
[notice] To update, run: python -m pip install --upgrade pip
dist-infos before: 1
loop exited normally after n_uninst=1 uninstalls
rc=0 dist-infos after: 0; import: ModuleNotFoundError: No module named 'numpy'
== T3 two stacked real copies (--ignore-installed)
[notice] To update, run: python -m pip install --upgrade pip
[notice] To update, run: python -m pip install --upgrade pip
dist-infos before: 2 numpy-1.23.5.dist-info numpy-1.26.4.dist-info 
loop exited normally after n_uninst=2 uninstalls
rc=0 dist-infos after: 0
== T4 cap: 11 stacked fake dist-infos
dist-infos before: 11
build_w2vu_env.sh: numpy still installed after 10 pip uninstalls: Version: 1.9.0 Location: /var/tmp/claude-2764/-u-hwu-setups-librispeech-960-2026-09-24-unsupervised/4186749c-0772-4a1a-a52d-b33a7b8fd5e1/scratchpad/numpy_loop/venv/lib/python3.10/site-packages 
rc=1 dist-infos after: 1
== T5 uninstall fails (no RECORD)
dist-infos before: 1
loop exited normally after n_uninst=0 uninstalls
rc=0 dist-infos after: 1 (the script's rm -rf follows)
```
T0: old loop hangs (the cause). T1: absent numpy exits at once, 0 uninstalls. T2: one real numpy
1.26.4 removed in 1 uninstall. T3: two stacked real dist-infos (1.23.5 plus `--ignore-installed`
1.26.4) removed in 2. T4: 11 stacked copies trip the cap, rc=1 with a message. T5: failing uninstall
breaks out, leaving the dist-info for the script's `rm -rf`.

Not verified: the full build script was not run (that is the executor's step).
