# exec_prepro_devother_launch_2026-09-21

BLOCKED

Config: config/sae_4a_prepro_devother.py contains only:
```
from speech_llm...configs.config_sae_4a_prepro_pack_v1 import py_dev_other
run = py_dev_other
```
It assigns `py_dev_other` to a module-level name `run` but never calls it. Sisyphus's config loader
looks for and calls a function literally named `py`; the manager log shows:
`WARNING: No function named 'py' found in module 'config.sae_4a_prepro_devother'`
followed immediately by `Finished updating job states` (nothing was registered) and then the
interactive prompt `All calculations are done, print verbose overview (v), update outputs and
alias (u), cancel (c)?`, which died with `EOFError: EOF when reading a line` because the manager
was started under nohup with no stdin. Per instructions I did not answer this prompt.

Result: no job was registered/submitted. No job dir for hash 0IOLr6hZnYWj exists under work/.
No manager process is alive (confirmed via ps).

Manager log: log/sae_4a_prepro_devother.manager.20260920T220210Z.log
Manager pid: none (process exited)
Job dir: none created
Slurm id: none

Blocker: the shim needs to actually call py_dev_other(), e.g. `def py(): return py_dev_other()`
or `py_dev_other()` at module scope, so its registered outputs reach the graph. This is a config
content issue, not something I'm authorized to edit — reporting per scope.

No other managers (4004154, 2945974, 1355841) or running jobs were touched.
