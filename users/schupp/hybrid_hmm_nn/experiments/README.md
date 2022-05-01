# Hybrid Conformer Experiments

Whole setups needs a lot of restructuring to make it clean. ( see new pipleline, and new experiments `experiments_rtc` )
But all the code for pipelines, network definition, config definition, etc. is there.

```
conformer_baseline.py, contains my ba conformer baseline code
```

As is this directory can be linked as the sisyphus config directory.
Configs are supposed to be run in separate tmux panes via `sis m <config-path>`

Also argument structure, defaults, imports have to be changed/re-done in a 'clean' way.
