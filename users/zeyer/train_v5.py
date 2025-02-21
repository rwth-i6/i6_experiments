"""
Train version 5, evolved from train_v4.py

Currently, this does not exist yet.
Here I just want to collect some ideas for the next version.

TODO

- We don't really need the model_def, train_def, etc, indirection.
  Instead, we can directly set get_model, train_step, in the config.
  get_model can have any arguments via functools.partial.
  We don't need to use the global RETURNN config for any model settings.
  So ModelDefWithCfg is not really needed anymore.
"""
