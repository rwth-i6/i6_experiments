# WARNING

The `returnn_common` setup is still under construction, so please expect API and other code changes.

## Discussions

* How to integrate `returnn_common.nn` into Sisyphus pipelines:
  https://github.com/rwth-i6/i6_experiments/issues/63
* How to handle Sisyphus hashes: https://github.com/rwth-i6/returnn_common/issues/51


# Concepts

The user provides code using `returnn_common.nn` to build and construct:

- extern data (`extern_data`)
- the model
- training losses

In the final RETURNN config, the model and losses ends up as the network dict (`network`).
(Potentially different models or losses per epoch via `get_network`.)
Extern data ends up as the extern data dict (`extern_data`).

Here we provide helpers to realize this.

There are a few conceptual variations and aspects:

- Where is the `returnn_common.nn` code executed effectively?
  - In the Sisyphus manager?
  - In `ReturnnTrainingJob` `create_files` task?
  - In `ReturnnTrainingJob` `run` task, inside the RETURNN config?
- Where is the model code (using `returnn_common.nn`) located?
  As this is part of the recipe, it would be somewhere in `i6_experiments.users...`.
  Of course this would effectively make use of other building blocks
  and `returnn_common.nn` itself.
- When then `ReturnnTrainingJob` job is created, would it copy over the model code
  or run it directly from `i6_experiments.users...`?
