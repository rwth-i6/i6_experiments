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
Dim tags are handled, i.e. defined in the RETURNN config,
and then used in extern data and in the model.

Here we provide helpers to realize this.

There are a few conceptual variations and aspects:

- Where is the `returnn_common.nn` code executed effectively?
  - In the Sisyphus manager?
    This could maybe be slow.
    Also, this means to rerun the manager each time the user changes the model code. 
  - In `ReturnnTrainingJob` `create_files` task?
    This means to rerun this task each time the user changes the model code.
  - In `ReturnnTrainingJob` `run` task, inside the RETURNN config?
- Where is the model code (using `returnn_common.nn`) located?
  As this is part of the recipe, it would be somewhere in `i6_experiments.users...`.
  Of course this would effectively make use of other building blocks
  and `returnn_common.nn` itself.
- When then `ReturnnTrainingJob` job is created, would it copy over the model code
  or run it directly from `i6_experiments.users...`?
- How to get from the Sisyphus dataset objects to extern data (`extern_data`)?
- How is the Sisyphus hash defined?
  - The net dict (as it used to be) would imply
    that the model code is executed in the Sisyphus manager.
    This hash is very fragile.
    Many changes to `returnn_common.nn` would probably introduce a new hash
    because many (irrelevant) details of the net dict construction logic are still changing,
    and it is unlikely that this will ever become stable.
  - Something more custom.

The helpers here in this Python package potentially allow to realize all of these variants.


# Examples

```python
from i6_core.returnn.training import ReturnnTrainingJob
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.common.setups.returnn_common.serialization import Network

model_def = Network("i6_experiments.users.zeyer.model.my_best_model_123")
config = ReturnnConfig(..., python_epilog=[model_def, ...])
train_job = ReturnnTrainingJob(config, ...)
```
