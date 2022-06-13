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
