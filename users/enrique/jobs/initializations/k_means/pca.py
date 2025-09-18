def compute_PCA_matrixes(model_def, pca_dim):
    from returnn.tensor import Tensor
    from returnn.config import get_global_config

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    extern_data_dict = config.typed_value("extern_data")
    data = Tensor(name=default_input_key, **extern_data_dict[default_input_key])
    targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
    assert targets.sparse_dim and targets.sparse_dim.vocab, f"no vocab for {targets}"

    model = model_def(epoch=0, in_dim=data.feature_dim, target_dim=targets.sparse_dim)
    print("YEEEEEEES")
    import sys

    sys.exit(0)
    return model
