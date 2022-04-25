# Generate conformer and config code using returnn_common
from recipe.returnn_common import nn
from recipe.returnn_common.nn import hybrid_hmm



def make_conformer(
    in_dim_size = 50,
    out_dim_size = 12001,
    feature_dim_conformer = 256,
    train = True
):
    time_dim = nn.SpatialDim("time") # Just the input time ( should be the same on target)
    input_dim = nn.FeatureDim("input", in_dim_size) # 50 dim gammatone
    # -> (T, 50)
    conformer_feature = nn.FeatureDim("residual_dim", feature_dim_conformer)

    output_dim = nn.FeatureDim("output", out_dim_size)
    # (T, 12001)

    targets = nn.get_extern_data(nn.Data("classes", dim_tags=[nn.batch_dim, time_dim], sparse_dim=output_dim))
    data    = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, input_dim]))

    class Model(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.encoder = nn.ConformerEncoder(
                out_dim=conformer_feature, ff_dim=nn.FeatureDim("ff", 17),
                num_heads=2, num_layers=2
            )

            self.linear = nn.Linear(output_dim)
            
        def __call__(self, x: nn.Tensor) -> nn.Tensor:
            x = self.encoder(x, in_spatial_dim=time_dim)
            x = self.linear(x)
            return x


    conformer = Model()

    out = conformer(data)

    loss = nn.cross_entropy(target=targets, estimated=out, estimated_type="logits")

    loss.mark_as_default_output()

    model_py_code_str = nn.get_returnn_config().get_complete_py_code_str(conformer)

    return model_py_code_str

# Be aware changing this code changes the exeriment hash
# TODO: not use this code, use the imported modification of nn.ConformerEncoder
# Then when I introduce a change to nn.ConformerEncoder do it in a separate file, only change the import 
def make_conformer_old2(
    in_dim_size = 50,
    out_dim_size = 12001,
    train = True
):
    input_dim = nn.FeatureDim("input", in_dim_size)
    time_dim = nn.SpatialDim("time")
    targets_time_dim = nn.SpatialDim("targets-time")
    output_dim = nn.FeatureDim("output", out_dim_size)

    class Model(nn.ConformerEncoder):
        def __init__(self):
            super(Model, self).__init__(
            # Smaller...
            num_layers=4, num_heads=4, out_dim=nn.FeatureDim("conformer", 256), ff_dim=nn.FeatureDim("ff", 512)),

        def __call__(self, x: nn.Tensor, *, in_spatial_dim: nn.Dim, **kwargs) -> nn.Tensor:
            x, out_spatial_dim = super(Model, self).__call__(x, in_spatial_dim=in_spatial_dim, **kwargs)

            assert isinstance(out_spatial_dim, nn.Dim)

            if out_spatial_dim != in_spatial_dim:
                out_spatial_dim.declare_same_as(nn.SpatialDim("downsampled-time"))
            #x = nn.log_softmax(x, axis=output_dim)
            return x, out_spatial_dim

    # TODO specaug
    conformer_encoder = Model()

    targets = nn.get_extern_data(nn.Data("classes", dim_tags=[nn.batch_dim, targets_time_dim], sparse_dim=output_dim))
    data    = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, input_dim]))


    output = nn.Linear(output_dim)

    out, out_spatial = conformer_encoder(
        data,
        in_spatial_dim=time_dim
    )

    final_out = output(out)

    final_out.mark_as_default_output()

    ce_loss = nn.cross_entropy(target=targets, estimated=final_out, estimated_type="logits")
    #ce_loss = nn.sparse_softmax_cross_entropy_with_logits(logits=logits, targets=targets, axis=output_dim)
    #ce_loss.mark_as_default_output()
    ce_loss.mark_as_loss()

    model_py_code_str = nn.get_returnn_config().get_complete_py_code_str(conformer_encoder)
    return model_py_code_str

def make_conformer_old(
    in_dim_size = 50,
    out_dim_size = 12001
):
    input_dim = nn.FeatureDim("input", in_dim_size)
    time_dim = nn.SpatialDim("time")
    targets_time_dim = nn.SpatialDim("targets-time")
    output_dim = nn.FeatureDim("output", out_dim_size)

    class Model(nn.ConformerEncoder):
        def __init__(self):
            super(Model, self).__init__(
            # Smaller...
            num_layers=4, num_heads=4, out_dim=nn.FeatureDim("conformer", 256), ff_dim=nn.FeatureDim("ff", 512)),
            self.output = nn.Linear(output_dim)  # +1 for blank

        def __call__(self, x: nn.Tensor, *, in_spatial_dim: nn.Dim, **kwargs) -> nn.Tensor:
            x, out_spatial_dim = super(Model, self).__call__(x, in_spatial_dim=in_spatial_dim, **kwargs)
            assert isinstance(out_spatial_dim, nn.Dim)
            if out_spatial_dim != in_spatial_dim:
                out_spatial_dim.declare_same_as(nn.SpatialDim("downsampled-time"))
            x = self.output(x)
            return x

    # TODO specaug
    conformer_encoder = Model()
    logits = conformer_encoder(
        nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, input_dim])),
        in_spatial_dim=time_dim)

#    loss = nn.ctc_loss(
#        logits=logits,
#        targets=nn.get_extern_data(nn.Data("classes", dim_tags=[nn.batch_dim, targets_time_dim], sparse_dim=output_dim)))

    hybrid = hybrid_hmm.HybridHMM(
        encoder=Model,
        out_dim = output_dim
    )

    logits2 = hybrid(
        nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, input_dim])),
        targets=nn.get_extern_data(nn.Data("classes", dim_tags=[nn.batch_dim, targets_time_dim], sparse_dim=output_dim))
    )

    model_py_code_str = nn.get_returnn_config().get_complete_py_code_str(hybrid)
    return model_py_code_str