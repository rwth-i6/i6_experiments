# Generate conformer and config code using returnn_common
from recipe.returnn_common import nn

def make_conformer(
    in_dim_size = 50,
    out_dim_size = 12000 # ( without blank label )
):
    input_dim = nn.FeatureDim("input", 50)
    time_dim = nn.SpatialDim("time")
    targets_time_dim = nn.SpatialDim("targets-time")
    output_dim = nn.FeatureDim("output", 12000)

    class Model(nn.ConformerEncoder):
        def __init__(self):
            super(Model, self).__init__(
            # Smaller...
            num_layers=4, num_heads=4, out_dim=nn.FeatureDim("conformer", 256), ff_dim=nn.FeatureDim("ff", 512)),
            self.output = nn.Linear(output_dim + 1)  # +1 for blank

        def __call__(self, x: nn.Tensor, *, in_spatial_dim: nn.Dim, **kwargs) -> nn.Tensor:
            x, out_spatial_dim = super(Model, self).__call__(x, in_spatial_dim=in_spatial_dim, **kwargs)
            assert isinstance(out_spatial_dim, nn.Dim)
            if out_spatial_dim != in_spatial_dim:
                out_spatial_dim.declare_same_as(nn.SpatialDim("downsampled-time"))
            x = self.output(x)
            return x

    # TODO specaug
    model = Model()
    logits = model(
    nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, input_dim])),
    in_spatial_dim=time_dim)
    loss = nn.ctc_loss(
    logits=logits,
    targets=nn.get_extern_data(nn.Data("classes", dim_tags=[nn.batch_dim, targets_time_dim], sparse_dim=output_dim)))
    loss.mark_as_loss()

    model_py_code_str = nn.get_returnn_config().get_complete_py_code_str(model)
    return model_py_code_str