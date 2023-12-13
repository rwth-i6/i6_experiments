from returnn_common import nn
from returnn_common.asr.specaugment import random_mask_v2, _mask_v2


def legacy_specaugment(
    x: nn.Tensor,
    *,
    spatial_dim: nn.Dim,
    feature_dim: nn.Dim = nn.NotSpecified,
    global_train_step_dependent: bool = True,
    only_on_train: bool = True,
    max_time_num: int = 1,
    max_time: int = 15,
    max_feature_num: int = 4,
    max_feature: int = 5,
) -> nn.Tensor:
    """
    specaugment with original configuration settings
    """
    if feature_dim is nn.NotSpecified:
        assert x.feature_dim
        feature_dim = x.feature_dim
    if global_train_step_dependent:
        step = nn.global_train_step()
        step1 = nn.where(step >= 2000, 1, 0)
    else:
        step1 = 1

    with nn.Cond(nn.train_flag() | (not only_on_train)) as cond:
        x_masked = x
        spatial_len = nn.dim_value(spatial_dim)
        # time mask
        x_masked = random_mask_v2(
            x_masked,
            mask_axis=spatial_dim,
            broadcast_axis=feature_dim,
            min_num=0,
            max_num=nn.maximum(spatial_len // int(1.0 / 0.7 * max_time), max_time_num) // (1 + step1),
            max_dims=max_time,
        )
        # feature mask
        x_masked = random_mask_v2(
            x_masked,
            mask_axis=feature_dim,
            broadcast_axis=spatial_dim,
            min_num=0,
            max_num=max_feature_num // (1 + step1),
            max_dims=max_feature,
        )
        cond.true = x_masked
        cond.false = x
    return cond.result
