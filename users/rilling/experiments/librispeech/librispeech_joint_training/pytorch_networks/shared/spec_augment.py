from torchaudio.functional import mask_along_axis

def apply_spec_aug(input, num_repeat_time, max_dim_time, num_repeat_feat, max_dim_feat):
    """
    :param Tensor input: the input audio features (B,T,F)
    :param int num_repeat_time: number of repetitions to apply time mask
    :param int max_dim_time: number of columns to be masked on time dimension will be uniformly sampled from [0, mask_param]
    :param int num_repeat_feat: number of repetitions to apply feature mask
    :param int max_dim_feat: number of columns to be masked on feature dimension will be uniformly sampled from [0, mask_param]
    """
    for _ in range(num_repeat_time):
        input = mask_along_axis(input, mask_param=max_dim_time, mask_value=0.0, axis=1)

    for _ in range(num_repeat_feat):
        input = mask_along_axis(input, mask_param=max_dim_feat, mask_value=0.0, axis=2)
    return input