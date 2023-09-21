import torch 

def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor, device: torch.DeviceObjType) -> torch.Tensor:
    """
    mask a tensor with a "positive" mask (boolean true means position is used)

    This function is traceable.

    :param tensor: [B,T,....]
    :param seq_len: [B]
    :return: [B,T]
    """
    r = torch.arange(tensor.shape[2], device=device)  # [T]
    seq_mask = torch.less(r[None, :], seq_len[:, None])  # broadcast to [B,T]
    return seq_mask