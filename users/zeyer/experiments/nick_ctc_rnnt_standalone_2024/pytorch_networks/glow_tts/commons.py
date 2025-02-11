import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# import mpmath
from scipy import special as sp

from IPython import embed


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def mle_loss(z, m, logs, logdet, mask):
    normalizer = torch.sum(torch.ones_like(z) * mask)
    l = torch.sum(logs) + 0.5 * torch.sum(
        torch.exp(-2 * logs) * ((z - m) ** 2)
    )  # neg normal likelihood w/o the constant term
    l = l - torch.sum(logdet)  # log jacobian determinant
    l = l / normalizer  # averaging across batch, channel and time axes
    l = l + 0.5 * math.log(2 * math.pi)  # add the remaining constant term
    return l


def vMF_ml_loss(z, m, logk, logdet, mask, d, log_C_k, kappa):
    normalizer = torch.sum(torch.ones_like(z) * mask)
    m_norm = nn.functional.normalize(m, dim=1)
    z_norm = nn.functional.normalize(z, dim=1)
    l = torch.matmul(m_norm.transpose(1,2), z_norm)
    l = kappa * l
    l = log_C_k + torch.sum(l)
    l = -1. * l
    l = l - torch.sum(logdet)
    l = l / normalizer
    return l

def length_loss(z, m, mask):
    normalizer = torch.sum(torch.ones_like(z) * mask)
    l = ((z - nn.functional.normalize(z, dim=1)) ** 2).sum()
    l = l / normalizer
    return l


def duration_loss(logw, logw_, lengths):
    l = torch.sum((logw - logw_) ** 2) / torch.sum(lengths)
    return l


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def shift_1d(x):
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def maximum_path(value, mask, max_neg_val=-np.inf):
    """Numpy-friendly version. It's about 4 times faster than torch version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    value = value * mask

    device = value.device
    dtype = value.dtype
    value = value.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy().astype(np.bool)

    b, t_x, t_y = value.shape
    direction = np.zeros(value.shape, dtype=np.int64)
    v = np.zeros((b, t_x), dtype=np.float32)
    x_range = np.arange(t_x, dtype=np.float32).reshape(1, -1)
    for j in range(t_y):
        v0 = np.pad(v, [[0, 0], [1, 0]], mode="constant", constant_values=max_neg_val)[:, :-1]
        v1 = v
        max_mask = v1 >= v0
        v_max = np.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask

        index_mask = x_range <= j
        v = np.where(index_mask, v_max + value[:, :, j], max_neg_val)
    direction = np.where(mask, direction, 1)

    path = np.zeros(value.shape, dtype=np.float32)
    index = mask[:, :, 0].sum(1).astype(np.int64) - 1
    index_range = np.arange(b)
    for j in reversed(range(t_y)):
        path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1
    path = path * mask.astype(np.float32)
    path = torch.from_numpy(path).to(device=device, dtype=dtype)
    return path




class vMFLogPartition(torch.autograd.Function):
    
    '''
    Evaluates log C_d(kappa) for vMF density
    Allows autograd wrt kappa

    Copied from https://github.com/minyoungkim21/vmf-lib/blob/main/models.py
    '''
    
    besseli = np.vectorize(sp.iv)
    log = np.log
    nhlog2pi = -0.5 * np.log(2*np.pi)
    
    @staticmethod
    def forward(ctx, *args):
        
        '''
        Args:
            args[0] = d; scalar (> 0)
            args[1] = kappa; (> 0) torch tensor of any shape
            
        Returns:
            logC = log C_d(kappa); torch tensor of the same shape as kappa
        '''
        
        d = args[0]
        kappa = args[1]
        
        s = 0.5*d - 1
        
        # log I_s(kappa)
        mp_kappa = kappa.detach().cpu().numpy()
        mp_logI = vMFLogPartition.log( vMFLogPartition.besseli(s, mp_kappa) )
        logI = torch.from_numpy( np.array(mp_logI.tolist(), dtype=float) ).to(kappa)
        
        if (logI!=logI).sum().item() > 0:  # there is nan
            raise ValueError('NaN is detected from the output of log-besseli()')
        
        logC = d * vMFLogPartition.nhlog2pi + s * kappa.log() - logI
        
        # save for backard()
        ctx.s, ctx.mp_kappa, ctx.logI = s, mp_kappa, logI
        
        return logC
        
    @staticmethod
    def backward(ctx, *grad_output):
        
        s, mp_kappa, logI = ctx.s, ctx.mp_kappa, ctx.logI 
    
        # log I_{s+1}(kappa)
        mp_logI2 = vMFLogPartition.log( vMFLogPartition.besseli(s+1, mp_kappa) )
        logI2 = torch.from_numpy( np.array(mp_logI2.tolist(), dtype=float) ).to(logI)
        
        if (logI2!=logI2).sum().item() > 0:  # there is nan
            raise ValueError('NaN is detected from the output of log-besseli()')
        
        dlogC_dkappa = -(logI2 - logI).exp()
        
        return None, grad_output[0] * dlogC_dkappa


def channel_squeeze(x, x_mask=None, n_sqz=2):
    """Function to perform squeeze operation increasing the channel size by factor n_sqz
    by reducing the number of time steps by factor n_sqz. Cuts of if number of time steps is odd.

    Args:
        x (torch.Tensor): Input tensor
        x_mask (torch.Tensor, optional): Masking tensor. Defaults to None.
        n_sqz (int, optional): Squeeze factor. Defaults to 2.

    Returns:
        torch.Tensor: Squeezed input tensor

        #TODO: Rename this and the next function
    """
    b, c, t = x.size()

    # cut of if odd number of time steps
    t = (t // n_sqz) * n_sqz
    x = x[:, :, :t]

    # squeeze
    x_sqz = x.view(b, c, t // n_sqz, n_sqz)
    x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c * n_sqz, t // n_sqz)

    if x_mask is not None:
        x_mask = x_mask[:, :, n_sqz - 1 :: n_sqz]
    else:
        x_mask = torch.ones(b, 1, t // n_sqz).to(device=x.device, dtype=x.dtype)
    return x_sqz * x_mask, x_mask


def channel_unsqueeze(x, x_mask=None, n_sqz=2):
    """Function to unsqueeze the input reducing the channel size by factor n_sqz
    by increasing the number of time steps by factor n_sqz.

    Args:
        x (torch.Tensor): Input tensor
        x_mask (torch.Tensor, optional): Masking tensor. Defaults to None.
        n_sqz (int, optional): Squeeze factor. Defaults to 2.

    Returns:
        torch.Tensor: Unsqueezed input tensor
    """
    b, c, t = x.size()

    x_unsqz = x.view(b, n_sqz, c // n_sqz, t)
    x_unsqz = x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c // n_sqz, t * n_sqz)

    if x_mask is not None:
        x_mask = x_mask.unsqueeze(-1).repeat(1, 1, 1, n_sqz).view(b, 1, t * n_sqz)
    else:
        x_mask = torch.ones(b, 1, t * n_sqz).to(device=x.device, dtype=x.dtype)
    return x_unsqz * x_mask, x_mask
