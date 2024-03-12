import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# import mpmath
from scipy import special as sp

# from librosa.filters import mel as librosa_mel_fn
# from audio_processing import dynamic_range_compression
# from audio_processing import dynamic_range_decompression
# from stft import STFT


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


def encoding_distance_loss(phoneme_sequences, encodings, seq_lenghts):
    # Create mask and apply mask an offset of -1 to have an additional value as the masked phoneme,
    # which can later be used to filter out the mean for that special label
    mask = sequence_mask(seq_lenghts, phoneme_sequences.shape[-1])
    phoneme_sequences_masked = ((phoneme_sequences + 1) * mask) - 1
    encodings_masked = ((encodings.transpose(1,2) + 1) * mask.unsqueeze(-1)) - 1

    # Reshape the encodings tensor [B, T, F] -> [B*T, F] and phonemes sequence [B, T] -> [B*T]
    encodings_flat = encodings_masked.reshape(-1, encodings_masked.size(-1))
    phoneme_sequences_flat = phoneme_sequences_masked.view(-1)

    # Use torch.unique to get list of unique phonemes in full batch and index for index_add function
    # and counts which can then be used for the mean calculation
    phonemes, inverse, counts = torch.unique(phoneme_sequences_flat, return_counts=True, return_inverse=True)

    # Initialize a tensor to store the sum of encodings for each phoneme
    sum_encodings = torch.zeros(phonemes.shape[0], encodings_masked.size(-1), device=phoneme_sequences.device)

    # Use index_add to collect the sum of all phoneme encodings in the current batch
    sum_encodings = sum_encodings.index_add(0, inverse, encodings_flat)

    # Remove the first label, if it is the -1 from the masking
    if phonemes[0] == -1:
        sum_encodings_masked = sum_encodings[1:]
        counts_masked = counts[1:]

    mean_encodings = (sum_encodings_masked / counts_masked.unsqueeze(1)).unsqueeze(0)

    # Use cdist to calculate pairwise distance for all encodings, the negative sum of which is the loss
    loss = -1 * (torch.cdist(mean_encodings, mean_encodings).sum() / (phoneme_sequences.shape[0] * float(phoneme_sequences.max())))
    return loss


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels, debug_name: str="None"):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act

    # if (debug_name != "None"):
    #     torch.save(in_act, f"{debug_name}_in_act.pkl")
    #     torch.save(t_act, f"{debug_name}_t_act.pkl")
    #     torch.save(s_act, f"{debug_name}_s_act.pkl")
    #     torch.save(acts, f"{debug_name}_acts.pkl")
    return acts

def fused_add_tanh_sigmoid_multiply_no_jit(input_a, input_b, n_channels, debug_name: str="None"):
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


def generate_path(duration, mask):
    """
    duration: [b, t_x]
    mask: [b, t_x, t_y]
    """
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
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

class Adam:
    def __init__(self, params, scheduler, dim_model, warmup_steps=4000, lr=1e0, betas=(0.9, 0.98), eps=1e-9):
        self.params = params
        self.scheduler = scheduler
        self.dim_model = dim_model
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.betas = betas
        self.eps = eps

        self.step_num = 1
        self.cur_lr = lr * self._get_lr_scale()

        self._optim = torch.optim.Adam(params, lr=self.cur_lr, betas=betas, eps=eps)

    def _get_lr_scale(self):
        if self.scheduler == "noam":
            return np.power(self.dim_model, -0.5) * np.min(
                [np.power(self.step_num, -0.5), self.step_num * np.power(self.warmup_steps, -1.5)]
            )
        else:
            return 1

    def _update_learning_rate(self):
        self.step_num += 1
        if self.scheduler == "noam":
            self.cur_lr = self.lr * self._get_lr_scale()
            for param_group in self._optim.param_groups:
                param_group["lr"] = self.cur_lr

    def get_lr(self):
        return self.cur_lr

    def step(self):
        self._optim.step()
        self._update_learning_rate()

    def zero_grad(self):
        self._optim.zero_grad()

    def load_state_dict(self, d):
        self._optim.load_state_dict(d)

    def state_dict(self):
        return self._optim.state_dict()


# class TacotronSTFT(nn.Module):
#     def __init__(
#         self,
#         filter_length=1024,
#         hop_length=256,
#         win_length=1024,
#         n_mel_channels=80,
#         sampling_rate=22050,
#         mel_fmin=0.0,
#         mel_fmax=8000.0,
#     ):
#         super(TacotronSTFT, self).__init__()
#         self.n_mel_channels = n_mel_channels
#         self.sampling_rate = sampling_rate
#         self.stft_fn = STFT(filter_length, hop_length, win_length)
#         mel_basis = librosa_mel_fn(sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
#         mel_basis = torch.from_numpy(mel_basis).float()
#         self.register_buffer("mel_basis", mel_basis)

#     def spectral_normalize(self, magnitudes):
#         output = dynamic_range_compression(magnitudes)
#         return output

#     def spectral_de_normalize(self, magnitudes):
#         output = dynamic_range_decompression(magnitudes)
#         return output

#     def mel_spectrogram(self, y):
#         """Computes mel-spectrograms from a batch of waves
#         PARAMS
#         ------
#         y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

#         RETURNS
#         -------
#         mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
#         """
#         assert torch.min(y.data) >= -1
#         assert torch.max(y.data) <= 1

#         magnitudes, phases = self.stft_fn.transform(y)
#         magnitudes = magnitudes.data
#         mel_output = torch.matmul(self.mel_basis, magnitudes)
#         mel_output = self.spectral_normalize(mel_output)
#         return mel_output


def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type

        p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


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
