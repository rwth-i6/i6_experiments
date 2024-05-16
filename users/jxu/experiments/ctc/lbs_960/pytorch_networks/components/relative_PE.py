import math

import torch

class OnnxRelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module (new implementation).
    Details can be found in https://github.com/espnet/espnet/pull/2816.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_seq_len (int): Maximum input length.
    """

    def __init__(self, d_model, max_seq_len=1024, use_cache=True):
        """Construct an PositionalEncoding object."""
        super(OnnxRelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.pe = None
        self.use_cache = use_cache
        if self.use_cache:
            self.extend_pe(torch.tensor(0.0).expand(1, max_seq_len))
        else:
            self.div_term = torch.exp(
                torch.arange(0, self.d_model, 2, dtype=torch.float32)
                * -(math.log(10000.0) / self.d_model)
            )

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None and self.pe.size(1) >= x.size(1) * 2 - 1:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.dtype != x.dtype or self.pe.device != x.device:
                self.pe = self.pe.to(dtype=x.dtype, device=x.device)
            return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def _get_pe(self, x):
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        theta = (
            torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1) * self.div_term
        )
        pe_positive[:, 0::2] = torch.sin(theta)
        pe_positive[:, 1::2] = torch.cos(theta)
        pe_negative[:, 0::2] = -1 * torch.sin(theta)
        pe_negative[:, 1::2] = torch.cos(theta)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        return torch.cat([pe_positive, pe_negative], dim=1)

    def forward(self, x: torch.Tensor, use_cache=True):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        x = x * self.xscale
        if self.use_cache:
            pos_emb = self.pe[
                :,
                self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
            ]
        else:
            pos_emb = self._get_pe(x)
        return x, pos_emb

