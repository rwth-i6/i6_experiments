import torch
from torch import nn

from .kazuki_lstm_zijian_variant_v1_cfg import ModelConfig

def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """
    mask a tensor with a "positive" mask (boolean true means position is used)

    :param tensor: [B,T,....]
    :param seq_len: [B]
    :return: [B,T] as boolean
    """
    seq_len = seq_len.to(device=tensor.device)
    r = torch.arange(tensor.shape[1], device=tensor.device)  # [T]
    seq_mask = torch.less(r[None, :], seq_len[:, None])  # broadcast to [B,T]
    return seq_mask


class Model(nn.Module):
    """
    Simple LSTM LM with an embedding, an LSTM, and a final linear
    """
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig(**model_config_dict)
        if self.cfg.dropout > 0:
            self.dropout = nn.Dropout(p=self.cfg.dropout)
        else:
            self.dropout = None
        self.use_bottle_neck = self.cfg.use_bottle_neck
        self.embed = nn.Embedding(self.cfg.vocab_dim, self.cfg.embed_dim)
        self.lstm = nn.LSTM(
            input_size=self.cfg.embed_dim,
            hidden_size=self.cfg.hidden_dim,
            num_layers=self.cfg.n_lstm_layers,
            bias=self.cfg.bias,
            batch_first=True,
            dropout=self.cfg.dropout,
            bidirectional=False,
        )
        if self.cfg.use_bottle_neck:
            self.bottle_neck = nn.Linear(self.cfg.hidden_dim,self.cfg.bottle_neck_dim, bias=True)
            self.final_linear = nn.Linear(self.cfg.bottle_neck_dim, self.cfg.vocab_dim, bias=True)
        else:
            self.final_linear = nn.Linear(self.cfg.hidden_dim, self.cfg.vocab_dim, bias=True)

        if self.cfg.init_args is not None:
            self._param_init(**self.cfg.init_args)


    def _param_init(self, init_args_w=None, init_args_b=None):
        for m in self.modules():
            for name, param in m.named_parameters():
                if 'bias' in name:
                    if init_args_b['func'] == 'normal':
                        init_func = nn.init.normal_
                    else:
                        NotImplementedError
                    hyp = init_args_b['arg']
                else:
                    if init_args_w['func'] == 'normal':
                        init_func = nn.init.normal_
                    else:
                        NotImplementedError
                    hyp = init_args_w['arg']
                init_func(param, **hyp)

    def forward(self, x, states):
        """
        Return logits of each batch at each time step
        :param x: (B, S, F)
        :param states: tuple of h, c, with [batch, #layers, F] each
        """
        x = self.embed(x)
        if self.dropout:
            x = self.dropout(x)
        batch_size = x.shape[0]
        if states is None:
            h0 = torch.zeros((self.cfg.n_lstm_layers, batch_size, self.cfg.hidden_dim), device=x.device).detach()
            c0 = torch.zeros_like(h0, device=x.device).detach()
        else:
            _h0, _c0 = states
            h0 = torch.transpose(_h0, 0, 1).contiguous()  # reshape to [#layers, batch, F]
            c0 = torch.transpose(_c0, 0, 1).contiguous()

        # This is a uni-directional LSTM, so sequence masking is not necessary
        x, (h, c) = self.lstm(x, (h0, c0))
        if self.dropout:
            x = self.dropout(x)
        if self.use_bottle_neck:
            x = self.bottle_neck(x)
            if self.dropout:
                x = self.dropout(x)
        x = self.final_linear(x)
        return x, (torch.transpose(h, 0, 1), torch.transpose(c, 0, 1))  # return states with [batch, #layers, F]
    
    
def train_step(*, model: Model, data, run_ctx, **kwargs):
    labels = data["data"]
    labels_len = data["data:size1"]
    delayed_labels = data["delayed"]

    lm_logits, _ = model(delayed_labels)  # (B, S, F)

    ce_loss = torch.nn.functional.cross_entropy(lm_logits.transpose(1, 2), labels.long(), reduction='none')
    seq_mask = mask_tensor(labels, labels_len)
    ce_loss = (ce_loss * seq_mask).sum()
    total_length = torch.sum(labels_len)
    
    run_ctx.mark_as_loss(name="ce", loss=ce_loss, inv_norm_factor=total_length)
