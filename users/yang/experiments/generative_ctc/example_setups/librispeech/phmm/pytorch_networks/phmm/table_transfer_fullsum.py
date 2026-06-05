import torch
from torch import nn

from .table_transfer_fullsum_cfg import ModelConfig
from ...loss.unsupervised_full_sum_loss import one_to_one_full_sum


class Model(nn.Module):
    """
    Table-only generative transfer model.

    The trainable parameter is `emission_table` with shape
    [input_vocab_size, output_vocab_size]. After log-softmax over the input
    dimension, it represents log p(input_id | output_id). For an input sequence
    x[b, t], the model emits log probabilities over output labels:

        log_probs[b, t, y] = log p(x[b, t] | y)

    These scores are consumed by the one-to-one full-sum criterion.
    """

    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
        if self.cfg.input_vocab_size <= 0:
            raise ValueError(f"input_vocab_size must be positive, got {self.cfg.input_vocab_size}")
        if self.cfg.output_vocab_size <= 0:
            raise ValueError(f"output_vocab_size must be positive, got {self.cfg.output_vocab_size}")
        if self.cfg.softmax_temperature <= 0.0:
            raise ValueError(f"softmax_temperature must be positive, got {self.cfg.softmax_temperature}")

        self.emission_table = nn.Parameter(
            self.cfg.table_init_scale
            * torch.randn(self.cfg.input_vocab_size, self.cfg.output_vocab_size, dtype=torch.float32)
        )
        self.register_buffer("lm_table", self._load_lm_table(self.cfg.lm_table_path), persistent=False)

    @staticmethod
    def _get_path(path):
        try:
            from returnn.util.basic import cf

            return cf(path)
        except Exception:
            return str(path)

    def _load_lm_table(self, path):
        table = torch.load(self._get_path(path), map_location="cpu")
        if isinstance(table, dict):
            for key in ("log_probs", "lm_table", "scores", "data"):
                if key in table:
                    table = table[key]
                    break
        table = torch.as_tensor(table, dtype=torch.float32)
        expected_shape = (self.cfg.lm_vocab_size ** self.cfg.lm_context_length, self.cfg.lm_vocab_size)
        if tuple(table.shape) != expected_shape:
            raise ValueError(f"Expected LM table shape {expected_shape}, got {tuple(table.shape)}")
        return table

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(dtype=torch.long)
        if input_ids.numel() > 0:
            min_id = int(input_ids.min().detach().cpu())
            max_id = int(input_ids.max().detach().cpu())
            if min_id < 0 or max_id >= self.cfg.input_vocab_size:
                raise ValueError(
                    f"input ids out of range: min={min_id}, max={max_id}, input_vocab_size={self.cfg.input_vocab_size}"
                )

        log_p_input_given_output = torch.log_softmax(
            self.emission_table / self.cfg.softmax_temperature,
            dim=0,
        )  # [input_vocab, output_vocab]
        return log_p_input_given_output[input_ids]  # [B, T, output_vocab]


def train_step(*, model: Model, data, run_ctx, **kwargs):
    input_ids = data["data"].to(torch.long)
    seq_len = data["data:size1"].to(torch.long)
    log_probs = model(input_ids)
    log_likelihood = one_to_one_full_sum(
        log_probs=log_probs,
        seq_len=seq_len,
        lm_table=model.lm_table,
        lm_vocab_size=model.cfg.lm_vocab_size,
        lm_context_length=model.cfg.lm_context_length,
        k=model.cfg.beam_size,
        use_lm_silence_score=model.cfg.use_lm_silence_score,
        lm_scale=model.cfg.lm_scale,
        am_scale=model.cfg.am_scale,
    )
    num_tokens = torch.clamp(seq_len.sum().to(dtype=torch.float32), min=1.0)
    run_ctx.mark_as_loss(
        name="full_sum",
        loss=-log_likelihood,
        inv_norm_factor=num_tokens,
    )
