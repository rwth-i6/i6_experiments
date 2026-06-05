import torch
from torch import nn
import torch.nn.functional as F

from .bert_mlm_cfg import ModelConfig


class Model(nn.Module):
    """
    BERT-style masked language model with factorized, tied token embeddings.

    The input and output token embedding matrix is shared. Because embedding_dim
    can differ from hidden_size, the encoder input is projected up to hidden_size
    and the encoder output is projected back to embedding_dim before the tied
    output projection.
    """

    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
        if self.cfg.hidden_size % self.cfg.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={self.cfg.hidden_size} must be divisible by "
                f"num_attention_heads={self.cfg.num_attention_heads}"
            )
        if self.cfg.mask_replace_probability + self.cfg.random_replace_probability > 1.0:
            raise ValueError("mask_replace_probability + random_replace_probability must be <= 1.0")

        try:
            from transformers import BertConfig, BertModel
        except ImportError as exc:
            raise ImportError("bert_mlm requires the 'transformers' package in the RETURNN environment.") from exc

        bert_config = BertConfig(
            vocab_size=self.cfg.vocab_size,
            hidden_size=self.cfg.hidden_size,
            num_hidden_layers=self.cfg.num_hidden_layers,
            num_attention_heads=self.cfg.num_attention_heads,
            intermediate_size=self.cfg.intermediate_size,
            hidden_dropout_prob=self.cfg.hidden_dropout_prob,
            attention_probs_dropout_prob=self.cfg.attention_probs_dropout_prob,
            max_position_embeddings=self.cfg.max_position_embeddings,
            layer_norm_eps=self.cfg.layer_norm_eps,
            pad_token_id=self.cfg.pad_token_id,
            type_vocab_size=1,
            initializer_range=self.cfg.initializer_range,
        )

        self.token_embedding = nn.Embedding(
            self.cfg.vocab_size,
            self.cfg.embedding_dim,
            padding_idx=self.cfg.pad_token_id,
        )
        self.input_projection = nn.Linear(self.cfg.embedding_dim, self.cfg.hidden_size)
        self.bert = BertModel(bert_config, add_pooling_layer=False)
        self.output_projection = nn.Linear(self.cfg.hidden_size, self.cfg.embedding_dim)
        if self.cfg.output_bias:
            self.output_bias = nn.Parameter(torch.zeros(self.cfg.vocab_size))
        else:
            self.register_parameter("output_bias", None)

        self._init_added_parameters()

    def _init_added_parameters(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=self.cfg.initializer_range)
        with torch.no_grad():
            self.token_embedding.weight[self.cfg.pad_token_id].zero_()
        nn.init.normal_(self.input_projection.weight, mean=0.0, std=self.cfg.initializer_range)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=self.cfg.initializer_range)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        embeddings = self.token_embedding(input_ids)
        hidden_inputs = self.input_projection(embeddings)
        encoder_out = self.bert(
            inputs_embeds=hidden_inputs,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(input_ids),
            return_dict=True,
        ).last_hidden_state
        output_embeddings = F.relu(self.output_projection(encoder_out))
        logits = torch.matmul(output_embeddings, self.token_embedding.weight.transpose(0, 1))
        if self.output_bias is not None:
            logits = logits + self.output_bias
        return logits

    def make_mlm_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        cfg = self.cfg
        labels = input_ids.clone()
        special_mask = (
            (input_ids == cfg.pad_token_id)
            | (input_ids == cfg.cls_token_id)
            | (input_ids == cfg.sep_token_id)
            | (attention_mask == 0)
        )
        mask_candidates = ~special_mask
        selected = (torch.rand(input_ids.shape, device=input_ids.device) < cfg.mlm_probability) & mask_candidates

        # Avoid zero masked positions in a batch, which would make the loss undefined.
        if not bool(selected.any()):
            candidate_positions = torch.nonzero(mask_candidates, as_tuple=False)
            if candidate_positions.numel() > 0:
                selected[tuple(candidate_positions[torch.randint(candidate_positions.shape[0], (1,), device=input_ids.device)][0])] = True

        labels[~selected] = -100
        corrupted = input_ids.clone()

        replace_draw = torch.rand(input_ids.shape, device=input_ids.device)
        replace_with_mask = selected & (replace_draw < cfg.mask_replace_probability)
        replace_with_random = selected & (
            replace_draw >= cfg.mask_replace_probability
        ) & (replace_draw < cfg.mask_replace_probability + cfg.random_replace_probability)
        corrupted[replace_with_mask] = cfg.mask_token_id

        if bool(replace_with_random.any()):
            random_tokens = torch.randint(0, cfg.vocab_size, input_ids.shape, device=input_ids.device)
            disallowed = (
                (random_tokens == cfg.pad_token_id)
                | (random_tokens == cfg.cls_token_id)
                | (random_tokens == cfg.sep_token_id)
                | (random_tokens == cfg.mask_token_id)
            )
            random_tokens = torch.where(disallowed, input_ids, random_tokens)
            corrupted[replace_with_random] = random_tokens[replace_with_random]

        return corrupted, labels


def train_step(*, model: Model, data, run_ctx, **kwargs):
    input_ids = data["data"].to(torch.long)
    seq_len = data["data:size1"].to(torch.long)
    max_time = input_ids.shape[1]
    attention_mask = torch.arange(max_time, device=input_ids.device)[None, :] < seq_len[:, None]

    corrupted_input_ids, labels = model.make_mlm_inputs(input_ids=input_ids, attention_mask=attention_mask)
    logits = model(input_ids=corrupted_input_ids, attention_mask=attention_mask.to(dtype=torch.long))
    loss = F.cross_entropy(
        logits.reshape(-1, model.cfg.vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    num_masked = torch.count_nonzero(labels != -100).to(dtype=torch.float32)
    run_ctx.mark_as_loss(
        name="mlm_ce",
        loss=loss,
        inv_norm_factor=torch.clamp(num_masked, min=1.0),
    )
