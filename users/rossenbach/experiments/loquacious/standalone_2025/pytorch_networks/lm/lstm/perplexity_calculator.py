from dataclasses import dataclass
import torch
from torch import tensor

@dataclass()
class PerplexityConfig():
    bpe_count_file: str
    word_count_file: str

def forward_init_hook(run_ctx, **kwargs):
    config = PerplexityConfig(**kwargs["config"])

    run_ctx.val_loss = tensor(0.0, dtype=torch.float64)
    run_ctx.val_elements = tensor(0)

    run_ctx.val_loss_no_eos = tensor(0.0, dtype=torch.float64)
    run_ctx.val_elements_no_eos = tensor(0)

    run_ctx.val_loss_no_unk = tensor(0.0, dtype=torch.float64)
    run_ctx.val_elements_no_unk = tensor(0)

    run_ctx.val_loss_no_unk_no_eos = tensor(0.0, dtype=torch.float64)
    run_ctx.val_elements_no_unk_no_eos = tensor(0)

    forward_dataset = run_ctx.engine.forward_dataset
    run_ctx.unk_label = forward_dataset.orth_symbols_map[forward_dataset.unknown_symbol]

    with open(config.word_count_file) as f_word:
        run_ctx.word_count = float(f_word.readline().strip())

    with open(config.bpe_count_file) as f_bpe:
        run_ctx.bpe_count = float(f_bpe.readline().strip())

    run_ctx.word_factor = run_ctx.bpe_count / run_ctx.word_count
    print(f"Correct Perplexity to Word-Level using a factor of {run_ctx.word_factor}")

def forward_finish_hook(run_ctx, **kwargs):
    with open("val_loss", "wt") as f:
        f.write(f"{run_ctx.val_loss.item()}\n")
    with open("val_loss_no_eos", "wt") as f:
        f.write(f"{run_ctx.val_loss_no_eos.item()}\n")
    with open("val_loss_no_unk", "wt") as f:
        f.write(f"{run_ctx.val_loss_no_unk.item()}\n")
    with open("val_loss_no_unk_no_eos", "wt") as f:
        f.write(f"{run_ctx.val_loss_no_unk_no_eos.item()}\n")

    val_loss = run_ctx.val_loss / run_ctx.val_elements
    with open("ppl", "wt") as f:
        f.write(f"{torch.exp(val_loss)}\n")
    with open("word_ppl", "wt") as f:
        f.write(f"{torch.exp(val_loss)**run_ctx.word_factor}\n")

    val_loss_no_eos = run_ctx.val_loss_no_eos / run_ctx.val_elements_no_eos
    with open("ppl_no_eos", "wt") as f:
        f.write(f"{torch.exp(val_loss_no_eos)}\n")
    with open("word_ppl_no_eos", "wt") as f:
        f.write(f"{torch.exp(val_loss_no_eos)**run_ctx.word_factor}\n")

    val_loss_no_unk = run_ctx.val_loss_no_unk / run_ctx.val_elements_no_unk
    with open("ppl_no_unk", "wt") as f:
        f.write(f"{torch.exp(val_loss_no_unk)}\n")
    with open("word_ppl_no_unk", "wt") as f:
        f.write(f"{torch.exp(val_loss_no_unk)**run_ctx.word_factor}\n")

    val_loss_no_unk_no_eos = run_ctx.val_loss_no_unk_no_eos / run_ctx.val_elements_no_unk_no_eos
    with open("ppl_no_unk_no_eos", "wt") as f:
        f.write(f"{torch.exp(val_loss_no_unk_no_eos)}\n")
    with open("word_ppl_no_unk_no_eos", "wt") as f:
        f.write(f"{torch.exp(val_loss_no_unk_no_eos)**run_ctx.word_factor}\n")


def forward_step(*, model, data, run_ctx, **kwargs):
    labels = data["data"].long()
    labels_len = data["data:size1"]
    delayed_labels = data["delayed"]

    lm_logits, _ = model(delayed_labels, states=None)  # (B, S, F)

    log_flat = lm_logits.flatten(0, 1)
    tar_flat = labels.flatten(0, 1)

    loss = torch.nn.functional.cross_entropy(log_flat, tar_flat, reduction="none")
    loss = loss.reshape(lm_logits.shape[:-1])

    r = torch.arange(loss.shape[1], device=loss.device)  # [T]

    seq_mask = torch.less(r[None, :], labels_len[:, None].to(loss.device))  # broadcast to [B,T]
    log_prob_targets = torch.where(seq_mask, loss, 0)
    log_prob_targets_seq = torch.sum(log_prob_targets, dim=-1)

    run_ctx.val_loss = run_ctx.val_loss + torch.sum(log_prob_targets_seq, dtype=torch.float64)
    run_ctx.val_elements = run_ctx.val_elements + torch.sum(labels_len)
    
    seq_mask_no_eos = torch.less(r[None, :], labels_len[:, None].to(loss.device) - 1)  # broadcast to [B,T]
    assert torch.sum(seq_mask_no_eos) == torch.sum(seq_mask) - labels_len.shape[0], (
        seq_mask_no_eos,
        seq_mask,
    )
    log_prob_targets_no_eos = torch.where(seq_mask_no_eos, loss, 0)
    log_prob_targets_seq_no_eos = torch.sum(log_prob_targets_no_eos, dim=-1)

    run_ctx.val_loss_no_eos = run_ctx.val_loss + torch.sum(log_prob_targets_seq_no_eos, dtype=torch.float64)
    run_ctx.val_elements_no_eos = run_ctx.val_elements + torch.sum(labels_len - 1)
    
    unk_mask: torch.Tensor = labels == run_ctx.unk_label

    num_unks = torch.sum(unk_mask, dim=-1).to(device=labels_len.device)
    log_prob_targets_no_unk = torch.where(unk_mask, 0, log_prob_targets)
    log_prob_targets_seq_no_unk = torch.sum(log_prob_targets_no_unk, dim=-1)
    
    run_ctx.val_loss_no_unk = run_ctx.val_loss + torch.sum(log_prob_targets_seq_no_unk, dtype=torch.float64)
    run_ctx.val_elements_no_unk = run_ctx.val_elements + torch.sum(labels_len - num_unks)
    
    log_prob_targets_no_unk_no_eos = torch.where(unk_mask, 0, log_prob_targets_no_eos)
    log_prob_targets_seq_no_unk_no_eos = torch.sum(log_prob_targets_no_unk_no_eos, dim=-1)

    run_ctx.val_loss_no_unk_no_eos = run_ctx.val_loss + torch.sum(log_prob_targets_seq_no_unk_no_eos, dtype=torch.float64)
    run_ctx.val_elements_no_unk_no_eos = run_ctx.val_elements + torch.sum(labels_len - num_unks - 1)
