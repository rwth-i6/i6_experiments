"""
Auto scaling, based on recog output.
"""

import sys
import argparse
import gzip
import torch
import torchaudio


def main():
    """main"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "recog_output_dir",
        nargs="+",
        help="from our recog, expect output.py.gz and output_ext.py.gz. assume first entry is ground truth",
    )
    arg_parser.add_argument("--device", default="cpu")
    arg_parser.add_argument("--num-steps", type=int, default=10_000)
    arg_parser.add_argument("--seqs-start", type=float, default=0)
    arg_parser.add_argument("--seqs-end", type=float, default=1)
    arg_parser.add_argument("--random-seed", type=int, default=42)
    arg_parser.add_argument("--lr", type=float, default=0.01)
    arg_parser.add_argument("--opt", default="Adam")
    arg_parser.add_argument("--init-scales")
    arg_parser.add_argument("--init-len-norm", type=float, default=1.0)
    arg_parser.add_argument("--train-len-norm", type=int, default=True)
    arg_parser.add_argument("--fix-scale0", type=int, default=True)
    arg_parser.add_argument("--softmax-temperature", type=float, default=10)
    args = arg_parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.random_seed)

    keys = []  # len: scores
    entries_scores = []  # list of sequences over tensor shape [beam,scores]
    entries_hyp_num_tokens = []  # list of sequences over tensor shape [beam]
    entries_num_err = []  # list of sequences over tensor shape [beam]
    total_num_ref_words = 0

    hyps = {}
    exts = {}
    for fn in args.recog_output_dir:
        print(f"* Reading entries from {fn}...")
        with gzip.open(fn + "/output.py.gz", "rt") as f:
            hyps_f = eval(f.read())
        with gzip.open(fn + "/output_ext.py.gz", "rt") as f:
            exts_f = eval(f.read())
        assert isinstance(hyps_f, dict) and isinstance(exts_f, dict) and set(hyps_f) == set(exts_f)
        assert not set(hyps_f.keys()).intersection(hyps.keys())
        hyps.update(hyps_f)
        exts.update(exts_f)

    print(f"* Processing data...")
    seq_tags = list(hyps)
    print("Num seqs:", len(seq_tags))
    seq_tags = [seq_tags[i] for i in torch.randperm(len(seq_tags))]
    if args.seqs_start != 0 or args.seqs_end != 1:
        assert 0 <= args.seqs_start <= 1 and 0 <= args.seqs_end <= 1 and args.seqs_start <= args.seqs_end
        start = int(args.seqs_start * len(seq_tags))
        end = int(args.seqs_end * len(seq_tags))
        seq_tags = seq_tags[start:end]
        print(f"Selected subset (after shuffling): [{start}:{end}], num seqs: {len(seq_tags)}")

    key_signs = None
    for seq_tag in seq_tags:
        hyps_ = hyps[seq_tag]
        exts_ = exts[seq_tag]
        assert isinstance(hyps_, list) and isinstance(exts_, list) and len(hyps_) == len(exts_)
        if not keys:
            keys = list(exts_[0].keys())
            print("Score keys:", keys)
            key_signs = [1.0 if "_neg_" not in key else -1.0 for key in keys]
            print("Key signs:", key_signs)
            print("Beam size:", len(hyps_))
        ref_words = hyps_[0][1].replace("@@ ", "").split()
        total_num_ref_words += len(ref_words)
        scores_ls = []
        hyp_num_tokens_ls = []
        num_err_ls = []
        for (_, hyp), ext in zip(hyps_, exts_):
            hyp_num_tokens = len(hyp.split())
            hyp_words = hyp.replace("@@ ", "").split()
            num_errors = torchaudio.functional.edit_distance(ref_words, hyp_words)
            scores_ls.append(torch.tensor([ext[key] for key in keys]))
            hyp_num_tokens_ls.append(float(hyp_num_tokens))
            num_err_ls.append(float(num_errors))
        entries_scores.append(torch.stack(scores_ls))
        entries_hyp_num_tokens.append(torch.tensor(hyp_num_tokens_ls))
        entries_num_err.append(torch.tensor(num_err_ls))

    entries_scores = torch.stack(entries_scores)  # [seqs,beam,scores]
    entries_hyp_num_tokens = torch.stack(entries_hyp_num_tokens)  # [seqs,beam]
    entries_num_err = torch.stack(entries_num_err)  # [seqs,beam]
    entries_num_err /= total_num_ref_words

    key_signs = torch.tensor(key_signs, device=device)  # [scores]
    entries_scores = entries_scores.to(device)
    entries_hyp_num_tokens = entries_hyp_num_tokens.to(device)
    entries_num_err = entries_num_err.to(device)

    print("* Start training...")

    if args.init_scales:
        init_scales = [float(s) for s in args.init_scales.split(",")]
        assert len(init_scales) == len(keys)
    else:
        init_scales = [1.0] * len(keys)
    print("Using initial scales:", init_scales, "initial len_norm:", args.init_len_norm)
    scales = torch.nn.Parameter(torch.tensor(init_scales, device=device))
    len_norm_scale = torch.nn.Parameter(torch.tensor(args.init_len_norm, device=device))
    print("Train len norm:", bool(args.train_len_norm))
    params = [scales, len_norm_scale] if args.train_len_norm else [scales]
    print("Fix scale0:", bool(args.fix_scale0))

    def _logits():
        return torch.einsum(
            "s,s,abs,ab->ab",
            scales,
            key_signs,
            entries_scores,
            torch.reciprocal(entries_hyp_num_tokens + 1) ** len_norm_scale,
        )  # [seqs,beam]

    def _loss():
        seq_scores = _logits()  # [seqs,beam]
        seq_scores *= args.softmax_temperature
        seq_probs = torch.nn.functional.softmax(seq_scores, dim=-1)  # [seqs,beam]
        return torch.einsum("sb,sb->", seq_probs, entries_num_err)

    def _err():
        seq_scores = _logits()  # [seqs,beam]
        best = seq_scores.argmax(dim=1)  # [seqs] -> beam
        err_ = batch_gather(entries_num_err, indices=best)  # [seqs]
        return torch.sum(err_)

    def _scales_fix(*, fix_scale0_anyway: bool = False):
        scales.data = torch.nn.functional.relu(scales)  # keep positive
        len_norm_scale.data = torch.nn.functional.relu(len_norm_scale)
        if args.fix_scale0 or fix_scale0_anyway:
            scales.data *= 1.0 / torch.maximum(scales[0], torch.tensor(0.01, device=device))

    def _scales_str():
        res = "scales [%s]" % ", ".join(f"{s:.4f}" for s in scales.detach().cpu())
        res += f", len_norm {len_norm_scale.detach().cpu():.4f}"
        return res

    print(f"Initial err: {_err():.4f}")

    opt = getattr(torch.optim, args.opt)(params, lr=args.lr)

    for step in range(args.num_steps):
        opt.zero_grad()
        loss = _loss()
        loss.backward()
        opt.step()
        _scales_fix()

        with torch.no_grad():
            if step % 10 == 0:
                print(f"step {step}, loss: {loss:.4f}")

            if step % 100 == 0:
                err = _err()
                print(f"err: {err:.4f}, {_scales_str()}")

    print("Finished.")
    with torch.no_grad():
        print(f"Final loss: {_loss():.4f}, err: {_err():.4f}, {_scales_str()}")
        if not args.fix_scale0:
            _scales_fix(fix_scale0_anyway=True)
            print(f"(Or scale0 fixed anyway: Final loss: {_loss():.4f}, err: {_err():.4f}, {_scales_str()})")


def batch_gather(values: torch.Tensor, *, indices: torch.Tensor) -> torch.Tensor:
    """
    :param values: shape [Batch,Indices,ValuesDims...], e.g. [Batch,InBeam,...]
    :param indices: shape [Batch,IndicesDims...] -> Indices, e.g. [Batch,OutBeam] -> InBeam
    :return: shape [Batch,IndicesDims...,ValuesDims...], e.g. [Batch,OutBeam,...]
    """
    # Derived from returnn.torch.frontend._backend.TorchBackend.gather.
    # Case indices.dims_set.intersection(source.dims_set - {axis}).
    # We cannot use index_select in this case. Need to fallback to gather.
    assert indices.shape[0] == values.shape[0]
    num_index_own_dims = indices.ndim - 1
    if num_index_own_dims == 1:
        indices_flat = indices  # good, [Batch,IndexDim]
    elif num_index_own_dims == 0:
        indices_flat = indices[:, None]  # [Batch,IndexDim=1]
    else:
        indices_flat = indices.flatten(1)  # [Batch,FlatIndexDim]
    indices_flat_bc = indices_flat.reshape(list(indices_flat.shape) + [1] * (values.ndim - 2))  # [Batch,IndexDim,1s...]
    indices_flat_exp = indices_flat_bc.expand(indices_flat.shape + values.shape[2:])  # [Batch,IndexDim,ValuesDims...]
    out = torch.gather(values, dim=1, index=indices_flat_exp.type(torch.int64))
    if num_index_own_dims == 1:
        pass  # nothing to do
    elif num_index_own_dims == 0:
        out = out.squeeze(1)
    else:
        out = out.unflatten(1, indices.shape[1:])
    assert out.shape == indices.shape + values.shape[2:]
    return out


def _setup():
    print("PyTorch:", torch.__version__)

    try:
        import better_exchook

        better_exchook.install()
    except ImportError:
        print("(no better_exchook)")

    try:
        import lovely_tensors

        lovely_tensors.monkey_patch()
    except ImportError:
        print("(no lovely_tensors)")


if __name__ == "__main__":
    _setup()
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        sys.exit(1)
