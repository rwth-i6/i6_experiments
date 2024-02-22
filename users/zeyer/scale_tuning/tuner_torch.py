"""
Auto scaling, based on recog output.
"""

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
    arg_parser.add_argument("--device", default="cuda")
    arg_parser.add_argument("--num-steps", type=int, default=10_000)
    args = arg_parser.parse_args()
    device = torch.device(args.device)

    keys = []
    entries = []  # list of sequences over list [((scores per key), hyp num tokens, num errors), ...] per hyp
    entries_num_err = []  # list of sequences over tensor shape [beam]
    total_num_ref_words = 0

    for fn in args.recog_output_dir:
        print(f"* Reading entries from {fn}...")
        with gzip.open(fn + "/output.py.gz", "rt") as f:
            hyps = eval(f.read())
        with gzip.open(fn + "/output_ext.py.gz", "rt") as f:
            exts = eval(f.read())
        assert isinstance(hyps, dict) and isinstance(exts, dict) and set(hyps) == set(exts)
        print(f"* Collecting data...")
        for seq_tag in hyps:
            hyps_ = hyps[seq_tag]
            exts_ = exts[seq_tag]
            assert isinstance(hyps_, list) and isinstance(exts_, list) and len(hyps_) == len(exts_)
            if not keys:
                keys = list(exts_[0].keys())
            ref_words = hyps_[0][1].replace("@@ ", "").split()
            total_num_ref_words += len(ref_words)
            entries.append([])
            num_err_ls = []
            for (_, hyp), ext in zip(hyps_, exts_):
                hyp_num_tokens = len(hyp.split())
                hyp_words = hyp.replace("@@ ", "").split()
                num_errors = torchaudio.functional.edit_distance(ref_words, hyp_words)
                entries[-1].append(
                    (
                        torch.tensor([ext[key] for key in keys], device=device),
                        torch.tensor(hyp_num_tokens, device=device),
                    )
                )
                num_err_ls.append(float(num_errors))
            entries_num_err.append(torch.tensor(num_err_ls, device=device))
    entries_num_err = [num_errors_ / total_num_ref_words for num_errors_ in entries_num_err]

    print("* Start training...")

    scales = torch.nn.Parameter(torch.ones([len(keys) + 1], device=device))

    def _logits(entries_):
        seq_scores = []
        for scores, hyp_num_tokens in entries_:
            seq_score = torch.dot(scales[:-1], scores)
            seq_score /= hyp_num_tokens ** scales[-1]
            seq_scores.append(seq_score)
        seq_scores = torch.stack(seq_scores)
        return seq_scores

    opt = torch.optim.Adam([scales])

    for step in range(args.num_steps):
        opt.zero_grad()
        loss = torch.zeros((), device=device)
        for entries_, num_errors_ in zip(entries, entries_num_err):
            seq_scores = _logits(entries_)
            seq_probs = torch.nn.functional.softmax(seq_scores, dim=0)
            loss += torch.dot(seq_probs, num_errors_)
        loss.backward()
        opt.step()
        print(f"step {step}, loss: {loss}")

        if step % 100 == 0:
            with torch.no_grad():
                err = torch.zeros((), device=device)
                for entries_, num_errors_ in zip(entries, entries_num_err):
                    seq_scores = _logits(entries_)
                    best = seq_scores.argmax()
                    err += num_errors_[best]
                print(f"err: {err}, scales {scales.detach().cpu().numpy().tolist()}")


if __name__ == "__main__":
    main()
