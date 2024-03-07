"""
Script to run our torch beam search implementations inside espnet inference module.
This means we ESPnet to get the model, data loader, tokenizer, etc
"""

import logging
import argparse

from types import SimpleNamespace
import sys
import numpy as np
import time

sys.path.append("/u/zeineldeen/dev/pylasr")
sys.path.append("/u/zeineldeen/dev/returnn")
sys.path.append("/u/zeineldeen/dev/espnet")
sys.path.append("/u/zeineldeen/setups/ubuntu_22_setups/2024-02-12--aed-beam-search/recipe")
sys.path.append(
    "/u/zeineldeen/setups/ubuntu_22_setups/2024-02-12--aed-beam-search/sisyphus"
)  # issue in import tk.setup_path

import torch

from espnet2.bin.asr_inference import Speech2Text
from espnet2.tasks.asr import ASRTask
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.torch_utils.device_funcs import to_device
from espnet2.text.build_tokenizer import build_tokenizer
from espnet.nets.scorers.ctc import CTCPrefixScorer


parser = argparse.ArgumentParser()
parser.add_argument("--model_tag", type=str, default="asapp/e_branchformer_librispeech")
# use pylasr decoder if this is set
parser.add_argument("--pylasr_recog_args", type=str, default=None)
# use returnn beam search if this is set
parser.add_argument("--returnn_recog_args", type=str, default=None)
# required paths
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--log_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
# overwrite with these params
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--nbest", type=int, default=1)
parser.add_argument("--maxlenratio", type=float, default=0.0)
parser.add_argument("--beam_size", type=int, default=12)
parser.add_argument("--ctc_weight", type=float, default=0.0)
parser.add_argument("--lm_weight", type=float, default=0.0)
parser.add_argument("--len_reward", type=float, default=0.0)
parser.add_argument("--converage_threshold", type=float, default=0.0)
parser.add_argument("--converage_scale", type=float, default=0.0)
parser.add_argument("--normalize_length", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=1)
#
parser.add_argument("--log_level", type=str, default="INFO")
# search dataset
parser.add_argument("--dataset", type=str, default="dev_other")

args = parser.parse_args()

batch_size = args.batch_size

set_all_random_seed(0)
logging.basicConfig(
    level=args.log_level,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)

import better_exchook

better_exchook.install()


def get_data_loader(dataset_name):
    loader = ASRTask.build_streaming_iterator(
        [(f"{args.data_path}/data/{dataset_name}/wav.scp", "speech", "sound")],
        dtype="float32",
        batch_size=batch_size,
        key_file=None,
        num_workers=2,
        preprocess_fn=ASRTask.build_preprocess_fn(speech2text.asr_train_args, False),
        collate_fn=ASRTask.build_collate_fn(speech2text.asr_train_args, False),
        allow_variable_data_keys=False,
        inference=True,
    )
    return loader


speech2text = Speech2Text.from_pretrained(
    cache_dir="/u/zeineldeen/setups/ubuntu_22_setups/2024-02-12--aed-beam-search/work/downloaded_models",
    model_tag=args.model_tag,
    maxlenratio=args.maxlenratio,  # uses end-detect function if set to 0
    minlenratio=0.0,
    beam_size=args.beam_size,
    ctc_weight=args.ctc_weight,
    lm_weight=args.lm_weight,
    penalty=args.len_reward,  # insertion penalty
    nbest=args.nbest,
    batch_size=batch_size,
    device=args.device,
    normalize_length=args.normalize_length,
)
asr_model = speech2text.asr_model  # already in eval mode

ctc_prefix_scorer = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)

data_loader = get_data_loader(args.dataset)

# build tokenizer
tokenizer = build_tokenizer(
    token_type=speech2text.asr_train_args.token_type,
    bpemodel=speech2text.asr_train_args.bpemodel,
)  # for bpe labels to words mapping

converter = speech2text.converter  # for idx to label mapping

if args.pylasr_recog_args:
    print("Using Pylasr beam search")

    with DatadirWriter(args.output_dir) as writer:
        if args.pylasr_recog_args:
            from pylasr.recognizers.treeAttentionDecoderRecognizerMultiLmXuttBeam import (
                TreeAttentionDecoderRecognizerMultiLmXuttBeam,
            )

            pylasr_recog_args = eval(args.pylasr_recog_args)
            recog_config = SimpleNamespace(**pylasr_recog_args)
            with open("sym", "w") as f:
                f.write("\n".join(asr_model.token_list))
            recog_config.sym = "sym"
            recog_config.sb = "<sos/eos>"
            recog_config.space = "<blank>"  # TODO: how space symbol is used in AED?
            recog_config.device = args.device

            # Trafo external LM
            lm_model = speech2text.lm_model  # required change in espnet asr_inferece.py code
            if lm_model:
                lm_model = lm_model.eval()

            ibest_writer = writer["1best_recog"]

            with torch.no_grad():
                att_decoder = TreeAttentionDecoderRecognizerMultiLmXuttBeam(cfg=recog_config)
                att_decoder(
                    dataloader=data_loader,
                    datawriter=ibest_writer,
                    tokenizer=tokenizer,
                    model=asr_model,
                    lmModel=lm_model,
                )

elif args.returnn_recog_args:
    print("Using pure torch beam search")

    from i6_experiments.users.zeyer.experiments.exp2023_04_25_rf.espnet import (
        get_our_label_scorer_intf,
    )  # ESPnet label scorer
    from i6_experiments.users.zeyer.decoding.beam_search_torch.scorers.length_reward import LengthRewardScorer
    from i6_experiments.users.zeyer.decoding.beam_search_torch.scorers.shallow_fusion import (
        ShallowFusedLabelScorers,
    )

    with DatadirWriter(args.output_dir) as writer:
        returnn_recog_args = eval(args.returnn_recog_args)
        max_seq_len_ratio = returnn_recog_args.pop("max_seq_len_ratio", 1.0)
        len_reward = returnn_recog_args.pop("length_reward", 0.0)
        ctc_weight = returnn_recog_args.pop("ctc_weight", 0.0)
        beam_search_variant = returnn_recog_args.pop("beam_search_variant")
        assert beam_search_variant in [
            "beam_search_v5",  # just like returnn
            "dyn_beam",  # use padding
            "sep_ended",  # use packing
            "sep_ended_keep",  # like pylasr, supports threshold pruning
        ]

        beam_search_func = None
        beam_search_opts = None

        if beam_search_variant == "beam_search_v5":
            from i6_experiments.users.zeyer.decoding.beam_search_torch.beam_search_v5 import (
                beam_search_v5,
                BeamSearchOptsV5,
            )

            beam_search_func = beam_search_v5
            beam_search_opts = BeamSearchOptsV5(
                bos_label=4999,
                eos_label=4999,
                num_labels=5000,
                **returnn_recog_args,
            )
        elif beam_search_variant == "dyn_beam":
            from i6_experiments.users.zeyer.decoding.beam_search_torch.beam_search_dyn_beam_v2 import (
                beam_search_dyn_beam_v2,
                BeamSearchDynBeamOpts,
            )

            beam_search_func = beam_search_dyn_beam_v2
            beam_search_opts = BeamSearchDynBeamOpts(
                bos_label=4999,
                eos_label=4999,
                num_labels=5000,
                **returnn_recog_args,
            )
        elif beam_search_variant == "sep_ended":
            from i6_experiments.users.zeyer.decoding.beam_search_torch.beam_search_sep_ended import (
                beam_search_sep_ended,
                BeamSearchDynBeamOpts,
            )

            beam_search_func = beam_search_sep_ended
            beam_search_opts = BeamSearchDynBeamOpts(
                bos_label=4999,
                eos_label=4999,
                num_labels=5000,
                **returnn_recog_args,
            )
        elif beam_search_variant == "sep_ended_keep":
            from i6_experiments.users.zeyer.decoding.beam_search_torch.beam_search_sep_ended_keep_v6 import (
                beam_search_sep_ended_keep_v6,
                BeamSearchSepEndedKeepOpts,
            )

            beam_search_func = beam_search_sep_ended_keep_v6
            beam_search_opts = BeamSearchSepEndedKeepOpts(
                bos_label=4999,
                eos_label=4999,
                num_labels=5000,
                **returnn_recog_args,
            )

        assert beam_search_func is not None
        assert beam_search_opts is not None

        ibest_writer = writer[f"1best_recog"]

        total_recog_time_in_sec = 0.0
        total_enc_recog_time_in_sec = 0.0
        total_dec_recog_time_in_sec = 0.0
        total_audio_length_in_sec = 0.0

        for keys, batch in data_loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"

            with torch.no_grad():
                start_time = time.perf_counter_ns()
                audio_dur = batch["speech_lengths"].sum().item()
                total_audio_length_in_sec += audio_dur * 0.0625 / 1000
                logging.info("speech length: " + str(audio_dur))  # log start of search
                batch = to_device(batch, device=args.device)
                enc, enc_olens = asr_model.encode(**batch)
                if enc.device.type == "cuda":
                    torch.cuda.synchronize(enc.device)
                enc_end_time = time.perf_counter_ns()

                decoder_label_scorer = get_our_label_scorer_intf(asr_model.decoder, enc=enc, enc_olens=enc_olens)
                label_scorers = {"decoder": (decoder_label_scorer, 1.0)}
                if len_reward:
                    label_scorers["len_reward"] = (LengthRewardScorer(), len_reward)
                if ctc_weight:
                    ctc_label_scorer = get_our_label_scorer_intf(ctc_prefix_scorer, enc=enc, enc_olens=enc_olens)
                    label_scorers["ctc"] = (ctc_label_scorer, ctc_weight)

                label_scorer = ShallowFusedLabelScorers(label_scorers=label_scorers)

                out_individual_seq_scores = {}

                seq_targets, seq_log_scores, out_seq_len = beam_search_func(
                    label_scorer=label_scorer,
                    batch_size=len(keys),
                    max_seq_len=enc_olens * max_seq_len_ratio,
                    device=args.device,
                    opts=beam_search_opts,
                    out_individual_seq_scores=out_individual_seq_scores,
                )  # [B,hyp,L]

                search_end_time = time.perf_counter_ns()

                total_enc_recog_time_in_sec += (enc_end_time - start_time) / 1e9
                total_dec_recog_time_in_sec += (search_end_time - enc_end_time) / 1e9
                total_recog_time_in_sec += (search_end_time - start_time) / 1e9

                assert seq_targets.shape[0] == len(keys), seq_targets.shape
                assert seq_log_scores.shape[0] == len(keys), seq_log_scores.shape

                for i, key in enumerate(keys):
                    best_hyp_index = torch.topk(seq_log_scores, 1, dim=-1).indices[0].item()
                    token_int = seq_targets[i, best_hyp_index, : out_seq_len[i, best_hyp_index]]  # [1, 1, L]
                    token_int = token_int[token_int != 4999]
                    token_int = token_int.tolist()

                    ibest_writer["token_int"][key] = " ".join(map(str, token_int))

                    token = converter.ids2tokens(token_int)
                    ibest_writer["token"][key] = " ".join(token)

                    # tokenize the results
                    text = tokenizer.tokens2text(token)
                    logging.info(f"best hypo: {text}")
                    ibest_writer["text"][key] = text

                    ibest_writer["score"][key] = str(seq_log_scores[i, best_hyp_index].item())

                    for k, v in out_individual_seq_scores.items():
                        ibest_writer[f"{k}_score"][key] = " ".join(str(v[i, best_hyp_index].item()))

        logging.info(f"Total recog time: {total_recog_time_in_sec:.3f} sec")
        logging.info(f"Total enc recog time: {total_enc_recog_time_in_sec:.3f} sec")
        logging.info(f"Total dec recog time: {total_dec_recog_time_in_sec:.3f} sec")
        logging.info(f"Total audio length: {total_audio_length_in_sec:.3f} sec")
        logging.info(f"Overall RTF: {total_recog_time_in_sec / total_audio_length_in_sec:.3f}")
        logging.info(f"Enc RTF: {total_enc_recog_time_in_sec / total_audio_length_in_sec:.3f}")
        logging.info(f"Dec RTF: {total_dec_recog_time_in_sec / total_audio_length_in_sec:.3f}")

else:
    print("Using espnet beam search")
    with DatadirWriter(args.output_dir) as writer:
        for keys, batch in data_loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            results = speech2text(**batch)

            key = keys[0]
            for n, (text, token, token_int, hyp) in zip(range(1, args.nbest + 1), results):
                # Create a directory: outdir/{n}best_recog
                ibest_writer = writer[f"{n}best_recog"]

                # Write the result to each file
                ibest_writer["token"][key] = " ".join(token)
                ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                ibest_writer["score"][key] = str(hyp.score)

                if text is not None:
                    ibest_writer["text"][key] = text
