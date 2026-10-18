from sisyphus import tk

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import (
    ConformerConvSubsample,
    ConformerEncoder,
    ConformerEncoderLayer,
    ConformerPositionwiseFeedForward,
)
from returnn.tensor import Dim, batch_dim
from returnn_common.datasets_old_2022_10.interface import DatasetConfigStatic

from i6_experiments.users.naveriani.experiments.exp2026_10_05_text_corpora.loquacious_lm import (
    dolma,
    dolma_pool,
    dolma_random,
    gutenberg,
    normalize,
    select,
    selectors,
)
from i6_experiments.users.naveriani.jobs.datasets import HfTextJob, MixTextJob
from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_lm_dataset
from i6_experiments.users.zeyer.datasets.loquacious import (
    get_hf_random_sorted_subset,
    get_hf_text_only,
    get_loquacious_task_raw_v2,
    get_loquacious_text_only_dataset,
    get_loquacious_train_subset_dataset_v2,
    get_vocab_by_str,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import _raw_sample_rate, train_exp
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import (
    _batch_size_factor,
    _get_cfg_lrlin_oclr_by_bs_nep_v4,
    config_96gb_bf16_accgrad1,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.lm import lm_model_def, lm_train_def
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc import model_recog_with_recomb
from i6_experiments.users.zeyer.experiments.exp2025_10_21_chunked_ctc import train as train_chunked_ctc
from i6_experiments.users.zeyer.train_v4 import ModelDefWithCfg, train
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.utils.sis_setup import disable_register_output, get_setup_prefix_for_module

from diffusion_lm_2025.sis_recipe.diff_lm_and_std_lm_comparison import ctc_recog_recomb_labelwise_prior_auto_scale
from diffusion_lm_2025.sis_recipe.diffusion_recog_batched import run_battery


def py():
    prefix = get_setup_prefix_for_module(__name__)
    dolma_ = dolma(prefix)
    tk.register_output(f"{prefix}/data/dolma/v1_7/stats.json", dolma_.out_stats)
    subset, docs = dolma_random(prefix, dolma_)
    tk.register_output(f"{prefix}/data/dolma-random-10x/docs-stats.json", docs.out_stats)
    books = gutenberg(prefix, dolma_)
    tk.register_output(f"{prefix}/data/gutenberg/docs-stats.json", books.out_stats)

    # 24L LMs with the Loquacious vocab, 5 full epochs each, with rescoring + first pass on Albert's base AM.
    # The 4 ranks do not shard the data, each sees a full subepoch: nEp = single-GPU subepochs / 4 (epEq).
    # On the Loquacious transcripts only (236M words), the reference for the external-text LMs
    loq_text = get_loquacious_text_only_dataset(vocab="spm10k", train_epoch_split=10, train_seq_ordering="random")
    lm = train_lm(
        f"{prefix}/lm/trafo-n24-d1024-nFullEp5-50epEq-nEp13-spm10k-ddp4-packed20k-graphc", loq_text, "text", 13
    )
    std_lm_recogs(f"{prefix}/recog/base/lm-n24", lm.get_last_fixed_epoch())
    # and on the LibriSpeech LM corpus (803M words of books); its seq-len stats jobs are not needed here
    with disable_register_output():
        ls_text = get_librispeech_lm_dataset(
            vocab=get_vocab_by_str("spm10k"), train_epoch_split=20, train_sort_order="random"
        )
    lm_ls = train_lm(
        f"{prefix}/lm/trafo-n24-d1024-lsLm-nFullEp5-100epEq-nEp25-spm10k-ddp4-packed20k-graphc", ls_text, "data", 25
    )
    std_lm_recogs(f"{prefix}/recog/base/lm-ls-n24", lm_ls.get_last_fixed_epoch())
    # and on random Dolma documents with 10x the transcript words, normalised like the transcripts (NeMo, on RZ in
    # loquacious_lm.py, job folder copied back). 2.49B words after normalisation, split 60 = ~41M words per subepoch
    norm = normalize(prefix, "dolma-random-10x", docs)
    tk.register_output(f"{prefix}/data/dolma-random-10x/stats.json", norm.out_stats)
    lm_dolma = train_lm(
        f"{prefix}/lm/trafo-n24-d1024-dolmaRandom10x-nFullEp5-300epEq-nEp75-spm10k-ddp4-packed20k-graphc",
        _lm_text(norm.out_texts, split=60),
        "data",
        75,
    )
    std_lm_recogs(f"{prefix}/recog/base/lm-dolma-random10x-n24", lm_dolma.get_last_fixed_epoch())
    # and on all of Project Gutenberg (Dolma books, decontaminated, 3.48B words after normalisation, split 80 =
    # ~44M words per subepoch), same protocol
    norm_books = normalize(prefix, "gutenberg", books)
    tk.register_output(f"{prefix}/data/gutenberg/stats.json", norm_books.out_stats)
    lm_books = train_lm(
        f"{prefix}/lm/trafo-n24-d1024-gutenberg-nFullEp5-400epEq-nEp100-spm10k-ddp4-packed20k-graphc",
        _lm_text(norm_books.out_texts, split=80),
        "data",
        100,
    )
    std_lm_recogs(f"{prefix}/recog/base/lm-gutenberg-n24", lm_books.get_last_fixed_epoch())
    # random web + Gutenberg in the Loquacious book share (LibriHeavy 104.0 of 235.9M words, random lines of both):
    # 10x = same words as random; all = all of random + Gutenberg for the same share (~4.45B words, split 112)
    parts = {"random": (norm.out_texts, norm.out_stats), "gutenberg": (norm_books.out_texts, norm_books.out_stats)}
    shares = {"random": 235.9 - 104.0, "gutenberg": 104.0}
    for name, total, split, n_ep in [("10x", norm.out_stats, 60, 75), ("all", None, 112, 140)]:
        mix = MixTextJob(parts, shares, total_stats=total)
        mix.add_alias(f"{prefix}/data/dolma-random-gutenberg-{name}/mix")
        tk.register_output(f"{prefix}/data/dolma-random-gutenberg-{name}/stats.json", mix.out_stats)
        lm_mix = train_lm(
            f"{prefix}/lm/trafo-n24-d1024-dolmaRandGut{name}-nFullEp5-{5 * split}epEq-nEp{n_ep}-spm10k-ddp4-packed20k-graphc",
            _lm_text(mix.out_texts, split=split),
            "data",
            n_ep,
        )
        std_lm_recogs(f"{prefix}/recog/base/lm-dolma-random-gutenberg-{name}-n24", lm_mix.get_last_fixed_epoch())
    # and on the fastText source selection from the pool, same words as the random set (scored, selected and
    # normalised on RZ, folder copied here; 2.464B words after normalisation, 1 % below random)
    sel = select(prefix, *dolma_pool(prefix, dolma_), selectors(prefix, dolma_, subset), books, docs)
    norm_sel = normalize(prefix, "dolma-selected-10x", sel)
    tk.register_output(f"{prefix}/data/dolma-selected-10x/norm-stats.json", norm_sel.out_stats)
    lm_sel = train_lm(
        f"{prefix}/lm/trafo-n24-d1024-dolmaSelected10x-nFullEp5-300epEq-nEp75-spm10k-ddp4-packed20k-graphc",
        _lm_text(norm_sel.out_texts, split=60),
        "data",
        75,
    )
    std_lm_recogs(f"{prefix}/recog/base/lm-dolma-selected10x-n24", lm_sel.get_last_fixed_epoch())
    # the selected web share (People's Speech, YODAS, CommonVoice, VoxPopuli windows, normalised on RZ) with real
    # Gutenberg books as the book share, same words as random. Enable once the normalised web folder is copied here
    # norm_web = normalize(prefix, "dolma-selected-web-10x", selected_web(prefix, sel))
    # parts = {"selected-web": (norm_web.out_texts, norm_web.out_stats), "gutenberg": (norm_books.out_texts, norm_books.out_stats)}
    # mix = MixTextJob(parts, {"selected-web": 235.9 - 104.0, "gutenberg": 104.0}, total_stats=norm.out_stats)
    # mix.add_alias(f"{prefix}/data/dolma-selected-web-gutenberg-10x/mix")
    # tk.register_output(f"{prefix}/data/dolma-selected-web-gutenberg-10x/stats.json", mix.out_stats)
    # lm_sw = train_lm(
    #     f"{prefix}/lm/trafo-n24-d1024-dolmaSelWebGut10x-nFullEp5-300epEq-nEp75-spm10k-ddp4-packed20k-graphc",
    #     _lm_text(mix.out_texts, split=60),
    #     "data",
    #     75,
    # )
    # std_lm_recogs(f"{prefix}/recog/base/lm-dolma-selected-web-gutenberg-10x-n24", lm_sw.get_last_fixed_epoch())

    for src, sel in selectors(prefix, dolma_, subset).items():
        tk.register_output(f"{prefix}/data/fasttext/{src}/stats.json", sel.out_stats)
    # the pool to select from, scored and selected on RZ (loquacious_lm.py) after copying this folder there
    _, pool_files = dolma_pool(prefix, dolma_)
    tk.register_output(f"{prefix}/data/dolma-pool/files", pool_files.out_dir)


def _lm_text(texts, split):
    # LmDataset over normalised text files with the Loquacious vocab, dev = the Loquacious dev transcripts
    vocab = get_vocab_by_str("spm10k")
    # dev = the same 5000 Loquacious dev transcripts as the transcript LM, as LmDataset text (a MetaDataset around
    # the HF dev dataset crashed at the first dev eval)
    loq_dev = HfTextJob(get_hf_random_sorted_subset(get_hf_text_only(), "dev", take_n=5000)).out_text
    return DatasetConfigStatic(
        extern_data={
            "data": {
                "dim_tags": [batch_dim, Dim(None, name="out-spatial")],
                "sparse_dim": Dim(vocab.get_num_classes(), name="vocab"),
                "vocab": vocab.get_opts(),
            }
        },
        default_input="data",
        default_target="data",
        train_dataset={
            "class": "LmDataset",
            "corpus_file": texts,
            "orth_vocab": vocab.get_opts(),
            "seq_end_symbol": None,
            "unknown_symbol": None,
            "partition_epoch": split,
            "seq_ordering": "random",
        },
        eval_datasets={
            "dev": {
                "class": "LmDataset",
                "corpus_file": loq_dev,
                "orth_vocab": vocab.get_opts(),
                "seq_end_symbol": None,
                "unknown_symbol": None,
            }
        },
    )


def train_lm(name, ds, key, n_ep, num_layers=24, batch=20_000, max_seqs=800):
    # Albert's Loquacious LM (exp2026_05_26_base_fzj get_lm) with fewer layers,
    # in the 4-GPU regime of the tune-20260907 sweep: DDP, packed batch per rank, CUDA graph, random order.
    # The base config keeps targets <= 75 tokens, packing adds a gap of 2 per seq for the BOS/EOS shift
    return train(
        name,
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep),
                "batch_size": None,
                "max_seqs": max_seqs,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
                "__env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
                "torch_distributed": {"reduce_type": "grad_explicit"},
                "__num_processes": 4,
                # per process, sis multiplies by 4: 400 GB and the full node
                "__mem_rqmt": 100,
                "__cpu_rqmt": 72,
                "packed_tensors": {"gap": 2},
                "packed_batch_size": {key: batch},
                "rf_packed_att_fast_paths": True,
                "packed_fallback_allowed": ["scaled_dot_product_attention"],
                # the all-reduce sits between step and optimizer, so the optimizer stays out of the graph
                "torch_cuda_graph": {
                    "batch_size_bound": max_seqs,
                    "dim_capacity": {key: 75},
                    "packed_total_bound": {key: batch + 2 * max_seqs},
                    "warmup_steps": 2,
                    "capture_optimizer": False,
                    "compile": True,
                },
            },
            ["batching"],
        ),
        post_config={"cleanup_old_models": {"keep": [n_ep // 5, n_ep // 2, n_ep], "keep_best_n": 1, "keep_last_n": 1}},
        train_dataset=ds,
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=num_layers,
                    model_dim=1024,
                    pos_enc=None,
                    norm=rf.build_dict(rf.RMSNorm),
                    ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                    decoder_layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                    dropout=0.1,
                    att_dropout=0.1,
                )
            },
        ),
        train_def=lm_train_def,
    )


def std_lm_recogs(prefix, lm):
    # Albert's ctc+lm-v2 cells (exp2025_10_21_chunked_ctc) with another LM: CTC-only beam 64, dev 64-best rescored with
    # LM + label-wise prior (scales tuned there) on all sets, first pass with those scales; GPU forwards 4-GPU batched.
    # Every LM on two AMs: base (recog/base/) and Albert's 40-epoch large model (recog/large/), CTC head of layer 16
    base, task = _base_am()
    prior_ds = get_loquacious_train_subset_dataset_v2(vocab="spm10k")
    for name, am in [("base", base), ("large", _large_am())]:
        run_battery(
            lambda am=am, name=name: ctc_recog_recomb_labelwise_prior_auto_scale(
                prefix=prefix.replace("/recog/base/", f"/recog/{name}/"),
                task=task,
                ctc_model=am,
                extra_config={"aux_loss_layers": [16]},
                lm=lm,
                prior_dataset=prior_ds,
                recog_def=model_recog_with_recomb,
            ),
            time=3,
        )


def _base_am():
    # Albert's Loquacious AED+CTC base (exp2025_10_21_chunked_ctc, CTC-only 7.32 / 8.10) from his recipe call,
    # the training is his finished WQbKY49ztiXt, imported from his FZJ setup
    with disable_register_output():
        exp, task, _ = train_chunked_ctc("base", {})
    return exp.get_last_fixed_epoch(), task


def _large_am():
    # Albert's AED+CTC base-v2-large-nFullEp40.0-nEp1000-totalHours1000k (exp2025_10_04_loquacious, 40 full epochs on
    # Loquacious large, CTC-only 5.66 / 6.26): his exact train call, so the job is his g8UGssIt95KO, copied from RZ
    with disable_register_output():
        exp = train_exp(
            "base-v2-large-nFullEp40.0-nEp1000-totalHours1000k",
            config_96gb_bf16_accgrad1,
            prefix="exp2025_10_04_loquacious/aed/",
            task=get_loquacious_task_raw_v2(vocab="spm10k", subset_name="large", train_epoch_split=25),
            model_config={
                "behavior_version": 24,
                "__serialization_version": 2,
                "enc_build_dict": rf.build_dict(
                    ConformerEncoder,
                    input_layer=rf.build_dict(
                        ConformerConvSubsample,
                        out_dims=[32, 64, 64],
                        filter_sizes=[(3, 3), (3, 3), (3, 3)],
                        pool_sizes=[(1, 2)],
                        strides=[(1, 1), (3, 1), (2, 1)],
                    ),
                    num_layers=16,
                    out_dim=1024,
                    encoder_layer=rf.build_dict(
                        ConformerEncoderLayer,
                        ff=rf.build_dict(
                            ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                        ),
                        num_heads=8,
                    ),
                ),
                "dec_build_dict": rf.build_dict(
                    TransformerDecoder,
                    num_layers=6,
                    model_dim=1024,
                    norm=rf.build_dict(rf.RMSNorm),
                    ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                    layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                ),
                "feature_batch_norm": True,
            },
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep_v4(1000, base_lr=0.5),
                "batch_size": 100_000 * _batch_size_factor,
                "optimizer.weight_decay": 1e-2,
                "accum_grad_multiple_step": 1,
                "aux_loss_layers": [4, 10, 16],
                "dec_aux_loss_layers": [3],
                "max_seq_length_default_target": None,
                "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm10k",
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )
    return exp.get_last_fixed_epoch()
