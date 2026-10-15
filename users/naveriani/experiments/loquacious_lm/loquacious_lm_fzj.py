from sisyphus import tk

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.naveriani.jobs.datasets import DownloadUrlsJob
from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_lm_dataset
from i6_experiments.users.zeyer.datasets.loquacious import (
    get_loquacious_text_only_dataset,
    get_loquacious_train_subset_dataset_v2,
    get_vocab_by_str,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import (
    _get_cfg_lrlin_oclr_by_bs_nep_v4,
    config_96gb_bf16_accgrad1,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.lm import lm_model_def, lm_train_def
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc import model_recog_with_recomb
from i6_experiments.users.zeyer.experiments.exp2025_10_21_chunked_ctc import train as train_chunked_ctc
from i6_experiments.users.zeyer.external_models.huggingface import DownloadHuggingFaceRepoJobV2
from i6_experiments.users.zeyer.train_v4 import ModelDefWithCfg, train
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.utils.sis_setup import disable_register_output, get_setup_prefix_for_module

from diffusion_lm_2025.sis_recipe.diff_lm_and_std_lm_comparison import ctc_recog_recomb_labelwise_prior_auto_scale
from diffusion_lm_2025.sis_recipe.diffusion_recog_batched import run_battery


def py():
    prefix = get_setup_prefix_for_module(__name__)
    # the HF repo has only the url lists, the data is on olmo-data.org
    urls = DownloadHuggingFaceRepoJobV2(repo_id="allenai/dolma", repo_type="dataset", include=["urls/*"])
    urls.add_alias(f"{prefix}/data/dolma/urls")
    # natural-language text only (~4.2 TB gz), no code, math or LaTeX
    srcs = [
        "books",
        "c4-filtered",
        "cc_en_head",
        "cc_en_middle",
        "cc_en_tail",
        "cc_news_head",
        "cc_news_middle",
        "cc_news_tail",
        "falcon-refinedweb-filtered",
        "pes2o",
        "reddit",
        "wiki",
        "wikiref_megawika",
    ]
    # https://olmo-data.org/dolma-v1_7/<source>/<file>.json.gz -> dolma/v1_7/<source>/<file>.json.gz
    dolma = DownloadUrlsJob(
        (urls.out_hub_cache_dir, "urls/v1_7.txt"), name="dolma/v1_7", include=[f"*/{s}/*" for s in srcs], strip=1
    )
    dolma.add_alias(f"{prefix}/data/dolma/v1_7")
    tk.register_output(f"{prefix}/data/dolma/v1_7/stats.json", dolma.out_stats)

    # 24L LMs with the Loquacious vocab, 5 full epochs each, with rescoring + first pass on Albert's base AM:
    # on the Loquacious transcripts only (236M words), the reference for the external-text LMs
    loq_text = get_loquacious_text_only_dataset(vocab="spm10k", train_epoch_split=10, train_seq_ordering="random")
    lm = train_lm(f"{prefix}/lm/trafo-n24-d1024-nFullEp5-nEp50-spm10k-ddp4-packed20k-graphc", loq_text, "text", 50)
    std_lm_recogs(f"{prefix}/recog/base/lm-n24", lm.get_last_fixed_epoch())
    # and on the LibriSpeech LM corpus (803M words of books); its seq-len stats jobs are not needed here
    with disable_register_output():
        ls_text = get_librispeech_lm_dataset(
            vocab=get_vocab_by_str("spm10k"), train_epoch_split=20, train_sort_order="random"
        )
    lm_ls = train_lm(
        f"{prefix}/lm/trafo-n24-d1024-lsLm-nFullEp5-nEp100-spm10k-ddp4-packed20k-graphc", ls_text, "data", 100
    )
    std_lm_recogs(f"{prefix}/recog/base/lm-ls-n24", lm_ls.get_last_fixed_epoch())


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
    # LM + label-wise prior (scales tuned there) on all sets, first pass with those scales; GPU forwards 4-GPU batched
    am, task = _base_am()
    prior_ds = get_loquacious_train_subset_dataset_v2(vocab="spm10k")
    run_battery(
        lambda: ctc_recog_recomb_labelwise_prior_auto_scale(
            prefix=prefix,
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
