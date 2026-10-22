"""
End-to-End Unsupervised Neural Machine Translation (UNMT) Pipeline for WMT14 German-English:
  1. Prepares WMT14 News Crawl monolingual corpora (de, en) and newstest2014 (de-en) from $HPCWORK.
  2. Trains independent 300-dim Skip-gram Word2Vec embeddings (sg=1, negative=10, window=10, dim=300)
     on German and English monolingual corpora (no HuBERT / wav2vec 2.0).
  3. Aligns the German and English embedding spaces via SelfLearningEmbeddingAlignmentJob using
     numeral initialization (matching ^[0-9]+$ tokens across vocabularies, Artetxe et al. ACL 2017).
  4. Evaluates the Word-by-Word Cross-Lingual Embedding Translation Baseline on newstest2014 (de->en and en->de)
     using BLEU-4 (ComputeBleuJob).
  5. Trains and evaluates the RETURNN port of UNdreaMT (2-layer BiGRU shared encoder + 2-layer StackedGRU
     attention decoders with input feeding, word-swap denoising autoencoding + on-the-fly greedy backtranslation)
     on newstest2014 (de->en and en->de) with beam search and BLEU-4 scoring.
"""

from typing import Any, Dict, List, Optional, Tuple, Sequence
import os
import copy

from sisyphus import tk
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.search import SearchTakeBestJob
from i6_experiments.common.setups.returnn.datasets.base import MetaDataset
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.schmitt.datasets.hdf import HdfDataset
from i6_experiments.users.zeyer.datasets.score_results import ScoreResult

from .data.common import TrainingDatasets, DatasetSettings
from .data.word2vec import TrainWord2VecJob, GetVocabSizeJob
from .data.wmt14_de_en import (
    PrepareWmt14TestSetJob,
    PrepareWmt14MonolingualNewsCrawlJob,
    DumpTextToReturnnHdfJob,
)
from .bleu_eval import ComputeBleuJob, generic_bleu_score_recog_out
from .eval_word_by_word_baseline import WordByWordCrossLingualEvalJob
from .pipeline import training, forward_single, get_checkpoint
from .config import get_forward_config
from .default_tools import RETURNN_EXE, RETURNN_ROOT
from ..models.self_learning_alignment import SelfLearningEmbeddingAlignmentJob


_raw_hpcwork = os.environ.get("HPCWORK", "/rwthfs/rz/cluster/hpcwork/bpd03090")
if _raw_hpcwork.startswith("/hpcwork/"):
    _raw_hpcwork = "/rwthfs/rz/cluster/hpcwork/" + _raw_hpcwork[len("/hpcwork/") :]
DEFAULT_HPCWORK_WMT14_DIR = os.path.join(_raw_hpcwork, "wmt14_de_en")


def get_wmt14_monolingual_paths(
    wmt14_dir: str = DEFAULT_HPCWORK_WMT14_DIR,
    years: Tuple[int, ...] = (2007, 2008, 2009, 2010, 2011, 2012, 2013),
) -> Tuple[List[tk.Path], List[tk.Path]]:
    """
    Returns lists of tk.Path objects pointing to WMT14 News Crawl monolingual .shuffled.gz files
    in $HPCWORK/wmt14_de_en/monolingual for German ('de') and English ('en').
    """
    de_files = []
    en_files = []
    for y in years:
        de_p = os.path.join(wmt14_dir, "monolingual", f"news.{y}.de.shuffled.gz")
        en_p = os.path.join(wmt14_dir, "monolingual", f"news.{y}.en.shuffled.gz")
        de_files.append(tk.Path(de_p))
        en_files.append(tk.Path(en_p))
    return de_files, en_files


def run_wmt14_de_en_unmt_pipeline(
    prefix_name: str = "exp2026_10_19_unmt/wmt14_de_en",
    train_name: str = "undreamt_rnn_aed",
    wmt14_dir: str = DEFAULT_HPCWORK_WMT14_DIR,
    news_crawl_years: Tuple[int, ...] = (2013,),
    max_mono_sentences: int = 2000000,
    unmt_train_sentences: int = 500000,
    dev_max_sentences: Optional[int] = None,
    vocab_size: int = 50000,
    w2v_dim: int = 300,
    w2v_window: int = 10,
    w2v_negative: int = 10,
    w2v_sg: int = 1,
    w2v_epochs: int = 5,
    num_epochs: int = 20,
    eval_epochs: Optional[Sequence[int]] = None,
    partition_epoch: int = 20,
    batch_size: int = 2500,
    batch_max_seqs: int = 50,
    learning_rate: float = 0.0002,
    beam_size: int = 12,
    lp_alpha: float = 0.6,
    device: str = "gpu",
):
    if eval_epochs is None:
        eval_epochs = [num_epochs]
    eval_epochs = sorted(set(eval_epochs))
    """
    Builds the full Sisyphus graph for WMT14 German-English Unsupervised NMT:
      1. WMT14 newstest2014 & News Crawl monolingual preprocessing
      2. Word2Vec (Skip-gram, negative=10, window=10, dim=300) on DE and EN independently
      3. Self-learning cross-lingual alignment with numeral initialization (^[0-9]+$)
      4. Word-by-word cross-lingual embedding translation baseline + BLEU-4 evaluation
      5. UNdreaMT (BiGRU encoder + StackedGRU attention decoder + denoising + backtranslation)
         RETURNN training + beam search decoding + BLEU-4 evaluation on newstest2014 (de->en & en->de).
    """
    # -------------------------------------------------------------------------
    # 1. Prepare WMT14 newstest2014 (and dev newstest2013) + Monolingual Corpora
    # -------------------------------------------------------------------------
    newstest2014_de_sgm = tk.Path(os.path.join(wmt14_dir, "raw", "test-full", "newstest2014-deen-src.de.sgm"))
    newstest2014_en_sgm = tk.Path(os.path.join(wmt14_dir, "raw", "test-full", "newstest2014-deen-ref.en.sgm"))

    test2014_job = PrepareWmt14TestSetJob(
        src_sgm_or_txt=newstest2014_de_sgm,
        trg_sgm_or_txt=newstest2014_en_sgm,
        src_lang="de",
        trg_lang="en",
        lowercase=False,
    )
    test2014_job.add_alias(f"{prefix_name}/data/prepare_newstest2014")
    tk.register_output(f"{prefix_name}/data/newstest2014.tok.de", test2014_job.out_src_tok)
    tk.register_output(f"{prefix_name}/data/newstest2014.tok.en", test2014_job.out_trg_tok)

    newstest2013_de_txt = tk.Path(os.path.join(wmt14_dir, "raw", "dev", "newstest2013.de"))
    newstest2013_en_txt = tk.Path(os.path.join(wmt14_dir, "raw", "dev", "newstest2013.en"))
    dev2013_job = PrepareWmt14TestSetJob(
        src_sgm_or_txt=newstest2013_de_txt,
        trg_sgm_or_txt=newstest2013_en_txt,
        src_lang="de",
        trg_lang="en",
        lowercase=False,
    )
    dev2013_job.add_alias(f"{prefix_name}/data/prepare_newstest2013")

    de_raw_files, en_raw_files = get_wmt14_monolingual_paths(wmt14_dir=wmt14_dir, years=news_crawl_years)

    mono_de_job = PrepareWmt14MonolingualNewsCrawlJob(
        input_files=de_raw_files,
        lang="de",
        min_tokens=3,
        max_tokens=50,
        max_sentences=max_mono_sentences,
        lowercase=False,
    )
    mono_de_job.add_alias(f"{prefix_name}/data/prepare_mono_de")
    tk.register_output(f"{prefix_name}/data/news_crawl.de.tok.txt", mono_de_job.out_tokenized_txt)

    mono_en_job = PrepareWmt14MonolingualNewsCrawlJob(
        input_files=en_raw_files,
        lang="en",
        min_tokens=3,
        max_tokens=50,
        max_sentences=max_mono_sentences,
        lowercase=False,
    )
    mono_en_job.add_alias(f"{prefix_name}/data/prepare_mono_en")
    tk.register_output(f"{prefix_name}/data/news_crawl.en.tok.txt", mono_en_job.out_tokenized_txt)

    # -------------------------------------------------------------------------
    # 2. Train Independent Word2Vec Models on German and English
    #    (Skip-gram sg=1, negative=10, window=10, vector_size=300)
    # -------------------------------------------------------------------------
    w2v_de_job = TrainWord2VecJob(
        token_text_file=mono_de_job.out_tokenized_txt,
        vector_size=w2v_dim,
        window=w2v_window,
        min_count=5,
        max_vocab_size=vocab_size,
        sg=w2v_sg,
        negative=w2v_negative,
        epochs=w2v_epochs,
        workers=8,
    )
    w2v_de_job.add_alias(f"{prefix_name}/word2vec/de_sg_neg10_win10_d300")
    tk.register_output(f"{prefix_name}/word2vec/de_embeddings.vec", w2v_de_job.out_embeddings_vec)
    tk.register_output(f"{prefix_name}/word2vec/de_vocab.txt", w2v_de_job.out_vocab_txt)

    w2v_en_job = TrainWord2VecJob(
        token_text_file=mono_en_job.out_tokenized_txt,
        vector_size=w2v_dim,
        window=w2v_window,
        min_count=5,
        max_vocab_size=vocab_size,
        sg=w2v_sg,
        negative=w2v_negative,
        epochs=w2v_epochs,
        workers=8,
    )
    w2v_en_job.add_alias(f"{prefix_name}/word2vec/en_sg_neg10_win10_d300")
    tk.register_output(f"{prefix_name}/word2vec/en_embeddings.vec", w2v_en_job.out_embeddings_vec)
    tk.register_output(f"{prefix_name}/word2vec/en_vocab.txt", w2v_en_job.out_vocab_txt)

    # -------------------------------------------------------------------------
    # 3. Cross-Lingual Embedding Alignment with Numeral Initialization (^[0-9]+$)
    # -------------------------------------------------------------------------
    align_job = SelfLearningEmbeddingAlignmentJob(
        src_embeddings_npy=w2v_de_job.out_embeddings_npy,
        trg_embeddings_npy=w2v_en_job.out_embeddings_npy,
        src_vocab_file=w2v_de_job.out_vocab_txt,
        trg_vocab_file=w2v_en_job.out_vocab_txt,
        src_counts_file=w2v_de_job.out_counts_txt,
        trg_counts_file=w2v_en_job.out_counts_txt,
        init_method="numerals",
        max_iters=100,
        tol=1e-6,
        retrieval_metric="cosine",
    )
    align_job.add_alias(f"{prefix_name}/alignment/de_en_numerals_init")
    tk.register_output(f"{prefix_name}/alignment/mapped_de_embeddings.npy", align_job.out_mapped_src_emb_npy)
    tk.register_output(f"{prefix_name}/alignment/normalized_en_embeddings.npy", align_job.out_normalized_trg_emb_npy)
    tk.register_output(f"{prefix_name}/alignment/mapped_de_embeddings.vec", align_job.out_mapped_src_vec_txt)
    tk.register_output(f"{prefix_name}/alignment/normalized_en_embeddings.vec", align_job.out_normalized_trg_vec_txt)
    tk.register_output(f"{prefix_name}/alignment/alignment_report.json", align_job.out_alignment_report)
    tk.register_output(f"{prefix_name}/alignment/de_to_en_dict.json", align_job.out_dictionary_json)

    # -------------------------------------------------------------------------
    # 4. Baseline System: Word-by-Word Translation via Cross-Lingual Embeddings
    #    Evaluated with BLEU-4 on newstest2014 (de->en and en->de)
    # -------------------------------------------------------------------------
    for metric in ("cosine", "csls"):
        # de -> en word-by-word baseline
        wbw_de2en_job = WordByWordCrossLingualEvalJob(
            src_embeddings_npy=align_job.out_mapped_src_emb_npy,
            trg_embeddings_npy=align_job.out_normalized_trg_emb_npy,
            src_vocab_file=w2v_de_job.out_vocab_txt,
            trg_vocab_file=w2v_en_job.out_vocab_txt,
            src_test_txt=test2014_job.out_src_tok,
            trg_ref_txt=test2014_job.out_trg_tok,
            direction="src2trg",
            retrieval_metric=metric,
            csls_k=10,
            copy_oov=True,
        )
        wbw_de2en_job.add_alias(f"{prefix_name}/baseline_word_by_word/de2en_{metric}")
        tk.register_output(f"{prefix_name}/baseline_word_by_word/de2en_{metric}/bleu", wbw_de2en_job.out_bleu)
        tk.register_output(f"{prefix_name}/baseline_word_by_word/de2en_{metric}/report", wbw_de2en_job.out_bleu_report_txt)
        tk.register_output(f"{prefix_name}/baseline_word_by_word/de2en_{metric}/hypotheses.txt", wbw_de2en_job.out_hypotheses_txt)

        # en -> de word-by-word baseline
        wbw_en2de_job = WordByWordCrossLingualEvalJob(
            src_embeddings_npy=align_job.out_mapped_src_emb_npy,
            trg_embeddings_npy=align_job.out_normalized_trg_emb_npy,
            src_vocab_file=w2v_de_job.out_vocab_txt,
            trg_vocab_file=w2v_en_job.out_vocab_txt,
            src_test_txt=test2014_job.out_trg_tok,
            trg_ref_txt=test2014_job.out_src_tok,
            direction="trg2src",
            retrieval_metric=metric,
            csls_k=10,
            copy_oov=True,
        )
        wbw_en2de_job.add_alias(f"{prefix_name}/baseline_word_by_word/en2de_{metric}")
        tk.register_output(f"{prefix_name}/baseline_word_by_word/en2de_{metric}/bleu", wbw_en2de_job.out_bleu)
        tk.register_output(f"{prefix_name}/baseline_word_by_word/en2de_{metric}/report", wbw_en2de_job.out_bleu_report_txt)
        tk.register_output(f"{prefix_name}/baseline_word_by_word/en2de_{metric}/hypotheses.txt", wbw_en2de_job.out_hypotheses_txt)

    total_vocab_dim = vocab_size + 4  # PAD=0, OOV=1, EOS=2, SOS=3, words=4..V+3


    train_de_hdf_job = DumpTextToReturnnHdfJob(
        text_file=mono_de_job.out_tokenized_txt,
        returnn_vocab_file=w2v_de_job.out_returnn_vocab,
        max_sentences=unmt_train_sentences,
        seq_tag_prefix="line",
    )
    train_de_hdf_job.add_alias(f"{prefix_name}/hdf/train_mono_de")

    train_en_hdf_job = DumpTextToReturnnHdfJob(
        text_file=mono_en_job.out_tokenized_txt,
        returnn_vocab_file=w2v_en_job.out_returnn_vocab,
        max_sentences=unmt_train_sentences,
        seq_tag_prefix="line",
    )
    train_en_hdf_job.add_alias(f"{prefix_name}/hdf/train_mono_en")

    dev2013_de_hdf_job = DumpTextToReturnnHdfJob(
        text_file=dev2013_job.out_src_tok,
        returnn_vocab_file=w2v_de_job.out_returnn_vocab,
        max_sentences=dev_max_sentences,
        seq_tag_prefix="line",
    )
    dev2013_en_hdf_job = DumpTextToReturnnHdfJob(
        text_file=dev2013_job.out_trg_tok,
        returnn_vocab_file=w2v_en_job.out_returnn_vocab,
        max_sentences=dev_max_sentences,
        seq_tag_prefix="line",
    )

    test2014_de_hdf_job = DumpTextToReturnnHdfJob(
        text_file=test2014_job.out_src_tok,
        returnn_vocab_file=w2v_de_job.out_returnn_vocab,
        seq_tag_prefix="line",
    )
    test2014_en_hdf_job = DumpTextToReturnnHdfJob(
        text_file=test2014_job.out_trg_tok,
        returnn_vocab_file=w2v_en_job.out_returnn_vocab,
        seq_tag_prefix="line",
    )

    train_dataset = MetaDataset(
        datasets={
            "de_mono": HdfDataset(
                files=[train_de_hdf_job.out_hdf],
                partition_epoch=partition_epoch,
                seq_ordering="random",
            ),
            "en_mono": HdfDataset(
                files=[train_en_hdf_job.out_hdf],
                partition_epoch=partition_epoch,
                seq_ordering="random",
            ),
        },
        data_map={
            "data": ("de_mono", "data"),
            "target": ("en_mono", "data"),
        },
        seq_order_control_dataset="de_mono",
    )

    dev_dataset = MetaDataset(
        datasets={
            "de_dev": HdfDataset(
                files=[dev2013_de_hdf_job.out_hdf],
                partition_epoch=1,
                seq_ordering="sorted_reverse",
            ),
            "en_dev": HdfDataset(
                files=[dev2013_en_hdf_job.out_hdf],
                partition_epoch=1,
                seq_ordering="sorted_reverse",
            ),
        },
        data_map={
            "data": ("de_dev", "data"),
            "target": ("en_dev", "data"),
        },
        seq_order_control_dataset="de_dev",
    )

    test2014_de2en_dataset = MetaDataset(
        datasets={
            "de_test": HdfDataset(files=[test2014_de_hdf_job.out_hdf]),
            "en_test": HdfDataset(files=[test2014_en_hdf_job.out_hdf]),
        },
        data_map={
            "data": ("de_test", "data"),
            "target": ("en_test", "data"),
        },
        seq_order_control_dataset="de_test",
    )

    test2014_en2de_dataset = MetaDataset(
        datasets={
            "en_test": HdfDataset(files=[test2014_en_hdf_job.out_hdf]),
            "de_test": HdfDataset(files=[test2014_de_hdf_job.out_hdf]),
        },
        data_map={
            "data": ("en_test", "data"),
            "target": ("de_test", "data"),
        },
        seq_order_control_dataset="en_test",
    )

    datastreams = {
        "data": LabelDatastream(
            available_for_inference=True,
            vocab=w2v_de_job.out_returnn_vocab,
            vocab_size=total_vocab_dim,
        ),
        "target": LabelDatastream(
            available_for_inference=False,
            vocab=w2v_en_job.out_returnn_vocab,
            vocab_size=total_vocab_dim,
        ),
    }

    training_datasets = TrainingDatasets(
        train=train_dataset,
        eval_datasets={"dev": dev_dataset},
        datastreams=datastreams,
    )

    network_module = "definitions.undreamt_rnn_aed_v1.Model"
    train_step_module = "train_steps.undreamt_backtranslation_train_step.train_step"
    forward_step_module = "recognition.undreamt_mt.forward_step.forward_step"
    callback_module = "recognition.undreamt_mt.callback.RecognitionToTextDictCallback"

    net_args = {
        "src_vocab_size": vocab_size,
        "trg_vocab_size": vocab_size,
        "embedding_size": w2v_dim,
        "hidden_size": 600,
        "layers": 2,
        "dropout": 0.3,
        "param_init": 0.1,
        "src_pretrained_emb_path": align_job.out_mapped_src_emb_npy,
        "trg_pretrained_emb_path": align_job.out_normalized_trg_emb_npy,
        "freeze_encoder_embeddings": True,
        "freeze_decoder_embeddings": False,
        "use_embedding_generator": False,
        "alignment_function": "general",
    }


    train_step_args = {
        "default_data_key": "data",
        "default_target_key": "target",
        "enable_denoising": True,
        "enable_backtranslation": True,
        "denoise_backtranslation": True,
        "bt_max_ratio": 2.0,
        "denoise_scale": 1.0,
        "bt_scale": 1.0,
    }

    returnn_train_config_dict = {
        "default_data_key": "data",
        "default_target_key": "target",
        "batch_size": batch_size,
        "max_seqs": batch_max_seqs,
        "optimizer": {"class": "adam", "epsilon": 1e-8},
        "learning_rate": learning_rate,
        "gradient_clip_global_norm": 5.0,
    }

    train_rqmt: Dict[str, Any] = {"mem_rqmt": 24, "time_rqmt": 12, "cpu_rqmt": 4}
    if device != "gpu":
        train_rqmt["device"] = device

    train_args = {
        "config": returnn_train_config_dict,
        "post_config": {
            "cleanup_old_models": {
                "keep_last_n": 4,
                "keep_best_n": 4,
                "keep": sorted(set(list(eval_epochs) + [num_epochs])),
            }
        },
        "python_prolog": [
            "import os",
            "import sys",
            'os.environ["PYTHONPATH"] = ":".join(sys.path)',
            'os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"',
        ],
        "network_module": network_module,
        "train_step_module": train_step_module,
        "net_args": net_args,
        "train_args": train_step_args,
        "rqmt": train_rqmt,
    }

    undreamt_train_name = f"{prefix_name}/{train_name}"
    train_job = training(
        training_name=undreamt_train_name,
        datasets=training_datasets,
        train_args=train_args,
        num_epochs=num_epochs,
        returnn_exe=RETURNN_EXE,
        returnn_root=RETURNN_ROOT,
    )

    # Evaluate checkpoints on newstest2014 in both directions (de -> en and en -> de)
    for eval_ep in eval_epochs:
        ckpt = get_checkpoint(
            training_name=undreamt_train_name,
            train_job=train_job,
            get_specific_checkpoint=eval_ep,
        )

        for direction_name, src_lang, trg_lang, test_ds, vocab_key, ref_file in [
            ("de2en", "src", "trg", test2014_de2en_dataset, "target", test2014_job.out_trg_tok),
            ("en2de", "trg", "src", test2014_en2de_dataset, "data", test2014_job.out_src_tok),
        ]:
            fwd_cfg = get_forward_config(
                config={
                    "default_data_key": "data",
                    "default_target_key": "target",
                    "batch_size": min(batch_size, 3000),
                    "max_seqs": 32,
                },
                network_module=network_module,
                extra_config=ReturnnConfig(
                    config={},
                    python_prolog=[
                        "import os, sys",
                        'os.environ["PYTHONPATH"] = ":".join(sys.path)',
                    ],
                ),
                net_args=net_args,
                decoder=forward_step_module,
                callback_module=callback_module,
                callback_opts={"include_beam": True},
                decoder_args={
                    "beam_size": beam_size,
                    "lp_alpha": lp_alpha,
                    "max_ratio": 2.0,
                    "src_lang": src_lang,
                    "trg_lang": trg_lang,
                    "input_data_key": "data",
                },
                datastreams=datastreams,
                vocab_key=vocab_key,
            )

            recog_prefix = f"{undreamt_train_name}/recog_{direction_name}_ep{eval_ep}/newstest2014"
            search_out = forward_single(
                prefix_name=recog_prefix,
                returnn_config=fwd_cfg,
                checkpoint=ckpt,
                dataset_dict=test_ds.as_returnn_opts(),
                returnn_exe=RETURNN_EXE,
                returnn_root=RETURNN_ROOT,
                rqmt={"device": "gpu", "gpu": 1, "cpu": 4, "mem": 16, "time": 2},
            )
            best_hyp = SearchTakeBestJob(search_out, output_gzip=True).out_best_search_results
            bleu_job = ComputeBleuJob(
                hyp_file=best_hyp,
                ref_file=ref_file,
                lowercase=False,
                remove_unk=True,
            )
            bleu_job.add_alias(f"{recog_prefix}/bleu_eval")
            tk.register_output(f"{recog_prefix}/bleu", bleu_job.out_bleu)
            tk.register_output(f"{recog_prefix}/bleu_report.txt", bleu_job.out_report)
            tk.register_output(f"{recog_prefix}/hypotheses.txt", bleu_job.out_hyp_txt)

    # -------------------------------------------------------------------------
    # 6. Cross-Lingual Shared Encoder PCA Analysis (German vs. English)
    # -------------------------------------------------------------------------
    from .analysis import analyze_encoder_states

    pca_jobs = analyze_encoder_states(
        config={
            "default_data_key": "data",
            "default_target_key": "target",
            "batch_size": 2500,
            "max_seqs": 32,
        },
        training_name=undreamt_train_name,
        train_job=train_job,
        train_args=train_args,
        train_data=training_datasets,
        test_data_dict={"newstest2014": test2014_de2en_dataset, "newstest2013_dev": dev_dataset},
        checkpoints=sorted(set(list(eval_epochs) + [num_epochs])),
        analysis_name="vis_encoder_pca_de_vs_en",
        cosine_similarity_summary=True,
        src_label="de",
        trg_label="en",
        extra_forward_config=ReturnnConfig(
            config={},
            python_prolog=[
                "import os, sys",
                'os.environ["PYTHONPATH"] = ":".join(sys.path)',
            ],
        ),
        rqmt={"device": "gpu", "gpu": 1, "cpu": 4, "mem": 16, "time": 1},
    )

    return {
        "test2014_job": test2014_job,
        "mono_de_job": mono_de_job,
        "mono_en_job": mono_en_job,
        "w2v_de_job": w2v_de_job,
        "w2v_en_job": w2v_en_job,
        "align_job": align_job,
        "train_job": train_job,
        "pca_jobs": pca_jobs,
    }
