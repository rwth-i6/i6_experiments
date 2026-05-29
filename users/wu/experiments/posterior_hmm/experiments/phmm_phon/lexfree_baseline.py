"""
Lexicon-free pHMM recognition with a stateful-ONNX phoneme Transformer LM.

Reuses the AM training from `eow_phon_phmm_ls960_base()` and adds the new
recognition path: phoneme-only Bliss lexicon + `LexiconfreeTimesyncBeamSearch`
+ `combine(no-op AM, stateful-onnx LM)`.
"""

import copy
from dataclasses import asdict
from typing import Iterable, List, Optional, Tuple

from sisyphus import tk

from ...default_tools import RETURNN_EXE, RETURNN_ROOT, LIBRASR_WHEEL, kenlm_repo
from ...pipeline import ASRModel, NeuralLM, search
from ...results import add_result
from ...pytorch_networks.phmm.decoder.rasr_phmm_lexfree_v1 import DecoderConfig as LexfreeDecoderConfig
from ...pytorch_networks.phmm.decoder.rasr_phmm_lexfree_ngram_v1 import DecoderConfig as NgramDecoderConfig
from ...rasr import (
    BuildPhonLexiconfreeLexiconJob,
    CreateLibrasrVenvWithKenLMJob,
    build_lexiconfree_phmm_recognition_config,
    build_lexiconfree_count_recognition_config,
)
from ...storage import get_lm_model
from ..lm_phon.count_ngram import build_phon_count_ngram_lm
# NOTE: sibling-package imports across the `experiments/` namespace package
# (no __init__.py) go through the absolute path to avoid relying on Python's
# namespace-package traversal from sisyphus's RecipeFinder, which is the same
# convention used by the rest of the setup (relative `from ...X` only ever
# walks up to the `posterior_hmm/` regular package).
from i6_experiments.users.wu.experiments.posterior_hmm.experiments.lm_phon.export_onnx import (
    ExportStatefulOnnxLMJob,
)
from i6_experiments.users.wu.experiments.posterior_hmm.experiments.lm_phon.trafo import (
    phon_trafo_12x512_baseline,
)
from .baseline import eow_phon_phmm_ls960_base


def _scale_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def lexfree_eow_phon_phmm_ls960(
    *,
    am_checkpoint_epochs: Iterable[int] = (10, 20, 40, 60, 80, 100, 110, 125),  # all baseline ckpt_list epochs
    lm_scales: Iterable[float] = (0.3, 0.5, 0.7),
    beam_sizes: Iterable[int] = (32,),
    score_thresholds: Iterable[float] = (14.0,),
    t_max: int = 1024,
    lm_name: str = "phon_trafo12x512_3ep",
):
    """
    Train the phoneme Transformer LM, export it to the stateful-ONNX triplet,
    and run lexicon-free recognition for each (epoch, lm_scale, beam_size,
    score_threshold) combination on dev-clean / dev-other.

    The trained pHMM AM is fetched from the existing baseline path -- nothing
    in the AM training pipeline changes.
    """
    am_artifacts = eow_phon_phmm_ls960_base()
    top_prefix = am_artifacts["prefix_name"]
    model_name = am_artifacts["model_name"]
    # Detailed reports: lexicon-free-search/<model>/neural_<lm>/...  (model name is the subdir).
    search_root = top_prefix + "/lexicon-free-search/" + model_name + "/neural_" + lm_name
    # Shared LM artifacts (independent of the AM model) live under a sibling _lm/ dir.
    lm_artifact_prefix = top_prefix + "/lexicon-free-search/_lm/" + lm_name
    phmm_lexicon = am_artifacts["phmm_lexicon"]
    asr_models_by_epoch = am_artifacts["asr_models_by_epoch"]
    dev_dataset_tuples = am_artifacts["dev_dataset_tuples"]
    default_returnn = am_artifacts["default_returnn"]

    # 1. Phoneme-only Bliss lexicon for lexicon-free search (build only; not registered).
    lexfree_lex_job = BuildPhonLexiconfreeLexiconJob(bliss_lexicon=phmm_lexicon)
    lexfree_lexicon = lexfree_lex_job.out_lexicon

    # 2. Train (or reuse) the phoneme Transformer LM and pull the NeuralLM record.
    phon_trafo_12x512_baseline()
    lm_model: NeuralLM = get_lm_model(lm_name)
    assert lm_model.phon_vocab is not None, f"NeuralLM {lm_name!r} has no phon_vocab path"

    # 3. Export the LM to the (state-initializer, state-updater, scorer) triplet.
    onnx_job = ExportStatefulOnnxLMJob(
        checkpoint=lm_model.checkpoint,
        net_args=lm_model.net_args,
        vocab=lm_model.phon_vocab,
        bos_token="<s>",
        t_max=t_max,
    )
    onnx_job.add_alias(lm_artifact_prefix + "/export_lm_onnx")
    tk.register_output(lm_artifact_prefix + "/lm_state_initializer.onnx", onnx_job.out_initializer)
    tk.register_output(lm_artifact_prefix + "/lm_state_updater.onnx", onnx_job.out_updater)
    tk.register_output(lm_artifact_prefix + "/lm_scorer.onnx", onnx_job.out_scorer)

    # 4. Sweep (lm_scale, beam, score_threshold) x (epoch).
    for epoch in am_checkpoint_epochs:
        asr_model = asr_models_by_epoch.get(epoch)
        if asr_model is None:
            print(f"[lexfree] skipping epoch {epoch} (no asr_model)")
            continue

        for beam_size in beam_sizes:
            for score_threshold in score_thresholds:
                for lm_scale in lm_scales:
                    rasr_cfg = build_lexiconfree_phmm_recognition_config(
                        lexicon_path=lexfree_lexicon,
                        onnx_state_initializer=onnx_job.out_initializer,
                        onnx_state_updater=onnx_job.out_updater,
                        onnx_scorer=onnx_job.out_scorer,
                        lm_scale=lm_scale,
                        am_scale=1.0,
                        max_beam_size=beam_size,
                        score_threshold=score_threshold,
                    )
                    decoder_cfg = LexfreeDecoderConfig(
                        rasr_config_file=rasr_cfg,
                        lexicon=lexfree_lexicon,
                    )

                    variant = f"lm{_scale_tag(lm_scale)}_beam{beam_size}_st{_scale_tag(score_threshold)}"
                    search_name = search_root + f"/recog_ep{epoch}/" + variant
                    asr_model_copy = copy.deepcopy(asr_model)
                    asr_model_copy.prior_file = None
                    _, wers = search(
                        search_name,
                        forward_config={"num_workers_per_gpu": 0},
                        asr_model=asr_model_copy,
                        decoder_module="phmm.decoder.rasr_phmm_lexfree_v1",
                        decoder_args={"config": asdict(decoder_cfg)},
                        test_dataset_tuples=dev_dataset_tuples,
                        **default_returnn,
                    )
                    add_result(
                        top_prefix,
                        search_type="lexicon-free",
                        model=model_name,
                        variant=f"neural:{lm_name} {variant}",
                        epoch=epoch,
                        wers={name.split("/")[-1]: wer for name, wer in wers.items()},
                    )

    return {
        "lexfree_lexicon": lexfree_lexicon,
        "onnx_initializer": onnx_job.out_initializer,
        "onnx_updater": onnx_job.out_updater,
        "onnx_scorer": onnx_job.out_scorer,
    }


def lexfree_count_eow_phon_phmm_ls960(
    *,
    am_checkpoint_epochs: Iterable[int] = (125,),
    orders: Iterable[int] = (6,),
    kenlm_max_order: int = 10,
    pruning: Optional[List[int]] = None,
    lm_scales: Iterable[float] = (0.3, 0.5, 0.7),
    beam_sizes: Iterable[int] = (32,),
    score_thresholds: Iterable[float] = (14.0,),
):
    """
    Lexicon-free recognition with a **count** phoneme n-gram (KenLM) LM instead of the
    neural Transformer LM. Trains the n-gram on the full LibriSpeech LM corpus and sweeps
    (order, lm_scale, beam, score_threshold) x (epoch) on dev-clean / dev-other.

    Phonemes carry less context per token than words, so the order is higher than a word
    4-gram; 6 is the default (sweep via ``orders``). Reuses the AM from the baseline.
    """
    am_artifacts = eow_phon_phmm_ls960_base()
    top_prefix = am_artifacts["prefix_name"]
    model_name = am_artifacts["model_name"]
    lexfree_root = top_prefix + "/lexicon-free-search/" + model_name
    phmm_lexicon = am_artifacts["phmm_lexicon"]
    asr_models_by_epoch = am_artifacts["asr_models_by_epoch"]
    dev_dataset_tuples = am_artifacts["dev_dataset_tuples"]

    # Phoneme-only Bliss lexicon for lexicon-free search (build only; not registered).
    lexfree_lexicon = BuildPhonLexiconfreeLexiconJob(bliss_lexicon=phmm_lexicon).out_lexicon

    # Dedicated recognition venv with the `kenlm` Python module built with MAX_ORDER set
    # (KenLM defaults to 6 and can't load an order>6 model). Kept separate from the baseline's
    # shared recog venv so this does NOT rehash/rerun the existing recogs.
    kenlm_returnn_exe = CreateLibrasrVenvWithKenLMJob(
        python_exe=RETURNN_EXE,
        librasr_wheel=LIBRASR_WHEEL,
        extra_pip_packages=["ninja", "dm-tree", "h5py"],
        python_wrapper_name="python_with_path",
        track_wheel_contents=True,
        kenlm_repository=kenlm_repo,
        kenlm_max_order=kenlm_max_order,
    ).out_python_bin
    kenlm_returnn = {"returnn_exe": kenlm_returnn_exe, "returnn_root": RETURNN_ROOT}

    for order in orders:
        count_lm = build_phon_count_ngram_lm(
            prefix=top_prefix + f"/lexicon-free-search/_lm/count_o{order}",
            librispeech_key="train-other-960",
            order=order,
            kenlm_max_order=kenlm_max_order,
            pruning=pruning,
        )
        for epoch in am_checkpoint_epochs:
            asr_model = asr_models_by_epoch.get(epoch)
            if asr_model is None:
                print(f"[lexfree_count] skipping epoch {epoch} (no asr_model)")
                continue
            for beam_size in beam_sizes:
                for score_threshold in score_thresholds:
                    for lm_scale in lm_scales:
                        rasr_cfg = build_lexiconfree_count_recognition_config(
                            lexicon_path=lexfree_lexicon,
                            lm_scale=lm_scale,
                            am_scale=1.0,
                            max_beam_size=beam_size,
                            score_threshold=score_threshold,
                        )
                        decoder_cfg = NgramDecoderConfig(
                            rasr_config_file=rasr_cfg,
                            lexicon=lexfree_lexicon,
                            kenlm_file=count_lm["binary"],
                        )
                        variant = f"lm{_scale_tag(lm_scale)}_beam{beam_size}_st{_scale_tag(score_threshold)}"
                        search_name = lexfree_root + f"/count_o{order}/recog_ep{epoch}/" + variant
                        asr_model_copy = copy.deepcopy(asr_model)
                        asr_model_copy.prior_file = None
                        _, wers = search(
                            search_name,
                            forward_config={"num_workers_per_gpu": 0},
                            asr_model=asr_model_copy,
                            decoder_module="phmm.decoder.rasr_phmm_lexfree_ngram_v1",
                            decoder_args={"config": asdict(decoder_cfg)},
                            test_dataset_tuples=dev_dataset_tuples,
                            **kenlm_returnn,
                        )
                        add_result(
                            top_prefix,
                            search_type="lexicon-free",
                            model=model_name,
                            variant=f"count_o{order} {variant}",
                            epoch=epoch,
                            wers={name.split("/")[-1]: wer for name, wer in wers.items()},
                        )

    return {"lexfree_lexicon": lexfree_lexicon}
