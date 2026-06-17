"""
Shared lexicon-free recognition drivers for the phon AMs (pHMM and CTC).

Both the pHMM and CTC lexicon-free paths run the same ``LexiconfreeTimesyncBeamSearch`` over a
phoneme-only Bliss lexicon, scored by a phoneme LM (neural Transformer via stateful-ONNX, or a count
KenLM n-gram). The hypothesis is a **phoneme stream**, so the only meaningful metric is the Phoneme
Error Rate (PER) -- the word-level WER returned by :func:`search` is discarded (this matches
:mod:`...results`, which renders lexicon-free rows as PER-only).

The two topologies differ only in name-implied ways, all passed as parameters:

* the phoneme-only lexicon is built from the pHMM lexicon (``[SILENCE]`` at index 0) or the CTC
  lexicon (``[BLANK]`` at index 0, ``special="blank"``) -- both share the same EOW phoneme inventory
  and trailing ``<s>``/``</s>`` indices, so the phoneme LM is index-for-index identical and reused,
* CTC search collapses repeated labels around the blank (``collapse_repeated_labels=True``) and the
  decoder's ``silence_label`` is ``[BLANK]`` instead of the ``[SILENCE]`` default.
"""
import copy
from dataclasses import asdict
from typing import Callable, Iterable, List, Optional

from sisyphus import tk

from .config import get_forward_config
from .default_tools import RETURNN_EXE, RETURNN_ROOT, LIBRASR_WHEEL, kenlm_repo
from .pipeline import NeuralLM, search, compute_per, forward_durations_ctm, PhonemeDurationStatsJob
from .results import add_result, add_duration_result
from .phon_am import (
    DEFAULT_CKPT_LIST,
    AM_FRAME_SHIFT_SECONDS,
    _select_eval_dataset_tuples,
    lm_scale_sweep_dev_other,
    compute_phon_prior,
)
from .pytorch_networks.phmm.decoder.rasr_phmm_lexfree_v1 import DecoderConfig as LexfreeDecoderConfig
from .pytorch_networks.phmm.decoder.rasr_phmm_lexfree_ngram_v1 import DecoderConfig as NgramDecoderConfig
from .pytorch_networks.phmm.decoder.rasr_phmm_lexfree_neural_v1 import DecoderConfig as NeuralDecoderConfig
from .pytorch_networks.phmm.decoder.ctm_lexfree_ngram_v1 import DecoderConfig as CtmNgramDecoderConfig
from .rasr import (
    BuildPhonLexiconfreeLexiconJob,
    CreateLibrasrVenvWithKenLMJob,
    build_lexiconfree_phmm_recognition_config,
    build_lexiconfree_count_recognition_config,
    build_lexiconfree_neural_python_recognition_config,
)
from .storage import get_lm_model

# Absolute imports for the experiments/ namespace-package siblings (no __init__.py at experiments/),
# matching the convention used elsewhere in the setup.
from i6_experiments.users.wu.experiments.posterior_hmm.experiments.lm_phon.export_onnx import (
    ExportStatefulOnnxLMJob,
)
from i6_experiments.users.wu.experiments.posterior_hmm.experiments.lm_phon.trafo import (
    phon_trafo_12x512_baseline,
)
from i6_experiments.users.wu.experiments.posterior_hmm.experiments.lm_phon.count_ngram import (
    build_phon_count_ngram_lm,
)

# AM checkpoints to recognize; defaults to the baseline's full ckpt_list (epochs missing from
# asr_models_by_epoch are skipped).
_ALL_CKPT_EPOCHS = tuple(DEFAULT_CKPT_LIST)


def _scale_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def lexfree_neural_search(
    *,
    am_artifacts: dict,
    source_lexicon: tk.Path,
    per_lexicon: tk.Path,
    lexfree_silence_phoneme: Optional[str] = None,
    decoder_silence_label: Optional[str] = None,
    collapse_repeated_labels: bool = False,
    am_checkpoint_epochs: Iterable[int] = _ALL_CKPT_EPOCHS,
    lm_scales: Iterable[float] = (0.3, 0.5, 0.7),
    beam_sizes: Iterable[int] = (32,),
    score_thresholds: Iterable[float] = (14.0,),
    t_max: int = 1024,
    lm_name: str = "phon_trafo12x512_3ep",
    ensure_lm_func: Callable[[], object] = phon_trafo_12x512_baseline,
    lm_backend: str = "torch_gpu",
    dataset_keys: Iterable[str] = ("dev-other",),
):
    """
    Lexicon-free recognition with the phoneme **Transformer** LM, scored by PER.

    Trains the phoneme Transformer LM (shared by both topologies via content hash) and sweeps
    (lm_scale, beam, score_threshold) x (epoch) on dev-clean / dev-other.

    :param lm_backend: how the LM is scored inside the lexicon-free beam search:
        * ``"torch_gpu"`` (default): a pure-Python torch label scorer running the LM directly on
          the GPU, batched over the beam (``rasr_phmm_lexfree_neural_v1`` /
          ``neural_label_scorer``). The forward job runs on GPU (``use_gpu=True``) so both the AM
          forward and the LM scoring are on the device. This avoids the CPU-only stateful-ONNX
          path, which is the measured bottleneck (``librasr`` here is built without ``MODULE_CUDA``
          and onnxruntime 1.15.1 is CPU-only, so the ONNX LM cannot use the GPU).
        * ``"onnx"``: the legacy stateful-ONNX label scorer (``rasr_phmm_lexfree_v1``); kept for
          reference / A-B. Runs the LM ONNX on the CPU execution provider.
    """
    assert lm_backend in ("torch_gpu", "onnx"), lm_backend
    top_prefix = am_artifacts["prefix_name"]
    model_name = am_artifacts["model_name"]
    # Detailed reports: lexicon-free-search/<model>/neural_<lm>/...  (model name is the subdir).
    search_root = top_prefix + "/lexicon-free-search/" + model_name + "/neural_" + lm_name
    # Shared LM artifacts (independent of the AM model) live under a sibling _lm/ dir.
    lm_artifact_prefix = top_prefix + "/lexicon-free-search/_lm/" + lm_name
    asr_models_by_epoch = am_artifacts["asr_models_by_epoch"]
    eval_dataset_tuples = _select_eval_dataset_tuples(am_artifacts, dataset_keys)
    default_returnn = am_artifacts["default_returnn"]

    # 1. Phoneme-only Bliss lexicon for lexicon-free search (build only; not registered).
    lex_kwargs = {} if lexfree_silence_phoneme is None else {"silence_phoneme": lexfree_silence_phoneme}
    lexfree_lexicon = BuildPhonLexiconfreeLexiconJob(bliss_lexicon=source_lexicon, **lex_kwargs).out_lexicon

    # 2. Train (or reuse) the phoneme Transformer LM and pull the NeuralLM record.
    ensure_lm_func()
    lm_model: NeuralLM = get_lm_model(lm_name)
    assert lm_model.phon_vocab is not None, f"NeuralLM {lm_name!r} has no phon_vocab path"

    # 3. (onnx backend only) Export the LM to the (state-initializer, state-updater, scorer)
    # triplet. The torch_gpu backend loads the checkpoint directly in the label scorer, so no
    # ONNX export is needed.
    onnx_job = None
    if lm_backend == "onnx":
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
            print(f"[lexfree_neural] skipping epoch {epoch} (no asr_model)")
            continue

        for beam_size in beam_sizes:
            for score_threshold in score_thresholds:
                for lm_scale in lm_scales:
                    decoder_kwargs = {} if decoder_silence_label is None else {"silence_label": decoder_silence_label}
                    if lm_backend == "torch_gpu":
                        rasr_cfg = build_lexiconfree_neural_python_recognition_config(
                            lexicon_path=lexfree_lexicon,
                            lm_scale=lm_scale,
                            am_scale=1.0,
                            max_beam_size=beam_size,
                            score_threshold=score_threshold,
                            collapse_repeated_labels=collapse_repeated_labels,
                        )
                        decoder_cfg = NeuralDecoderConfig(
                            rasr_config_file=rasr_cfg,
                            lexicon=lexfree_lexicon,
                            lm_checkpoint=lm_model.checkpoint,
                            lm_net_args=lm_model.net_args,
                            **decoder_kwargs,
                        )
                        decoder_module = "phmm.decoder.rasr_phmm_lexfree_neural_v1"
                        # The torch LM scorer batches the whole beam into one GPU forward per frame;
                        # run the forward job on GPU so both the AM forward and the LM scoring are
                        # on the device. The per-hypothesis KV cache lives on the GPU (a few hundred
                        # MB for beam 32, since blank/loop don't advance the LM history) and is freed
                        # as contexts die; no host-RAM ONNX cache.
                        #
                        # use_gpu=True lands on the default 11 GB GPU partition (search_single does
                        # not expose gpu_mem). The librasr search is per-segment and serial, so the
                        # AM-forward batch only affects AM throughput, not the (dominant) per-frame
                        # LM search -- keep it small so the conformer forward + KV cache fit 11 GB.
                        search_kwargs = dict(
                            forward_config={
                                "num_workers_per_gpu": 0,
                                "torch_dataloader_opts": {"num_workers": 0},
                                "batch_size": 50 * 16000,
                                "max_seqs": 16,
                            },
                            decoder_module=decoder_module,
                            use_gpu=True,
                            mem_rqmt=16,
                        )
                    else:  # "onnx" -- legacy CPU stateful-ONNX scorer
                        rasr_cfg = build_lexiconfree_phmm_recognition_config(
                            lexicon_path=lexfree_lexicon,
                            onnx_state_initializer=onnx_job.out_initializer,
                            onnx_state_updater=onnx_job.out_updater,
                            onnx_scorer=onnx_job.out_scorer,
                            lm_scale=lm_scale,
                            am_scale=1.0,
                            max_beam_size=beam_size,
                            score_threshold=score_threshold,
                            collapse_repeated_labels=collapse_repeated_labels,
                        )
                        decoder_cfg = LexfreeDecoderConfig(
                            rasr_config_file=rasr_cfg,
                            lexicon=lexfree_lexicon,
                            **decoder_kwargs,
                        )
                        # Smaller forward batch than the get_forward_config default (1000*16000):
                        # the AM encoder forward is O(T^2) in attention and runs on CPU here, so a
                        # ~1000 s batch alone is several GB. Combined with the bounded LM scorer
                        # cache (max_cached_score_vectors in rasr.py) this keeps the host-RAM peak
                        # well under mem_rqmt. The librasr search is per-segment and single-threaded,
                        # so a smaller batch only adds minor AM-forward overhead.
                        search_kwargs = dict(
                            forward_config={
                                "num_workers_per_gpu": 0,
                                "torch_dataloader_opts": {"num_workers": 0},
                                "batch_size": 300 * 16000,
                                "max_seqs": 60,
                            },
                            decoder_module="phmm.decoder.rasr_phmm_lexfree_v1",
                            # The stateful-ONNX LM scorer keeps ~25 MB KV-cache hidden states per
                            # cached context (on the CPU execution provider, i.e. host RAM); 24 GB
                            # gives ample headroom above the bounded cache (<=6.4 GB) + AM forward.
                            mem_rqmt=24,
                        )

                    variant = f"lm{_scale_tag(lm_scale)}_beam{beam_size}_st{_scale_tag(score_threshold)}"
                    search_name = search_root + f"/recog_ep{epoch}/" + variant
                    asr_model_copy = copy.deepcopy(asr_model)
                    asr_model_copy.prior_file = None
                    search_jobs, _ = search(
                        search_name,
                        asr_model=asr_model_copy,
                        decoder_args={"config": asdict(decoder_cfg)},
                        test_dataset_tuples=eval_dataset_tuples,
                        **search_kwargs,
                        **default_returnn,
                    )
                    if lm_backend == "torch_gpu":
                        # Request a 24 GB GPU so the recog lands on the gpu_24gb partition where
                        # training reliably runs. The default (no gpu_mem) gpu_11gb pool has flaky
                        # nodes where nvidia-smi + a current driver are present but torch's
                        # ``cuInit()`` fails ("CUDA driver initialization failed", typically a
                        # missing /dev/nvidia-uvm) -> "No GPU device found, but config requested
                        # 'gpu'". settings.py already excludes several gpu_11gb nodes for this.
                        # rqmt (incl. gpu_mem) is NOT hashed, so this does not re-key the jobs.
                        for _sj in search_jobs:
                            _sj.rqmt["gpu_mem"] = 24
                    # PER is the only meaningful metric (phoneme-stream hypothesis); WER is discarded.
                    pers = compute_per(
                        search_name,
                        search_jobs,
                        eval_dataset_tuples,
                        per_lexicon,
                        hyp_is_phonemes=True,
                    )
                    add_result(
                        top_prefix,
                        search_type="lexicon-free",
                        model=model_name,
                        variant=f"neural:{lm_name} {variant}",
                        epoch=epoch,
                        pers={name.split("/")[-1]: per for name, per in pers.items()},
                    )

    return {
        "lexfree_lexicon": lexfree_lexicon,
        "onnx_initializer": onnx_job.out_initializer if onnx_job is not None else None,
        "onnx_updater": onnx_job.out_updater if onnx_job is not None else None,
        "onnx_scorer": onnx_job.out_scorer if onnx_job is not None else None,
    }


def lexfree_count_search(
    *,
    am_artifacts: dict,
    source_lexicon: tk.Path,
    per_lexicon: tk.Path,
    sweep_epoch: int,
    lexfree_silence_phoneme: Optional[str] = None,
    decoder_silence_label: Optional[str] = None,
    collapse_repeated_labels: bool = False,
    orders: Iterable[int] = (6,),
    kenlm_max_order: int = 10,
    pruning: Optional[List[int]] = None,
    lm_scales: Iterable[float] = (0.3, 0.5, 0.7, 1.0),
    prior_scale: float = 0.3,
    beam_size: int = 32,
    score_threshold: float = 14.0,
    dataset_keys: Iterable[str] = ("dev-other",),
    use_eow_phonemes: bool = True,
    compute_durations: bool = False,
    duration_lm_scale: float = 1.0,
    duration_dataset_key: str = "dev-other",
    duration_count_order: Optional[int] = None,
):
    """
    Lexicon-free recognition with a **count** phoneme n-gram (KenLM) LM, scored by PER.

    Focused recognition matching the lexical path: ALWAYS apply the acoustic prior
    (``prior_scale=0.3`` -- the count path decodes the SAME peaky CTC posteriors, so it needs the same
    ``/p(label)`` correction) and sweep the LM scale on dev-other at one converged checkpoint
    (``sweep_epoch``). No no-prior recog and no per-epoch grid. The count LM is identical across
    topologies (same phoneme corpus / vocab) and reused via its content hash.

    :param compute_durations: also run a single fixed-scale decode that emits a phoneme CTM from the
        RASR traceback and aggregate average phoneme / silence durations (ms) into the
        ``duration-analysis`` section of the summary report. The LM scale here is NOT swept
        (``duration_lm_scale``, default 1.0); the durations come from one decode at the converged
        checkpoint on ``duration_dataset_key``. The blank handling follows the topology: a real
        ``[SILENCE]`` (pHMM) is split into leading / trailing / between-words silence, while a CTC
        ``[BLANK]`` (when ``decoder_silence_label == "[BLANK]"``) is folded into the preceding phoneme.
    :param duration_count_order: which count-LM order to use for the duration decode (defaults to the
        first of ``orders``); the durations are an AM property, so the order is essentially arbitrary.
    """
    top_prefix = am_artifacts["prefix_name"]
    model_name = am_artifacts["model_name"]

    # Phoneme-only Bliss lexicon for lexicon-free search (build only; not registered).
    lex_kwargs = {} if lexfree_silence_phoneme is None else {"silence_phoneme": lexfree_silence_phoneme}
    lexfree_lexicon = BuildPhonLexiconfreeLexiconJob(bliss_lexicon=source_lexicon, **lex_kwargs).out_lexicon

    # Dedicated recognition venv with the `kenlm` Python module built with MAX_ORDER set (KenLM
    # defaults to 6 and can't load an order>6 model).
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

    # Acoustic prior for the sweep checkpoint (shared by content hash with the lexical path's prior).
    prior_file = compute_phon_prior(am_artifacts, epoch=sweep_epoch)
    prior_tag = _scale_tag(prior_scale)

    for order in orders:
        count_lm = build_phon_count_ngram_lm(
            prefix=top_prefix + f"/lexicon-free-search/_lm/count_o{order}",
            librispeech_key="train-other-960",
            order=order,
            kenlm_max_order=kenlm_max_order,
            pruning=pruning,
            use_eow_phonemes=use_eow_phonemes,
        )

        def _make_count_recog(lm_scale, _count_lm=count_lm):
            rasr_cfg = build_lexiconfree_count_recognition_config(
                lexicon_path=lexfree_lexicon,
                lm_scale=lm_scale,
                am_scale=1.0,
                max_beam_size=beam_size,
                score_threshold=score_threshold,
                collapse_repeated_labels=collapse_repeated_labels,
            )
            decoder_kwargs = {} if decoder_silence_label is None else {"silence_label": decoder_silence_label}
            decoder_cfg = NgramDecoderConfig(
                rasr_config_file=rasr_cfg,
                lexicon=lexfree_lexicon,
                kenlm_file=_count_lm["binary"],
                prior_file=prior_file,
                prior_scale=prior_scale,
                **decoder_kwargs,
            )
            return {
                "decoder_module": "phmm.decoder.rasr_phmm_lexfree_ngram_v1",
                "decoder_config": decoder_cfg,
                "search_kwargs": {
                    "forward_config": {
                        "num_workers_per_gpu": 0,
                        "torch_dataloader_opts": {"num_workers": 0},
                    }
                },
            }

        lm_scale_sweep_dev_other(
            artifacts=am_artifacts,
            make_recog=_make_count_recog,
            lm_scales=lm_scales,
            epoch=sweep_epoch,
            search_type="lexicon-free",
            variant_prefix=f"count_o{order} prior{prior_tag}",
            variant_suffix=f"_beam{beam_size}_st{_scale_tag(score_threshold)}",
            returnn=kenlm_returnn,
            per_lexicon=per_lexicon,
            hyp_is_phonemes=True,
            report_wer=False,
            dataset_keys=dataset_keys,
        )

    if compute_durations:
        _compute_phon_durations(
            am_artifacts=am_artifacts,
            top_prefix=top_prefix,
            model_name=model_name,
            lexfree_lexicon=lexfree_lexicon,
            kenlm_returnn=kenlm_returnn,
            prior_file=prior_file,
            prior_scale=prior_scale,
            sweep_epoch=sweep_epoch,
            order=duration_count_order if duration_count_order is not None else list(orders)[0],
            kenlm_max_order=kenlm_max_order,
            pruning=pruning,
            use_eow_phonemes=use_eow_phonemes,
            decoder_silence_label=decoder_silence_label,
            collapse_repeated_labels=collapse_repeated_labels,
            beam_size=beam_size,
            score_threshold=score_threshold,
            lm_scale=duration_lm_scale,
            dataset_key=duration_dataset_key,
        )

    return {"lexfree_lexicon": lexfree_lexicon}


def _compute_phon_durations(
    *,
    am_artifacts: dict,
    top_prefix: str,
    model_name: str,
    lexfree_lexicon: tk.Path,
    kenlm_returnn: dict,
    prior_file: tk.Path,
    prior_scale: float,
    sweep_epoch: int,
    order: int,
    kenlm_max_order: int,
    pruning: Optional[List[int]],
    use_eow_phonemes: bool,
    decoder_silence_label: Optional[str],
    collapse_repeated_labels: bool,
    beam_size: int,
    score_threshold: float,
    lm_scale: float,
    dataset_key: str,
):
    """
    One fixed-scale count-LM decode that writes a phoneme CTM from the RASR traceback, then aggregate
    average phoneme / silence durations (ms) into the report's ``duration-analysis`` section.

    Reuses the exact lexfree count search (same prior, beam, threshold, collapse) as the PER sweep,
    only swapping the decoder for the CTM-emitting variant and fixing the LM scale (no sweep).
    """
    asr_model = am_artifacts["asr_models_by_epoch"].get(sweep_epoch)
    if asr_model is None:
        print(f"[_compute_phon_durations] no asr_model for epoch {sweep_epoch}; skipping")
        return

    eval_tuples = _select_eval_dataset_tuples(am_artifacts, (dataset_key,))
    dataset, _ref = eval_tuples[dataset_key]

    count_lm = build_phon_count_ngram_lm(
        prefix=top_prefix + f"/lexicon-free-search/_lm/count_o{order}",
        librispeech_key="train-other-960",
        order=order,
        kenlm_max_order=kenlm_max_order,
        pruning=pruning,
        use_eow_phonemes=use_eow_phonemes,
    )

    rasr_cfg = build_lexiconfree_count_recognition_config(
        lexicon_path=lexfree_lexicon,
        lm_scale=lm_scale,
        am_scale=1.0,
        max_beam_size=beam_size,
        score_threshold=score_threshold,
        collapse_repeated_labels=collapse_repeated_labels,
    )
    # CTC blank ([BLANK]) is folded into the preceding phoneme; the pHMM [SILENCE] is real silence.
    effective_silence = decoder_silence_label or "[SILENCE]"
    fold_blank = effective_silence == "[BLANK]"

    decoder_cfg = CtmNgramDecoderConfig(
        rasr_config_file=rasr_cfg,
        lexicon=lexfree_lexicon,
        kenlm_file=count_lm["binary"],
        prior_file=prior_file,
        prior_scale=prior_scale,
        silence_label=effective_silence,
        frame_shift_seconds=AM_FRAME_SHIFT_SECONDS,
    )
    returnn_config = get_forward_config(
        network_module=asr_model.network_module,
        config={"num_workers_per_gpu": 0, "torch_dataloader_opts": {"num_workers": 0}},
        net_args=asr_model.net_args,
        decoder="phmm.decoder.ctm_lexfree_ngram_v1",
        decoder_args={"config": asdict(decoder_cfg)},
    )

    prefix = f"{top_prefix}/lexicon-free-search/{model_name}/durations/recog_ep{sweep_epoch}/{dataset_key}"
    ctm_file, _fwd = forward_durations_ctm(
        prefix,
        returnn_config,
        checkpoint=asr_model.checkpoint,
        recognition_dataset=dataset,
        returnn_exe=kenlm_returnn["returnn_exe"],
        returnn_root=kenlm_returnn["returnn_root"],
    )
    tk.register_output(prefix + "/durations.ctm", ctm_file)

    stats_job = PhonemeDurationStatsJob(
        ctm_file,
        silence_label=effective_silence,
        use_eow=use_eow_phonemes,
        fold_blank_into_phoneme=fold_blank,
    )
    stats_job.add_alias(prefix + "/duration_stats")
    tk.register_output(prefix + "/durations.json", stats_job.out_stats)

    add_duration_result(
        top_prefix,
        model=model_name,
        epoch=sweep_epoch,
        durations={cat: {dataset_key: var} for cat, var in stats_job.out_vars.items()},
    )
