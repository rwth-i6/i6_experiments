"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/blankfree_train_jobs.py
(``NET_ARGS``, ``_dataset``, ``build_blankfree_train_config``), with the EMC constants of
``sae/emc/emc_train_jobs.py`` and the ``get_model`` deltas the phase-4a configs wrote on top of it
(``configs/config_sae_4a_lexlat_k2_pack_v1._model_args_delta``,
``config_sae_4a_lexlat_k2_prior_ablation_v1._set_prior_schedule``,
``config_sae_4a_lexlat_v2_em_v1.stage1_config``, ``config_sae_4a_lexlat_ladder_v1.rt_train_config``,
``config_sae_4a_prepro_pack_v1._set_seq_order_offset``).

:func:`build_train_config` is the ONE builder of a blank-free EMC training ``ReturnnConfig``.  Its
defaults are the bed's, and every phase-4a arm is the bed plus explicit keyword deltas:

* ``ctrl_20``: the N = 20 schedules only;
* the k2 arms: ``lexlat_k2=lexlat_k2_model_args(...)`` (the default-off ``lexlat_k2_*`` block);
* D15: ``prior_weight_schedule`` (``schedules.phone_trigram_weight_schedule``);
* ``ctrl_20_rc``: ``sil_run_collapse`` (the SIL-run fix of T1.6, SaeBlankfreeModelV1's class comment);
* L2-1 stage 1 (phi-first EM): ``null_recognizer`` / ``freeze_recognizer``, ``adam_betas`` /
  ``adam_eps``, ``random_seed`` / ``random_seed_offset``, a 4-entry tau schedule;
* durinit / durfrz: ``reverse_duration_prior`` + ``reverse_duration_prior_mode`` ("init" / "freeze");
* theta / phi from a checkpoint: ``flat_checkpoint`` (any recognizer-only checkpoint) and
  ``reverse_checkpoint_path`` (a reverse-only checkpoint, e.g. ``ExtractSubmoduleCheckpointJob``
  with ``prefix="reverse."``).

The ``get_model`` keyword ORDER is the one the banked configs were written in (the base arguments,
then ``null_recognizer``, ``freeze_recognizer``, ``reverse_checkpoint_path``, the duration prior, the
``lexlat_k2_*`` block, ``prior_weight_schedule`` and ``sil_run_collapse``), so a preset's ``returnn.config`` diffs
line-for-line against its banked one.  The order does not enter the hash (sisyphus hashes a dict
order-independently).

Cut: the back-translation auxiliary (``lam_bt`` and its four companions) -- off in every in-scope
run; the arguments stay and raise when set -- and the two ``sys.path`` inserts the source configs
carried (the speech-llm source root and the setup dir), which this package does not need.
"""

from typing import Any, Dict, Optional, Sequence

from sisyphus import tk

__all__ = [
    "NET_ARGS",
    "DEFAULT_TEMPERATURE_SCHEDULE",
    "THETA_LEARNING_RATE",
    "PHI_LEARNING_RATE",
    "lexlat_k2_model_args",
    "build_train_config",
]

#: the blank-free recognizer (theta) of the bed: 40 outputs (39 phones + SIL, no blank)
NET_ARGS = dict(
    in_dim=1024, n_out=40, kernel=9, stride=3, n_layers=1, dropout=0.1, batch_norm=30.0, residual=True, bias=False
)

#: the bed's own tau anneal over its four sub-epochs (8 -> 2), the default of
#: :func:`build_train_config`.  The model indexes such a list 1-based and holds its last entry, so a
#: list shorter than the run would silently anneal on another clock; the builder therefore refuses
#: this default for any other number of sub-epochs.
DEFAULT_TEMPERATURE_SCHEDULE = [8.0, 5.04, 3.17, 2.0]

# --- the EMC constants the builder writes (emc_train_jobs.py) -----------------------------------
EMC_LAM_AGG = 0.1
EMC_COUNT_EMA_DECAY = 0.99
EMC_BAND = 25
EMC_PRIOR_WEIGHT = 1.0
EMC_RATE_FD_EPS = 0.25
EMC_RATE_FD_MODE = "central"
#: sub-epoch = train-clean-100 / 4 (~7.1k utts)
EMC_PARTITION_EPOCH = 4
EMC_MAX_SEQS = 128
EMC_BATCH_SIZE_FRAMES = 88_000
#: theta's base learning rate; phi's is PHI_LEARNING_RATE through ``emc_param_groups``' multiplier
THETA_LEARNING_RATE = 1.0e-4
PHI_LEARNING_RATE = 3.0e-3
EMC_ADAM_BETAS = (0.5, 0.98)
EMC_ADAM_EPS = 1e-6
EMC_WEIGHT_DECAY = 0.0
EMC_GRAD_CLIP_GLOBAL_NORM = 5.0
MULTI_PROC_NUM_WORKERS = 6
MULTI_PROC_BUFFER_SEQS = 128
# the cut back-translation auxiliary: its defaults, which are the only accepted values
EMC_LAM_BT = 0.0
EMC_BT_DEPTH = "full"
EMC_BT_RAMP_EPOCHS = 4
EMC_BT_BATCH_SENTS = 128

#: the k2 arm's decoder constants (``config_sae_4a_lexlat_k2_v1``: SEARCH_BEAM / OUTPUT_BEAM /
#: MIN_ACTIVE_STATES) and curriculum (``config_sae_4a_lexlat_pack_v1``: RAMP / LAM_LEX)
LEXLAT_K2_SEARCH_BEAM = 20.0
LEXLAT_K2_OUTPUT_BEAM = 8.0
LEXLAT_K2_MIN_ACTIVE_STATES = 30
LEXLAT_K2_RAMP = 3
LEXLAT_K2_FULL_LAM = 1.0

_PKG = __name__.rsplit(".", 2)[0]  # i6_experiments.users.wu.experiments.unsupervised_asr


def lexlat_k2_model_args(
    *,
    hlg: tk.Path,
    stats: tk.Path,
    resources: tk.Path,
    expected_build: Dict[str, Any],
    max_active: int,
    onset: int,
    ramp: int = LEXLAT_K2_RAMP,
    full_lam: float = LEXLAT_K2_FULL_LAM,
    search_beam: float = LEXLAT_K2_SEARCH_BEAM,
    output_beam: float = LEXLAT_K2_OUTPUT_BEAM,
    min_active_states: int = LEXLAT_K2_MIN_ACTIVE_STATES,
    chunk_seqs: Optional[int] = None,
) -> Dict[str, Any]:
    """The ``get_model`` keyword block of a k2 word-graph arm, in the banked order
    (``config_sae_4a_lexlat_k2_pack_v1._model_args_delta``; ``lexlat_k2_chunk_seqs`` last and only
    when stated, as the ladder wrote it).

    ``hlg`` / ``stats`` are the graph job's ``HLG.pt`` / ``build.json``, ``resources`` the trie
    npz the monitors read the ``<unk>`` word id from, ``expected_build`` the four ``build.json``
    fields (``backoff_loops`` / ``escape`` / ``sil_prob`` / ``theta``) the runtime asserts on the
    graph before a frame is scored.  ``max_active`` has no default: it is the arm's pruning width.
    """
    expected_build = dict(expected_build)
    assert set(expected_build) == {"backoff_loops", "escape", "sil_prob", "theta"}, sorted(expected_build)
    out = {
        "lexlat_k2_hlg": hlg,
        "lexlat_k2_stats": stats,
        "lexlat_k2_resources": resources,
        "lexlat_k2_expected_build": expected_build,
        "lexlat_k2_max_active": int(max_active),
        "lexlat_k2_onset": int(onset),
        "lexlat_k2_ramp": int(ramp),
        "lexlat_k2_full_lam": float(full_lam),
        "lexlat_k2_search_beam": float(search_beam),
        "lexlat_k2_output_beam": float(output_beam),
        "lexlat_k2_min_active_states": int(min_active_states),
    }
    if chunk_seqs is not None:
        out["lexlat_k2_chunk_seqs"] = int(chunk_seqs)
    return out


def _dataset(features, units, original, *, segments, partition, ordering, workers):
    from i6_experiments.common.setups.returnn.datasets.base import MetaDataset
    from i6_experiments.common.setups.returnn.datasets.generic import HDFDataset

    meta = MetaDataset(
        data_map={"features": ("feats", "data"), "units": ("units", "data"), "original_length": ("original", "data")},
        datasets={
            "feats": HDFDataset(
                files=list(features), partition_epoch=partition, seq_ordering=ordering, segment_file=segments
            ),
            "units": HDFDataset(files=list(units)),
            "original": HDFDataset(files=list(original)),
        },
        seq_order_control_dataset="feats",
    ).as_returnn_opts()
    if not workers:
        return meta
    return {
        "class": "MultiProcDataset",
        "dataset": meta,
        "num_workers": MULTI_PROC_NUM_WORKERS,
        "buffer_size": MULTI_PROC_BUFFER_SEQS,
    }


def _seq_order_dataset(dataset: dict) -> dict:
    """The sub-dataset that computes the sequence order of ``dataset`` (``MetaDataset`` delegates it
    to ``seq_order_control_dataset``), asserted rather than assumed."""
    d = dataset
    while d.get("class") == "MultiProcDataset":
        d = d["dataset"]
    if d.get("class") != "MetaDataset":
        return d
    key = d["seq_order_control_dataset"]
    assert key, f"MetaDataset without seq_order_control_dataset: {sorted(d)}"
    return d["datasets"][key]


def _set_seq_order_offset(dataset: dict, offset: int) -> None:
    """Move the SEQUENCE ORDER of a train stream by ``offset``."""
    inner = _seq_order_dataset(dataset)
    assert str(inner.get("seq_ordering", "")).startswith("laplace"), (
        f"the stream's order is {inner.get('seq_ordering')!r}, not laplace: random_seed_offset "
        "would move nothing"
    )
    assert int(inner.get("partition_epoch", 1)) >= 1, inner.get("partition_epoch")
    assert "random_seed_offset" not in inner, inner["random_seed_offset"]
    assert inner.get("fixed_random_seed") is None, inner.get("fixed_random_seed")
    inner["random_seed_offset"] = int(offset)


def build_train_config(
    *,
    train_feature_hdfs: Sequence[tk.Path],
    train_units_hdfs: Sequence[tk.Path],
    train_original_hdfs: Sequence[tk.Path],
    dev_feature_hdfs: Sequence[tk.Path],
    dev_units_hdfs: Sequence[tk.Path],
    dev_original_hdfs: Sequence[tk.Path],
    train_segments: tk.Path,
    dev_segments: tk.Path,
    prior_npz: tk.Path,
    eta_npz: tk.Path,
    flat_checkpoint: tk.Path,
    num_subepochs: int = 4,
    temperature_schedule: Optional[Sequence[float]] = None,
    learning_rates: Optional[Sequence[float]] = None,
    null_recognizer: bool = False,
    freeze_recognizer: bool = False,
    reverse_checkpoint_path: Optional[tk.Path] = None,
    reverse_duration_prior: Optional[tk.Path] = None,
    reverse_duration_prior_mode: Optional[str] = None,
    lexlat_k2: Optional[Dict[str, Any]] = None,
    prior_weight_schedule: Optional[Sequence[float]] = None,
    sil_run_collapse: bool = False,
    adam_betas: Sequence[float] = EMC_ADAM_BETAS,
    adam_eps: float = EMC_ADAM_EPS,
    random_seed: Optional[int] = None,
    random_seed_offset: Optional[int] = None,
    lam_bt: float = EMC_LAM_BT,
    bt_depth: str = EMC_BT_DEPTH,
    bt_ramp_epochs: int = EMC_BT_RAMP_EPOCHS,
    bt_batch_sents: int = EMC_BT_BATCH_SENTS,
    bt_text_phn: Optional[tk.Path] = None,
):
    """The blank-free EMC training ``ReturnnConfig``; the defaults are the banked four-sub-epoch bed's.

    ``num_subepochs`` is the LENGTH of the two per-sub-epoch schedules and must equal the
    ``num_epochs`` of the training job (``jobs.train_arm``): ``learning_rates`` is read as
    ``{sub-epoch: value}`` by RETURNN's learning-rate control and ``temperature_schedule`` by the
    model, and both hold their last entry beyond their end, so a mismatch would silently run part of
    the training on a held constant.  ``temperature_schedule = None`` is the bed's own four-entry
    anneal (:data:`DEFAULT_TEMPERATURE_SCHEDULE`) and is accepted only for a four-sub-epoch run;
    ``learning_rates = None`` is the flat :data:`THETA_LEARNING_RATE` at every sub-epoch.

    Model deltas (each written into ``get_model``'s arguments only when not at its default, so the
    bed's own arguments are untouched):

    :param flat_checkpoint: theta's init (``recognizer_checkpoint_path``): the flat init
        (``init.FlatRecognizerInitJob``) or any recognizer-only checkpoint.
    :param null_recognizer: L2-1 (phi-first EM): theta replaced by the uniform ``log_q``; the model
        requires ``freeze_recognizer=True`` with it.
    :param freeze_recognizer: keep theta out of the optimizer.
    :param reverse_checkpoint_path: phi's init, a reverse-only checkpoint.
    :param reverse_duration_prior: ``prior.json`` of the duration-prior mean job (D17 / A9), with
    :param reverse_duration_prior_mode: ``"init"`` (durinit) or ``"freeze"`` (durfrz).
    :param lexlat_k2: the k2 word-graph block, :func:`lexlat_k2_model_args`.
    :param prior_weight_schedule: D15's per-sub-epoch phone-trigram weight beta (length
        ``num_subepochs``, ``schedules.phone_trigram_weight_schedule``); ``None`` keeps the scalar
        ``prior_weight`` 1.0 at every sub-epoch.
    :param sil_run_collapse: the train lattice reads a SIL run as ONE token (the DP's history is
        rebuilt from the blank-free cfg); ``False`` writes no key and keeps the banked SIL split.

    Optimizer / seed deltas (stage 1 of L2-1 writes all four):

    :param adam_betas: / :param adam_eps: Adam's betas / eps (the bed: (0.5, 0.98) / 1e-6).
    :param random_seed: RETURNN's ``random_seed``; ``None`` writes no key (RETURNN's default 42).
    :param random_seed_offset: the train stream's sequence-order seed, written into the
        ``seq_order_control_dataset`` (``feats``); ``None`` writes no key.

    ``lam_bt`` / ``bt_depth`` / ``bt_ramp_epochs`` / ``bt_batch_sents`` / ``bt_text_phn`` are the
    back-translation auxiliary, which no phase-4a arm runs; any non-default value raises.
    """
    from i6_core.returnn.config import CodeWrapper, ReturnnConfig
    from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
    from i6_experiments.common.setups.serialization import Import, NonhashedCode, PartialImport
    from returnn.util.pprint import pformat

    bt_set = {
        "lam_bt": (float(lam_bt), EMC_LAM_BT),
        "bt_depth": (str(bt_depth), EMC_BT_DEPTH),
        "bt_ramp_epochs": (int(bt_ramp_epochs), EMC_BT_RAMP_EPOCHS),
        "bt_batch_sents": (int(bt_batch_sents), EMC_BT_BATCH_SENTS),
        "bt_text_phn": (bt_text_phn, None),
    }
    bt_changed = sorted(k for k, (v, default) in bt_set.items() if v != default)
    if bt_changed:
        raise ValueError(
            f"{bt_changed} set to a non-default value: the back-translation auxiliary was removed in "
            "the port (no phase-4a run uses it)"
        )

    num_subepochs = int(num_subepochs)
    assert num_subepochs >= 1, f"a run of {num_subepochs} sub-epochs trains nothing"
    if temperature_schedule is None:
        assert num_subepochs == len(DEFAULT_TEMPERATURE_SCHEDULE), (
            f"the bed's default anneal states {len(DEFAULT_TEMPERATURE_SCHEDULE)} temperatures and "
            f"this run has {num_subepochs} sub-epochs; a run on another budget states its own "
            "temperature_schedule, it does not inherit this one held flat"
        )
        temperature_schedule = DEFAULT_TEMPERATURE_SCHEDULE
    temperature_schedule = [float(tau) for tau in temperature_schedule]
    if learning_rates is None:
        learning_rates = [THETA_LEARNING_RATE] * num_subepochs
    learning_rates = [float(lr) for lr in learning_rates]
    assert len(temperature_schedule) == len(learning_rates) == num_subepochs, (
        f"{len(temperature_schedule)} temperatures and {len(learning_rates)} learning rates for "
        f"{num_subepochs} sub-epochs; both schedules are read per sub-epoch and hold their last "
        "entry, so a short list would silently train part of the run on a constant"
    )

    extern_data = {
        "features": {"dim": 1024, "shape": (None, 1024), "dtype": "float16"},
        "units": {"dim": 500, "shape": (None,), "sparse": True, "dtype": "int32"},
        "original_length": {"dim": 1, "shape": (None,), "sparse": True, "dtype": "int32"},
    }
    model_args = {
        "temperature_schedule": temperature_schedule,
        "anchor_weight_schedule": 0.0,
        "lam_agg": EMC_LAM_AGG,
        "count_ema_decay": EMC_COUNT_EMA_DECAY,
        "band": EMC_BAND,
        "prior_weight": EMC_PRIOR_WEIGHT,
        "prior_npz_path": prior_npz,
        "eta_table_path": eta_npz,
        "recognizer_checkpoint_path": flat_checkpoint,
        "lam_rate": 3.0,
        "rate_rho_hz": 9.6619373279,
        "rate_fd_eps": EMC_RATE_FD_EPS,
        "rate_fd_mode": EMC_RATE_FD_MODE,
        "lattice_reduction": "matmul",
        "lattice_checkpoint": 32,
        "lattice_float64": True,
    }
    # --- the model deltas, in the order the banked configs wrote them ----------------------------
    if null_recognizer:
        model_args["null_recognizer"] = True
    if freeze_recognizer:
        model_args["freeze_recognizer"] = True
    if reverse_checkpoint_path is not None:
        model_args["reverse_checkpoint_path"] = reverse_checkpoint_path
    if reverse_duration_prior is not None or reverse_duration_prior_mode is not None:
        assert reverse_duration_prior is not None and reverse_duration_prior_mode in ("init", "freeze"), (
            "reverse_duration_prior needs its prior.json AND a mode in ('init', 'freeze'), got "
            f"{reverse_duration_prior!r} / {reverse_duration_prior_mode!r}"
        )
        model_args["reverse_duration_prior"] = reverse_duration_prior
        model_args["reverse_duration_prior_mode"] = str(reverse_duration_prior_mode)
    if lexlat_k2 is not None:
        assert lexlat_k2 and all(k.startswith("lexlat_k2_") for k in lexlat_k2), sorted(lexlat_k2)
        model_args.update(lexlat_k2)
    if prior_weight_schedule is not None:
        prior_weight_schedule = [float(v) for v in prior_weight_schedule]
        assert len(prior_weight_schedule) == num_subepochs, (
            f"{len(prior_weight_schedule)} prior weights for {num_subepochs} sub-epochs"
        )
        assert all(0.0 <= v <= 1.0 for v in prior_weight_schedule), prior_weight_schedule
        assert model_args["prior_weight"] == 1.0, model_args["prior_weight"]
        model_args["prior_weight_schedule"] = prior_weight_schedule
    if sil_run_collapse:
        model_args["sil_run_collapse"] = True

    ser = Collection(
        serializer_objects=[
            NonhashedCode(f"extern_data = {pformat(extern_data)}\n"),
            PartialImport(
                code_object_path=f"{_PKG}.model.blankfree_model.SaeBlankfreeModelV1",
                unhashed_package_root=f"{_PKG}.model",
                hashed_arguments=model_args,
                unhashed_arguments={},
                import_as="get_model",
            ),
            Import(
                code_object_path=f"{_PKG}.model.train_step.train_step",
                unhashed_package_root=f"{_PKG}.model",
                import_as="train_step",
            ),
            PartialImport(
                code_object_path=f"{_PKG}.model.param_groups.emc_param_groups",
                unhashed_package_root=f"{_PKG}.model",
                hashed_arguments={
                    "recognizer_lr_multiplier": 1.0,
                    "reverse_lr_multiplier": PHI_LEARNING_RATE / THETA_LEARNING_RATE,
                },
                unhashed_arguments={},
                import_as="emc_param_groups",
            ),
        ],
        make_local_package_copy=False,
        packages=None,
    )
    train = _dataset(
        train_feature_hdfs,
        train_units_hdfs,
        train_original_hdfs,
        segments=train_segments,
        partition=EMC_PARTITION_EPOCH,
        ordering="laplace:.1000",
        workers=True,
    )
    dev = _dataset(
        dev_feature_hdfs,
        dev_units_hdfs,
        dev_original_hdfs,
        segments=dev_segments,
        partition=None,
        ordering="sorted",
        workers=False,
    )
    config = {
        "train": train,
        "dev": dev,
        "batching": "random",
        "batch_size": {"features": EMC_BATCH_SIZE_FRAMES},
        "max_seqs": EMC_MAX_SEQS,
        "accum_grad_multiple_step": 1,
        "gradient_clip_global_norm": EMC_GRAD_CLIP_GLOBAL_NORM,
        "optimizer": {
            "class": "adam",
            "betas": [float(b) for b in adam_betas],
            "eps": float(adam_eps),
            "weight_decay": EMC_WEIGHT_DECAY,
            "param_groups_custom": CodeWrapper("emc_param_groups"),
        },
        "learning_rate": THETA_LEARNING_RATE,
        "learning_rates": learning_rates,
        "newbob_multi_num_epochs": EMC_PARTITION_EPOCH,
    }
    if random_seed is not None:
        config["random_seed"] = int(random_seed)
    if random_seed_offset is not None:
        _set_seq_order_offset(config["train"], int(random_seed_offset))
        # the dev set is "sorted": deterministic, unseeded, and the same for every arm
        assert _seq_order_dataset(config["dev"])["seq_ordering"] == "sorted"
    return ReturnnConfig(
        config=config,
        post_config={
            "backend": "torch",
            "cleanup_old_models": {"keep_best_n": 0, "keep_last_n": 1},
            "torch_dataloader_opts": {"num_workers": 1},
            "stop_on_nonfinite_train_score": True,
            "watch_memory": True,
            "log_batch_size": True,
        },
        python_prolog=[ser],
    )
