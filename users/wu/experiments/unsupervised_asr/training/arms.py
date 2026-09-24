"""Ported from speech-llm c49559ce src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/
``config_sae_4a_prepro_pack_v1.py`` (``ctrl_20``), ``config_sae_4a_lexlat_k2_pack_v1.py``
(``k2lat_20_ma3000``), ``config_sae_4a_lexlat_k2_pack3_v1.py`` (``off4_k2lat_20``),
``config_sae_4a_lexlat_k2_prior_ablation_v1.py`` (D15's ramp-out) and
``config_sae_4a_lexlat_k2_ext_v1.py`` (E60).

The phase-4a arm presets.  Each returns an :class:`Arm` (name, ``ReturnnConfig``, sub-epochs, kept
checkpoints); ``jobs.train_arm(**arm.job_kwargs())`` makes its training job, which picks the k2
interpreter by itself for a k2 arm.

Inputs come in as two dicts, so no preset hardcodes a path:

* ``data``: :func:`~.config.build_train_config`'s input keywords -- ``train_feature_hdfs``,
  ``train_units_hdfs``, ``train_original_hdfs``, ``dev_feature_hdfs``, ``dev_units_hdfs``,
  ``dev_original_hdfs``, ``train_segments``, ``dev_segments``, ``prior_npz``, ``eta_npz``,
  ``flat_checkpoint``.  The banked arms' dev set is the train stream's HDFs under the CV segments.
* ``graph``: the k2 word graph -- ``hlg`` (``HLG.pt``), ``stats`` (``build.json``), ``resources``
  (the lexicon npz) and ``expected_build`` (its four ``build.json`` fields).

``num_sub_epochs`` is 20 (the phase's runs) or 60 (E60: one job whose first 20 sub-epochs are the
20-sub-epoch run, then 40 held sub-epochs; ``schedules.phase_schedules``).  The kept checkpoints are
sub-epochs 1 / 4 / 10 / 20, plus 30 / 40 / 50 / 60 at 60.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from i6_core.returnn.config import ReturnnConfig

from .config import build_train_config, lexlat_k2_model_args
from .schedules import PHONE_TRIGRAM_MODES, phase_schedules, phone_trigram_weight_schedule

__all__ = [
    "Arm",
    "KEEP_EPOCHS",
    "K2_ONSET",
    "ctrl_20",
    "ctrl_20_s1",
    "CTRL_20_S1_SEEDS",
    "ctrl_20_rc",
    "k2lat_20_ma3000",
    "off4_k2lat_20",
    "k2_word_lm",
    "ARM_PRESETS",
    "ARM_DATA_KEYS",
    "arm_data",
    "arm_graph",
]

#: the kept checkpoints per run length
KEEP_EPOCHS = {20: (1, 4, 10, 20), 60: (1, 4, 10, 20, 30, 40, 50, 60)}
#: the word graph's on-set sub-epoch of every phase-4a k2 arm (the ramp is 3 sub-epochs to full_lam 1)
K2_ONSET = 8
#: ``build.json``'s ``theta`` (the word-LM pruning threshold of the graph build) per graph kind: the
#: in-house word trigram is unpruned, the official 4-gram graph is pruned at 5.0 (off4 / pack3)
_GRAPH_THETA = {"inhouse": 0.0, "official": 5.0}
_DATA_KEYS = (
    "train_feature_hdfs",
    "train_units_hdfs",
    "train_original_hdfs",
    "dev_feature_hdfs",
    "dev_units_hdfs",
    "dev_original_hdfs",
    "train_segments",
    "dev_segments",
    "prior_npz",
    "eta_npz",
    "flat_checkpoint",
)


@dataclass
class Arm:
    name: str
    returnn_config: ReturnnConfig
    num_epochs: int
    keep_epochs: Tuple[int, ...]

    def job_kwargs(self) -> Dict[str, Any]:
        """The keywords of ``jobs.train_arm``."""
        return dict(
            name=self.name, returnn_config=self.returnn_config, num_epochs=self.num_epochs, keep_epochs=self.keep_epochs
        )


def _check_n(num_sub_epochs: int) -> int:
    n = int(num_sub_epochs)
    if n not in KEEP_EPOCHS:
        raise ValueError(f"the phase-4a arms run {sorted(KEEP_EPOCHS)} sub-epochs, not {n}")
    return n


def _name(base: str, n: int) -> str:
    return base if n == 20 else f"{base}_x{n}"


def _data(data: Dict[str, Any]) -> Dict[str, Any]:
    missing = sorted(set(_DATA_KEYS) - set(data))
    unknown = sorted(set(data) - set(_DATA_KEYS))
    assert not missing and not unknown, f"data dict: missing {missing}, unknown {unknown}"
    return dict(data)


def _graph(graph: Dict[str, Any], kind: str) -> Dict[str, Any]:
    keys = {"hlg", "stats", "resources", "expected_build"}
    assert set(graph) == keys, f"graph dict keys {sorted(graph)}, expected {sorted(keys)}"
    theta = float(graph["expected_build"]["theta"])
    assert theta == _GRAPH_THETA[kind], (
        f"the {kind} word graph is built at theta {_GRAPH_THETA[kind]}, this graph states {theta}"
    )
    return dict(graph)


def ctrl_20(*, data: Dict[str, Any], num_sub_epochs: int = 20) -> Arm:
    """The blank-free control: the bed at the phase's N-sub-epoch schedules, no word graph."""
    n = _check_n(num_sub_epochs)
    tau, lr = phase_schedules(n)
    cfg = build_train_config(**_data(data), num_subepochs=n, temperature_schedule=tau, learning_rates=lr)
    return Arm(_name("ctrl_20", n), cfg, n, KEEP_EPOCHS[n])


#: ``ctrl_20_s1``'s seeds (JUPITER ``SAE_4A_prepro.md`` l.13: "ctrl_20_s1 moves flat_seed 1,
#: random_seed 1, random_seed_offset 1000"): theta's flat init seed, RETURNN's ``random_seed`` and the
#: train stream's sequence-order offset
CTRL_20_S1_SEEDS = {"flat_seed": 1, "random_seed": 1, "random_seed_offset": 1000}


def ctrl_20_s1(*, data: Dict[str, Any], num_sub_epochs: int = 20) -> Arm:
    """The seed replicate of :func:`ctrl_20` (its seed band): the same bed and schedules, with theta's
    flat init at ``FlatRecognizerInitJob(net_args=NET_ARGS, seed=1)`` (replacing
    ``data["flat_checkpoint"]``), ``random_seed`` 1 and ``random_seed_offset`` 1000
    (:data:`CTRL_20_S1_SEEDS`)."""
    from .config import NET_ARGS
    from .init import FlatRecognizerInitJob

    n = _check_n(num_sub_epochs)
    seeds = CTRL_20_S1_SEEDS
    flat = FlatRecognizerInitJob(net_args=NET_ARGS, seed=seeds["flat_seed"])
    flat.add_alias(f"sae/4a/init/flat_s{seeds['flat_seed']}")
    d = _data(data)
    d["flat_checkpoint"] = flat.out_checkpoint
    tau, lr = phase_schedules(n)
    cfg = build_train_config(**d, num_subepochs=n, temperature_schedule=tau, learning_rates=lr,
                             random_seed=seeds["random_seed"], random_seed_offset=seeds["random_seed_offset"])
    return Arm(_name("ctrl_20_s1", n), cfg, n, KEEP_EPOCHS[n])


def ctrl_20_rc(*, data: Dict[str, Any], num_sub_epochs: int = 20) -> Arm:
    """The fixed control: :func:`ctrl_20` plus ``sil_run_collapse`` (the train lattice reads a SIL run
    as one token, T1.6; ``SaeBlankfreeModelV1``'s class comment) and nothing else."""
    n = _check_n(num_sub_epochs)
    tau, lr = phase_schedules(n)
    cfg = build_train_config(**_data(data), num_subepochs=n, temperature_schedule=tau, learning_rates=lr,
                             sil_run_collapse=True)
    return Arm(_name("ctrl_20_rc", n), cfg, n, KEEP_EPOCHS[n])


def _k2_arm(
    base_name: str,
    *,
    data: Dict[str, Any],
    graph: Dict[str, Any],
    graph_kind: str,
    max_active: int,
    phone_trigram: str,
    num_sub_epochs: int,
) -> Arm:
    if phone_trigram not in PHONE_TRIGRAM_MODES:
        raise ValueError(f"phone_trigram is one of {PHONE_TRIGRAM_MODES}, not {phone_trigram!r}")
    n = _check_n(num_sub_epochs)
    g = _graph(graph, graph_kind)
    tau, lr = phase_schedules(n)
    k2 = lexlat_k2_model_args(
        hlg=g["hlg"],
        stats=g["stats"],
        resources=g["resources"],
        expected_build=g["expected_build"],
        max_active=max_active,
        onset=K2_ONSET,
    )
    prior_weight_schedule = None
    if phone_trigram != "full":
        # "full" writes no schedule: the scalar prior_weight 1.0 is what the banked arms ran
        prior_weight_schedule = phone_trigram_weight_schedule(
            phone_trigram,
            n,
            onset=k2["lexlat_k2_onset"],
            ramp=k2["lexlat_k2_ramp"],
            full_lam=k2["lexlat_k2_full_lam"],
        )
    cfg = build_train_config(
        **_data(data),
        num_subepochs=n,
        temperature_schedule=tau,
        learning_rates=lr,
        lexlat_k2=k2,
        prior_weight_schedule=prior_weight_schedule,
    )
    return Arm(_name(base_name, n), cfg, n, KEEP_EPOCHS[n])


def k2lat_20_ma3000(
    *, data: Dict[str, Any], graph: Dict[str, Any], num_sub_epochs: int = 20, phone_trigram: str = "full"
) -> Arm:
    """The banked k2 arm: the IN-HOUSE word-trigram graph (``lmplz -o 3``, theta 0), max_active 3000,
    the phone trigram at full weight.  ``phone_trigram="rampout"`` is D15's ``k2lat_20_ma3000_rp``."""
    base = "k2lat_20_ma3000" if phone_trigram == "full" else f"k2lat_20_ma3000_pt{phone_trigram}"
    return _k2_arm(
        base,
        data=data,
        graph=graph,
        graph_kind="inhouse",
        max_active=3000,
        phone_trigram=phone_trigram,
        num_sub_epochs=num_sub_epochs,
    )


def off4_k2lat_20(
    *, data: Dict[str, Any], graph: Dict[str, Any], num_sub_epochs: int = 20, phone_trigram: str = "full"
) -> Arm:
    """The banked official-LM arm (pack3): the OFFICIAL 4-gram graph (theta 5.0), max_active 1000, the
    phone trigram at full weight."""
    base = "off4_k2lat_20" if phone_trigram == "full" else f"off4_k2lat_20_pt{phone_trigram}"
    return _k2_arm(
        base,
        data=data,
        graph=graph,
        graph_kind="official",
        max_active=1000,
        phone_trigram=phone_trigram,
        num_sub_epochs=num_sub_epochs,
    )


def k2_word_lm(
    *, data: Dict[str, Any], graph: Dict[str, Any], num_sub_epochs: int = 20, phone_trigram: str = "rampout"
) -> Arm:
    """The port's default k2 arm (PORT_WORK design decision 2): the official 4-gram graph (theta 5.0),
    max_active 1000, and the phone trigram ramped OUT as the word graph ramps in (D15's schedule), so
    the word LM is the only text model from the end of the ramp on.  This combination was never run
    as such.  ``phone_trigram="full"`` is ``off4_k2lat_20``'s treatment; ``"off"`` trains with NO LM
    term before the on-set (``schedules.phone_trigram_weight_schedule``)."""
    base = "k2_word_lm" if phone_trigram == "rampout" else f"k2_word_lm_pt{phone_trigram}"
    return _k2_arm(
        base,
        data=data,
        graph=graph,
        graph_kind="official",
        max_active=1000,
        phone_trigram=phone_trigram,
        num_sub_epochs=num_sub_epochs,
    )


#: preset name -> builder
ARM_PRESETS = {
    "ctrl_20": ctrl_20,
    "ctrl_20_s1": ctrl_20_s1,
    "ctrl_20_rc": ctrl_20_rc,
    "k2lat_20_ma3000": k2lat_20_ma3000,
    "off4_k2lat_20": off4_k2lat_20,
    "k2_word_lm": k2_word_lm,
}


# public aliases of helpers used by reverse_model/ (same objects)
ARM_DATA_KEYS = _DATA_KEYS
arm_data = _data
arm_graph = _graph
