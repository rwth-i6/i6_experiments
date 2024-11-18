"""
CTC experiments.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Optional, Union, Sequence, Callable, Dict, Any

import returnn.frontend as rf

from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, TrainDef
from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config

from i6_core.util import uopen

from .configs import _get_cfg_lrlin_oclr_by_bs_nep, _batch_size_factor, config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4, dict_update_deep, post_config
from .model import Model, ctc_model_def
from .training import ctc_training
from .recog_def import model_recog, model_recog_lm

if TYPE_CHECKING:
    from i6_experiments.common.setups import serialization
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints
    from i6_experiments.users.mueller.datasets.task import Task
    from i6_experiments.users.zeyer.datasets.score_results import RecogOutput


_raw_sample_rate = _batch_size_factor * 100  # bs factor is from 10ms frames to raw samples


def py():
    """Sisyphus entry point"""
    # relPosAttDef: Use the default RelPosSelfAttention instead of the Shawn et al 2018 style, old RETURNN way.
    enc_conformer_layer_default = rf.build_dict(
        rf.encoder.conformer.ConformerEncoderLayer,
        ff_activation=rf.build_dict(rf.relu_square),
        num_heads=8,
    )

    # Supervised CTC-baseline
    # vocab = "spm20k"
    # vocab = "char"
    vocab = "bpe128"
    # vocab = "bpe10k"
    use_flashlight = True
    use_greedy = False
    epochs = 500
    self_training_rounds = 1
    test_self_training_on_small_dataset = 0 # TODO remove this parameter
    train_small = True
    with_prior = True
    
    if train_small:
        epochs = 50
    if self_training_rounds > 0:
        self_epochs = 450
    
    decoder_hyperparameters = None
    if use_greedy:
        decoder_hyperparameters = {
            "greedy": True
        }
        greedy_str = "-recog_greedy"
        if with_prior:
            decoder_hyperparameters["prior_weight"] = 0.2
            greedy_str += f"_p{str(decoder_hyperparameters['prior_weight']).replace('.', '')}"
    elif use_flashlight:
        decoder_hyperparameters = {
            "log_add": False,
            "nbest": 1,
            "beam_size": 80,
            "lm_weight": 1.9,
            "use_logsoftmax": True,
            "use_lm": True,
            "use_lexicon": True,
        }
        if with_prior:
            decoder_hyperparameters["prior_weight"] = 0.2
        p0 = f"_p{str(decoder_hyperparameters['prior_weight']).replace('.', '')}" if with_prior else ""
        p1 = "sum" if decoder_hyperparameters['log_add'] else "max"
        p2 = f"n{decoder_hyperparameters['nbest']}"
        p3 = f"b{decoder_hyperparameters['beam_size']}"
        p4 = f"w{str(decoder_hyperparameters['lm_weight']).replace('.', '')}"
        p5 = "_logsoftmax" if decoder_hyperparameters['use_logsoftmax'] else ""
        p6 = "_noLM" if not decoder_hyperparameters['use_lm'] else ""
        p7 = "_noLEX" if not decoder_hyperparameters['use_lexicon'] else ""
        lm_hyperparamters_str = f"{p0}_{p1}_{p2}_{p3}_{p4}{p5}{p6}{p7}"
        
    train_exp(
        f"ctc-baseline" +
        (f"-self_training_{self_training_rounds}" if self_training_rounds > 0 else "") + (f"-dataset_size_{test_self_training_on_small_dataset}" if test_self_training_on_small_dataset > 0 else "") +
        (f"-ds100h" if train_small else "") +
        f"-{vocab}" + (greedy_str if use_greedy else (("-recog_lm" + lm_hyperparamters_str) if use_flashlight else "-recog_albert")),
        config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
        model_recog_lm if (use_flashlight or use_greedy) else model_recog,
        decoder_hyperparameters=decoder_hyperparameters,
        model_config={"enc_conformer_layer": enc_conformer_layer_default, "feature_batch_norm": True},
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(15_000, epochs),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "max_seq_length_default_target": None,
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        config_updates_self_training={
            **_get_cfg_lrlin_oclr_by_bs_nep(15_000, self_epochs),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "max_seq_length_default_target": None,
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        } if self_training_rounds > 0 else None,
        vocab=vocab,
        self_training_rounds=self_training_rounds,
        train_small=train_small,
        test_self_training_on_small_dataset=test_self_training_on_small_dataset,
        with_prior=with_prior,
    )
    

_train_experiments: Dict[str, ModelWithCheckpoints] = {}


# noinspection PyShadowingNames
def train_exp(
    name: str,
    config: Dict[str, Any],
    decoder_def: Callable,
    *,
    decoder_hyperparameters: dict = None,
    model_def: Optional[Union[ModelDefWithCfg, ModelDef[Model]]] = None,
    vocab: str = "bpe10k",
    train_vocab_opts: Optional[Dict[str, Any]] = None,
    train_def: Optional[TrainDef[Model]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    config_updates: Optional[Dict[str, Any]] = None,
    config_updates_self_training: Optional[Dict[str, Any]] = None,
    config_deletes: Optional[Sequence[str]] = None,
    post_config_updates: Optional[Dict[str, Any]] = None,
    epilog: Sequence[serialization.SerializerObject] = (),
    num_epochs: int = 2000,
    gpu_mem: Optional[int] = 24,
    num_processes: Optional[int] = None,
    time_rqmt: Optional[int] = None,  # set this to 1 or below to get the fast test queue
    env_updates: Optional[Dict[str, str]] = None,
    enabled: bool = True,
    self_training_rounds: int = 0,
    train_small: bool = False,
    test_self_training_on_small_dataset: int = 0,
    with_prior: bool = False,
) -> Optional[ModelWithCheckpoints]:
    """
    Train experiment
    """
    from i6_experiments.users.mueller.train import train
    from i6_experiments.users.mueller.recog import recog_training_exp
    from i6_experiments.users.mueller.datasets.librispeech import get_librispeech_task_raw_v2, TrainDatasetSel

    print("Job Name:", name)
    if not enabled:
        return None

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = _sis_prefix + "/" + name
    
    task, pseudo_labels_ds = get_librispeech_task_raw_v2(
        vocab=vocab,
        train_vocab_opts=train_vocab_opts,
        save_pseudo_labels = self_training_rounds > 0,
        test_self_training_on_small_dataset = test_self_training_on_small_dataset,
        ds_sel = TrainDatasetSel.train_100h if train_small else TrainDatasetSel.train_960h,
        with_prior=with_prior
    )
    config = config.copy()
    config = dict_update_deep(config, config_updates, config_deletes)
    # This logic is also in train(), but keep it here because it would break the hash because of _RecogAndScoreFunc...
    if "__train_audio_preprocess" in config:
        task: Task = copy.copy(task)
        task.train_dataset = copy.copy(task.train_dataset)
        task.train_dataset.train_audio_preprocess = config.pop("__train_audio_preprocess")

    if not model_def:
        model_def = ctc_model_def
    if model_config:
        model_def = ModelDefWithCfg(model_def, model_config)
    if not train_def:
        train_def = ctc_training
    model_with_checkpoint = train(
        prefix,
        task=task,
        config=config,
        post_config=dict_update_deep(post_config, post_config_updates),
        epilog=epilog,
        model_def=model_def,
        train_def=train_def,
        num_epochs=num_epochs,
        gpu_mem=gpu_mem,
        num_processes=num_processes,
        time_rqmt=time_rqmt,
    )
    train_job = model_with_checkpoint.get_training_job()
    if env_updates:
        for k, v in env_updates.items():
            train_job.set_env(k, v)

    recog_post_proc_funcs = []
    if config.get("use_eos_postfix", False):
        recog_post_proc_funcs.append(_remove_eos_label_v2)
    pseudo_label_path_dict, best_epoch_path = recog_training_exp(
        prefix,
        task,
        model_with_checkpoint,
        recog_def=decoder_def,
        decoder_hyperparameters=decoder_hyperparameters,
        save_pseudo_labels=pseudo_labels_ds,
        recog_post_proc_funcs=recog_post_proc_funcs
    )
    
    # Do self training on pseudo labels
    for i in range(self_training_rounds):
        assert pseudo_label_path_dict is not None, "Pseudo label path is not set"
        assert best_epoch_path is not None, "Best epoch path is not set"
        prefix_self_training = prefix + f"/self-training-{i+1}"
        task, _ = get_librispeech_task_raw_v2(
            vocab=vocab,
            train_vocab_opts=train_vocab_opts,
            ds_sel = TrainDatasetSel.train_860h if train_small else TrainDatasetSel.train_960h,
            with_prior=with_prior,
            pseudo_label_path = pseudo_label_path_dict
        )
        
        config_self = config.copy()
        config_self = dict_update_deep(config_self, config_updates_self_training)
        # This logic is also in train(), but keep it here because it would break the hash because of _RecogAndScoreFunc...
        if "__train_audio_preprocess" in config_self:
            task: Task = copy.copy(task)
            task.train_dataset = copy.copy(task.train_dataset)
            task.train_dataset.train_audio_preprocess = config_self.pop("__train_audio_preprocess")
            
        d = eval(uopen(best_epoch_path, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict), "Has to be a dict containing the best epoch during scoring."
        best_epoch = d["best_epoch"]

        # TODO whole training has to be adapted for sum criterion
        model_with_checkpoint = train(
            prefix_self_training,
            task=task,
            config=config_self,
            post_config=dict_update_deep(post_config, post_config_updates),
            epilog=epilog,
            model_def=model_def,
            train_def=train_def,
            init_params=model_with_checkpoint.get_epoch(best_epoch).checkpoint,
            num_epochs=num_epochs,
            gpu_mem=gpu_mem,
            num_processes=num_processes,
            time_rqmt=time_rqmt,
        )
        train_job = model_with_checkpoint.get_training_job()
        if env_updates:
            for k, v in env_updates.items():
                train_job.set_env(k, v)
        
        pseudo_label_path_dict, best_epoch_path = recog_training_exp(
            prefix_self_training,
            task,
            model_with_checkpoint,
            recog_def=decoder_def,
            decoder_hyperparameters=decoder_hyperparameters,
            save_pseudo_labels=None if i+1 == self_training_rounds else pseudo_labels_ds,
            recog_post_proc_funcs=recog_post_proc_funcs,
        )

    _train_experiments[name] = model_with_checkpoint
    return model_with_checkpoint


def _remove_eos_label_v2(res: RecogOutput) -> RecogOutput:
    from i6_experiments.users.zeyer.datasets.score_results import RecogOutput
    from i6_core.returnn.search import SearchRemoveLabelJob

    return RecogOutput(SearchRemoveLabelJob(res.output, remove_label="</s>", output_gzip=True).out_search_results)


_sis_prefix: Optional[str] = None


def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name
