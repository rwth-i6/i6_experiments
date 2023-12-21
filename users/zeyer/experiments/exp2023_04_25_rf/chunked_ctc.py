"""
Chunked Attention-based Encoder-Decoder Model for Streaming Speech Recognition
https://arxiv.org/abs/2309.08436
"""


from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Any, Union, Tuple, Dict, Sequence, List
import math
import numpy as np
import hashlib

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from .chunked_conformer import ChunkedConformerEncoder, ConformerConvSubsample

from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.lr_schedules.piecewise_linear import dyn_lr_piecewise_linear

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints, ModelWithCheckpoint


# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    _sis_setup_global_prefix(prefix_name)

    # train_exp("chunk-C20-R15-H2-bs15k", config_24gb)
    train_exp("chunk-C20-R15-H2-bs22k", config_24gb, config_updates=_cfg_bs22k)

    train_exp(
        "chunk-C20-R15-H2-11gb-f32-bs8k-lrlin1e_5_562k-accgrad2-mgpu4-p100",
        config_24gb,
        config_updates={
            "__gpu_mem": 11,
            "batch_size": 8_000 * _batch_size_factor,
            "accum_grad_multiple_step": 2,
            **_cfg_mgpu4_p100,
            # ~2500 steps/ep -> 1.250k steps/500ep
            "learning_rate_piecewise_steps": [562_000, 1_125_000, 1_200_000],
        },
        config_deletes=["torch_amp"],  # f32
    )


_sis_prefix: Optional[str] = None


def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from .sis_setup import get_prefix_for_config

        prefix_name = get_prefix_for_config(__file__)
    global _sis_prefix
    _sis_prefix = prefix_name


def _recog(name: str, model_with_checkpoint: ModelWithCheckpoint, config: Optional[Dict[str, Any]] = None):
    from sisyphus import tk
    from i6_experiments.users.zeyer.recog import recog_model

    task = _get_ls_task()

    res = recog_model(task, model_with_checkpoint, model_recog_v2, config=config)
    tk.register_output(_sis_prefix + "/" + name, res.output)


# noinspection PyShadowingNames
def train_exp(
    name: str,
    config: Dict[str, Any],
    *,
    config_updates: Optional[Dict[str, Any]] = None,
    config_deletes: Optional[Sequence[str]] = None,
    num_epochs: int = 2000,
    gpu_mem: Optional[int] = 24,
    num_processes: Optional[int] = None,
    fine_tune: Optional[Union[int, List[Tuple[int, Dict[str, Any]]]]] = None,
    time_rqmt: Optional[int] = None,
) -> ModelWithCheckpoints:
    """
    Train experiment
    """
    from .train import train
    from i6_experiments.users.zeyer.recog import recog_training_exp

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = _sis_prefix + "/" + name
    task = _get_ls_task()
    config = config.copy()
    config = dict_update_deep(config, config_updates, config_deletes)
    if "__num_epochs" in config:
        num_epochs = config.pop("__num_epochs")
    if "__gpu_mem" in config:
        gpu_mem = config.pop("__gpu_mem")
    if "__num_processes" in config:
        num_processes = config.pop("__num_processes")

    model_with_checkpoint = train(
        prefix,
        task=task,
        config=config,
        post_config=post_config,
        model_def=from_scratch_model_def,
        train_def=from_scratch_training,
        num_epochs=num_epochs,
        gpu_mem=gpu_mem,
        num_processes=num_processes,
        distributed_launch_cmd="torchrun" if num_processes else "mpirun",
        time_rqmt=time_rqmt,
    )
    recog_training_exp(
        prefix,
        task,
        model_with_checkpoint,
        recog_def=model_recog_v2,
        search_config={"chunk_opts": config["chunk_opts"]},
    )

    if fine_tune:
        if isinstance(fine_tune, int):
            fine_tune = [(fine_tune, {})]
        for ep, opts in fine_tune:
            assert isinstance(ep, int) and isinstance(opts, dict)
            suffix = f"/finetune/{ep}"
            opts = opts.copy()
            if opts:
                for k, v in sorted(opts.items()):
                    k: str
                    suffix += "-" + k.lstrip("_")
                    v = str(v).replace("-", "_")
                    if len(v) > 16 and not k.startswith("_"):
                        suffix += "_" + hashlib.md5(v.encode("utf8")).hexdigest()[:8]
                    else:
                        suffix += v
            num_epochs_ = opts.pop("num_epochs", 50)
            config_ = config.copy()
            config_["import_model_train_epoch1"] = model_with_checkpoint.get_epoch(ep).checkpoint
            config_.pop("dynamic_learning_rate")
            lrs = opts.pop("learning_rates", None)
            if lrs is None:
                lr_decay_type = opts.pop("lr_decay_type", "geomspace")  # geomspace or linspace
                lr_decay_func = getattr(np, lr_decay_type)
                lr = config_["learning_rate"]
                final_lr = opts.pop("final_lr", 1e-7)
                lrs = list(lr_decay_func(lr, final_lr, num=num_epochs_))
            else:
                assert isinstance(lrs, (list, tuple))
                assert len(lrs) == num_epochs_
            config_["learning_rates"] = lrs
            config_["learning_rate"] = float(lrs[-1])
            config_["specaugment_steps"] = (0, 0, 0)
            config_.update({k: v for k, v in opts.items() if not k.startswith("_")})

            finetune_model_with_ckpt = train(
                prefix + suffix,
                task=task,
                config=config_,
                post_config=post_config,
                model_def=from_scratch_model_def,
                train_def=from_scratch_training,
                num_epochs=num_epochs_,
                gpu_mem=gpu_mem,
            )
            # _recog(name + suffix + "/recog/last", finetune_model_with_ckpt.get_last_fixed_epoch())
            recog_training_exp(prefix + suffix, task, finetune_model_with_ckpt, recog_def=model_recog_v2)

    return model_with_checkpoint


_ls_task = None


def _get_ls_task():
    global _ls_task
    if _ls_task:
        return _ls_task

    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_bpe10k_raw

    _ls_task = get_librispeech_task_bpe10k_raw(with_eos_postfix=True)
    return _ls_task


py = sis_run_with_prefix  # if run directly via `sis m ...`

_batch_size_factor = 160

config_24gb = dict(
    __gpu_mem=24,
    batching="laplace:.1000",
    torch_amp="bfloat16",
    grad_scaler=None,
    batch_size=15_000 * _batch_size_factor,
    accum_grad_multiple_step=2,
    max_seqs=200,
    max_seq_length_default_target=75,
    specaugment_steps=(5_000, 15_000, 25_000),
    optimizer={
        "class": "adamw",
        "epsilon": 1e-16,
        "weight_decay": 1e-6,
        "weight_decay_modules_blacklist": [
            "rf.Embedding",
            "rf.LearnedRelativePositionalEncoding",
        ],
    },
    gradient_clip_global_norm=5.0,
    learning_rate=1.0,
    dynamic_learning_rate=dyn_lr_piecewise_linear,
    # total steps after 2000 epochs: ~2608k
    learning_rate_piecewise_steps=[295_000 * 4, 590_000 * 4, 652_000 * 4],
    learning_rate_piecewise_values=[1e-5, 1e-3, 1e-5, 1e-6],
    aux_loss_layers=[4, 8],
    chunk_opts=dict(
        chunk_stride=120,
        chunk_history=2,
        input_chunk_size=210,
        end_chunk_size=20,
    ),
)
post_config = dict(
    cleanup_old_models=dict(keep_last_n=5),
    torch_dataloader_opts=dict(num_workers=1),
    # https://github.com/rwth-i6/returnn/issues/1478
    reset_dev_memory_caches=True,
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
)

_cfg_lrlin1e_5_295k = {  # for bs15k, mgpu4
    "learning_rate": 1.0,
    "dynamic_learning_rate": dyn_lr_piecewise_linear,
    # total steps after 500 epochs: ~652k
    "learning_rate_piecewise_steps": [295_000, 590_000, 652_000],
    "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
}

_cfg_bs22k = {
    "batch_size": 22_000 * _batch_size_factor,
    # total steps after 2000 epochs: bs15k: ~2608k, bs30k: ~1305k, est: bs22k: ~2000k
    "learning_rate_piecewise_steps": [900_000, 1_800_000, 1_999_000],
}

_cfg_mgpu4_p100 = {
    "__num_processes": 4,  # multi-GPU
    "__num_epochs": 500,  # because of multi-GPU, 1 subepoch here is like 4 subepochs in single-GPU
    "torch_distributed": {"reduce_type": "param", "param_sync_step": 100},  # multi-GPU
}


class MakeModel:
    """for import"""

    def __init__(self, in_dim: int, target_dim: int, *, eos_label: int = 0, num_enc_layers: int = 12):
        self.in_dim = in_dim
        self.target_dim = target_dim
        self.eos_label = eos_label
        self.num_enc_layers = num_enc_layers

    def __call__(self) -> Model:
        from returnn.datasets.util.vocabulary import Vocabulary

        in_dim = Dim(name="in", dimension=self.in_dim, kind=Dim.Types.Feature)
        target_dim = Dim(name="target", dimension=self.target_dim, kind=Dim.Types.Feature)
        target_dim.vocab = Vocabulary.create_vocab_from_labels(
            [str(i) for i in range(target_dim.dimension)], eos_label=self.eos_label
        )

        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)

        return self.make_model(
            in_dim,
            target_dim,
            num_enc_layers=self.num_enc_layers,
            **_transform_chunk_opts(
                config.typed_value(
                    "chunk_opts",
                    dict(
                        # defaults. usually we would set the chunk_opts, but e.g. for importing the params,
                        # it does not matter.
                        chunk_stride=120,
                        chunk_history=2,
                        input_chunk_size=210,
                        end_chunk_size=20,
                    ),
                )
            ),
        )

    @classmethod
    def make_model(
        cls, in_dim: Dim, target_dim: Dim, *, num_enc_layers: int = 12, pos_emb_dropout: float = 0.1, **extra
    ) -> Model:
        """make"""
        return Model(
            in_dim,
            num_enc_layers=num_enc_layers,
            enc_model_dim=Dim(name="enc", dimension=512, kind=Dim.Types.Feature),
            enc_ff_dim=Dim(name="enc-ff", dimension=2048, kind=Dim.Types.Feature),
            enc_att_num_heads=8,
            enc_conformer_layer_opts=dict(
                conv_norm_opts=dict(use_mask=True),
                self_att_opts=dict(
                    # Shawn et al 2018 style, old RETURNN way.
                    with_bias=False,
                    with_linear_pos=False,
                    with_pos_bias=False,
                    learnable_pos_emb=True,
                    separate_pos_emb_per_head=False,
                    pos_emb_dropout=pos_emb_dropout,
                ),
                ff_activation=lambda x: rf.relu(x) ** 2.0,
            ),
            target_dim=target_dim,
            # TODO we do not set wb_target_dim, but we should set it to target_dim.
            #   Currently it would use target_dim+1, which is incorrect, as we reuse EOS here for blank.
            #   It's not so critical, though.
            blank_idx=_get_eos_idx(target_dim),  # blank is end-of-chunk (EOC) which is end-of-sequence (EOS)
            bos_idx=_get_bos_idx(target_dim),
            eos_idx=_get_eos_idx(target_dim),
            **extra,
        )


class Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: Dim,
        *,
        num_enc_layers: int = 12,
        target_dim: Dim,
        wb_target_dim: Optional[Dim] = None,
        blank_idx: int,
        eos_idx: int,
        bos_idx: int,
        input_chunk_size_dim: Dim,
        chunk_stride: int,
        chunk_history: int,
        end_chunk_size_dim: Dim,
        enc_aux_logits: Sequence[int] = (),  # layers
        enc_model_dim: Dim = Dim(name="enc", dimension=512),
        enc_ff_dim: Dim = Dim(name="enc-ff", dimension=2048),
        enc_att_num_heads: int = 4,
        enc_conformer_layer_opts: Optional[Dict[str, Any]] = None,
        enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
        att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
        att_dropout: float = 0.1,
        enc_dropout: float = 0.1,
        enc_att_dropout: float = 0.1,
    ):
        super(Model, self).__init__()

        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)

        self.in_dim = in_dim
        self.input_chunk_size_dim = input_chunk_size_dim
        self.chunk_stride = chunk_stride
        self.chunk_history = chunk_history
        self.end_chunk_size_dim = end_chunk_size_dim
        self.encoder = ChunkedConformerEncoder(
            in_dim,
            enc_model_dim,
            ff_dim=enc_ff_dim,
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            encoder_layer_opts=enc_conformer_layer_opts,
            num_layers=num_enc_layers,
            num_heads=enc_att_num_heads,
            dropout=enc_dropout,
            att_dropout=enc_att_dropout,
            chunk_history=chunk_history,
            end_chunk_size_dim=end_chunk_size_dim,
        )

        self.target_dim = target_dim
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        self.enc_key_total_dim = enc_key_total_dim
        self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
        self.att_num_heads = att_num_heads
        self.att_dropout = att_dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()

        if not wb_target_dim:
            wb_target_dim = target_dim + 1
        self.wb_target_dim = wb_target_dim
        for i in enc_aux_logits:
            setattr(self, f"enc_aux_logits_{i}", rf.Linear(self.encoder.out_dim, wb_target_dim))
        self.logits = rf.Linear(self.encoder.out_dim, wb_target_dim)  # final

        self._specaugment_opts = {
            "steps": config.typed_value("specaugment_steps") or (0, 1000, 2000),
            "max_consecutive_spatial_dims": config.typed_value("specaugment_max_consecutive_spatial_dims") or 20,
            "max_consecutive_feature_dims": config.typed_value("specaugment_max_consecutive_feature_dims")
            or (_log_mel_feature_dim // 5),
            "num_spatial_mask_factor": config.typed_value("specaugment_num_spatial_mask_factor") or 100,
        }

        self._mixup = None
        if config.typed_value("mixup", None) is not None:
            from i6_experiments.users.zeyer.returnn.models.rf_mixup import Mixup, MixupOpts

            self._mixup = Mixup(feature_dim=self.in_dim, opts=MixupOpts(**config.typed_value("mixup")))

    def encode(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dim, Dim]:
        """encode, and extend the encoder output for things we need in the decoder"""
        # log mel filterbank features
        source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
            source,
            in_spatial_dim=in_spatial_dim,
            out_dim=self.in_dim,
            sampling_rate=16_000,
            log_base=math.exp(2.3026),  # almost 10.0 but not exactly...
        )

        # Mixup
        if self._mixup:
            source = self._mixup(source, spatial_dim=in_spatial_dim)

        # SpecAugment
        source = rf.audio.specaugment(
            source,
            spatial_dim=in_spatial_dim,
            feature_dim=self.in_dim,
            **self._specaugment_opts,
        )

        # Chunk
        source, chunked_time_dim = rf.window(
            source,
            spatial_dim=in_spatial_dim,
            window_dim=self.input_chunk_size_dim,
            window_left=0,
            stride=self.chunk_stride,
        )

        # Encoder including convolutional frontend
        enc, enc_spatial_dim = self.encoder(
            source,
            in_spatial_dim=self.input_chunk_size_dim,
            chunked_time_dim=chunked_time_dim,
            collected_outputs=collected_outputs,
        )

        return enc, enc_spatial_dim, chunked_time_dim


def _get_bos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.bos_label_id is not None:
        bos_idx = target_dim.vocab.bos_label_id
    elif target_dim.vocab.eos_label_id is not None:
        bos_idx = target_dim.vocab.eos_label_id
    elif "<sil>" in target_dim.vocab.user_defined_symbol_ids:
        bos_idx = target_dim.vocab.user_defined_symbol_ids["<sil>"]
    else:
        raise Exception(f"cannot determine bos_idx from vocab {target_dim.vocab}")
    return bos_idx


def _get_eos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.eos_label_id is not None:
        eos_idx = target_dim.vocab.eos_label_id
    else:
        raise Exception(f"cannot determine eos_idx from vocab {target_dim.vocab}")
    return eos_idx


def from_scratch_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa
    enc_aux_logits = config.typed_value("aux_loss_layers")
    pos_emb_dropout = config.float("pos_emb_dropout", 0.1)
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
    return MakeModel.make_model(
        in_dim,
        target_dim,
        enc_aux_logits=enc_aux_logits or (),
        pos_emb_dropout=pos_emb_dropout,
        **_transform_chunk_opts(config.typed_value("chunk_opts")),
    )


from_scratch_model_def: ModelDef[Model]
from_scratch_model_def.behavior_version = 19
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = _batch_size_factor


def from_scratch_training(
    *, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim
):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    use_normalized_loss = config.bool("use_normalized_loss", True)

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    collected_outputs = {}
    enc, enc_spatial_dim, chunked_time_dim = model.encode(
        data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs
    )

    ctc_logits = []
    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers):
                continue
            linear = getattr(model, f"enc_aux_logits_{layer_idx}")
            aux_enc: Tensor = collected_outputs[str(layer_idx - 1)]
            ctc_logits.append((aux_enc, linear, f"ctc_{layer_idx}", aux_loss_scales[i]))
    ctc_logits.append((enc, model.logits, "ctc", 1.0))  # final

    for enc_, linear, loss_name, loss_scale in ctc_logits:
        enc_, _ = rf.slice(enc_, axis=enc_spatial_dim, size=model.end_chunk_size_dim)
        enc_, enc_spatial_dim_ = rf.merge_dims(enc_, dims=(chunked_time_dim, model.end_chunk_size_dim))
        logits = linear(enc_)
        loss = rf.ctc_loss(
            logits=logits,
            targets=targets,
            input_spatial_dim=enc_spatial_dim_,
            targets_spatial_dim=targets_spatial_dim,
            blank_index=model.blank_idx,
        )
        loss.mark_as_loss(
            loss_name,
            scale=loss_scale,
            custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
            use_normalized_loss=use_normalized_loss,
        )
        # decoded, decoded_spatial_dim = rf.ctc_greedy_decode(aux_logits, in_spatial_dim=enc_spatial_dim)
        # error = rf.edit_distance(
        #     a=decoded, a_spatial_dim=decoded_spatial_dim, b=targets, b_spatial_dim=targets_spatial_dim
        # )
        # error.mark_as_loss("label", as_error=True, custom_inv_norm_factor=targets_spatial_dim.get_size_tensor())


from_scratch_training: TrainDef[Model]
from_scratch_training.learning_rate_control_error_measure = "dev_score_full_sum"


def model_recog_v2(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    from returnn.config import get_global_config
    from returnn.datasets.util.vocabulary import Vocabulary
    from returnn.frontend.tensor_array import TensorArray

    config = get_global_config(return_empty_if_none=True)

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    enc, enc_spatial_dim, chunked_time_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    enc, _ = rf.slice(enc, axis=enc_spatial_dim, size=model.end_chunk_size_dim)
    enc, enc_spatial_dim_ = rf.merge_dims(enc, dims=(chunked_time_dim, model.end_chunk_size_dim))
    logits = model.logits(enc)
    label_log_probs = rf.log_softmax(logits, axis=model.wb_target_dim)
    label_log_probs_ta = TensorArray.unstack(label_log_probs, axis=enc_spatial_dim_)

    beam_size = config.int("beam_size", 12)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        label_log_prob = label_log_probs_ta[i]
        # Filter out finished beams
        label_log_prob = rf.where(
            rf.copy_to_device(i < enc_spatial_dim_.get_size_tensor()),
            label_log_prob,
            rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
        )
        seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab
        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{i}-beam"), axis=[beam_dim, model.wb_target_dim]
        )  # seq_log_prob, backrefs, target: Batch, Beam
        seq_targets.append(target)
        seq_backrefs.append(backrefs)
        i += 1

        if bool(i >= enc_spatial_dim_.get_dim_value()):
            break

    # Backtrack via backrefs, resolve beams.
    seq_targets_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_.insert(0, rf.gather(target, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets__ = TensorArray(seq_targets_[0])
    for target in seq_targets_:
        seq_targets__ = seq_targets__.push_back(target)
    out_spatial_dim = enc_spatial_dim_
    seq_targets = seq_targets__.stack(axis=out_spatial_dim)

    vocab_labels = list(model.target_dim.vocab.labels)
    if model.wb_target_dim.dimension > model.target_dim.dimension:
        vocab_labels += ["<blank>"]
    assert len(vocab_labels) == model.wb_target_dim.dimension, f"{model.target_dim} vs {model.wb_target_dim}?"
    vocab_labels[model.blank_idx] = "<blank>"
    seq_targets.sparse_dim.vocab = Vocabulary.create_vocab_from_labels(
        vocab_labels, user_defined_symbols={"<blank>": model.blank_idx}
    )

    print("** out lens:", out_spatial_dim.get_size_tensor().raw_tensor)
    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog_v2: RecogDef[Model]
model_recog_v2.output_with_beam = True
model_recog_v2.output_blank_label = "<blank>"  # EOC
model_recog_v2.batch_size_dependent = False


def _transform_chunk_opts(opts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not opts:
        opts = {}
    else:
        opts = opts.copy()
    if "input_chunk_size" in opts and "input_chunk_size_dim" not in opts:
        opts["input_chunk_size_dim"] = Dim(opts.pop("input_chunk_size"), name="input-chunk-size")
    if "end_chunk_size" in opts and "end_chunk_size_dim" not in opts:
        opts["end_chunk_size_dim"] = Dim(opts.pop("end_chunk_size"), name="sliced-chunk-size")
    return opts
