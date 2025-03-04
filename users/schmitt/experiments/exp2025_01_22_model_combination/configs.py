"""
Configurations, i.e. RETURNN settings,
shared across several setups here in this directory.
"""

from __future__ import annotations
from typing import Any, Dict

from i6_experiments.users.schmitt.util.dict_update import dict_update_deep
from i6_experiments.users.berger.corpus.librispeech.ctc_data import get_librispeech_data
import i6_experiments.users.raissi.utils.default_tools as run_tools
from i6_experiments.users.berger.systems.dataclasses import FeatureType

from .model.decoder import GlobalAttDecoder
from .train import _returnn_v2_train_step, from_scratch_training
from .model.conformer_tina import ConformerEncoderLayerWithSwitchedOrder

import returnn.frontend as rf
from returnn.frontend import Dim
from returnn.frontend.encoder.conformer import ConformerEncoder
from returnn.frontend.build_from_dict import _get_cls_name

_batch_size_factor = 160

config_24gb_num_epochs = 2000
config_24gb_v1 = dict(
  model_opts=dict(
    encoder_opts=rf.build_dict(
      rf.encoder.conformer.ConformerEncoder,
      out_dimension=512,
      input_layer_cls=rf.build_dict(rf.encoder.conformer.ConformerConvSubsample),
      num_layers=12,
      num_heads=8,
      dropout=0.1,
      att_dropout=0.1,
      encoder_layer=rf.build_dict(
        rf.encoder.conformer.ConformerEncoderLayer,
        conv_norm_opts=dict(use_mask=True),
        self_att_opts=dict(
          # Shawn et al 2018 style, old RETURNN way.
          with_bias=False,
          with_linear_pos=False,
          with_pos_bias=False,
          learnable_pos_emb=True,
          separate_pos_emb_per_head=False,
          pos_emb_dropout=0.0,
        ),
        self_att=rf.build_dict(rf.RelPosSelfAttention),
        ff_activation=rf.build_dict(rf.relu_square),
      )
    ),
    decoder_opts=rf.build_dict(
      GlobalAttDecoder,
      use_weight_feedback=True,
      use_att_ctx_in_state=True,
    )
  ),
  train_opts=dict(
    train_step_func=_returnn_v2_train_step,
    train_def=from_scratch_training,
    max_seq_length={"data": 19.5 * 16_000},  # 19.5 seconds (input is raw audio with 16kHz)
    rf_att_dropout_broadcast=False,
    specaugment_steps=(5_000, 15_000, 25_000),
    accum_grad_multiple_step=2,
    batch_size=35_000,
    batching="laplace:.1000",
    grad_scaler=None,
    gradient_clip_global_norm=5.0,
    optimizer={
      "class": "adamw",
      "epsilon": 1e-8,
      "weight_decay": 1e-6,
      "weight_decay_modules_blacklist": [
        "rf.Embedding",
        "rf.LearnedRelativePositionalEncoding",
      ],
    },
    num_epochs=config_24gb_num_epochs,
    lr_opts=dict(
      type="dyn_lr_piecewise_linear_epoch-wise",
      peak_lr=1e-3,
      init_lr=1e-3 * 1e-2,
      num_epochs=config_24gb_num_epochs,
    ),
    dataset_opts=dict(
      use_speed_pert=True,
      epoch_wise_filter={(1, 5): {"max_mean_len": 1000}},
    ),
    aux_loss_layers=(),
    cleanup_old_models=dict(
      keep=[config_24gb_num_epochs],
      keep_best_n=4,
      keep_last_n=1,
    )
  )
)

gammatone_data = get_librispeech_data(
  run_tools.u16_tools_factored.returnn_root,
  run_tools.u16_tools_factored.returnn_python_exe,
  rasr_binary_path=run_tools.u16_tools_factored.rasr_binary_path,
  add_unknown_phoneme_and_mapping=False,
  use_augmented_lexicon=True,
  use_wei_lexicon=False,
  feature_type=FeatureType.GAMMATONE_16K,
)
config_24gb_v2 = dict_update_deep(
  config_24gb_v1,
  {
    # model opts
    "model_opts.encoder_opts.encoder_layer.class": _get_cls_name(ConformerEncoderLayerWithSwitchedOrder),
    "model_opts.encoder_opts.encoder_layer.self_att_opts.learnable_pos_emb_clipping": 32,
    "model_opts.encoder_opts.encoder_layer.ff_activation": rf.swish,
    "model_opts.encoder_opts.encoder_layer.conv_norm_opts.use_mask": False,
    "model_opts.encoder_opts.encoder_layer.conv_norm_opts.eps": 1e-5,
    "model_opts.encoder_opts.input_layer_opts": {"activation": rf.swish},
    "model_opts.feature_extraction": None,
    "model_opts.feature_dimension": 50,
    # train opts
    "train_opts.aux_loss_layers": (4, 8),
    "train_opts.dataset_opts.hdf_features": {
      "train": gammatone_data.train_data_config["files"],
      "devtrain": gammatone_data.train_data_config["files"],
      "cv": gammatone_data.cv_data_config["files"],
    },
    "train_opts.dataset_opts.seq_order_control_dataset": {
      "cv": "features",
    },
    "train_opts.dataset_opts.segment_paths": {
      "cv": None,
    },
    "train_opts.max_seq_length": {"data": 19.5 * 100},  # 19.5 seconds (input is gammatone every 10ms)
  }
)

config_11gpu_mgpu_num_epochs = 500
config_11gb_mgpu_v1 = dict_update_deep(
  config_24gb_v1,
  {
    "train_opts.accum_grad_multiple_step": 4,
    "train_opts.batch_size": 15_000,
    "train_opts.num_epochs": config_11gpu_mgpu_num_epochs,
    "train_opts.lr_opts.num_epochs": config_11gpu_mgpu_num_epochs,
    "train_opts.cleanup_old_models.keep": [config_11gpu_mgpu_num_epochs],
    "train_opts.torch_distributed": {"reduce_type": "param", "param_sync_step": 100},
  }
)

config_tina_v1 = dict(
  model_opts=dict(
    feature_extraction=None,
    feature_dimension=50,  # gammatone
  )
)

config_ctc_v1 = dict(
  model_opts=dict(
    feature_extraction=None,
    feature_dimension=50,  # gammatone
    target_dimension=79,
    blank_or_sil_idx=0,
  )
)

config_ctc_v2 = dict_update_deep(
  config_24gb_v1,
  {},
  [
    "model_opts.decoder_opts"
  ]
)

config_phon_transducer_v1 = dict(
  model_opts=dict(
    feature_extraction=None,
    feature_dimension=50,  # gammatone
    target_dimension=79,
    blank_or_sil_idx=0,
  )
)

config_post_hmm_v1 = dict(
  model_opts=dict(
    feature_extraction=None,
    feature_dimension=50,  # gammatone
    target_dimension=84,
    blank_or_sil_idx=81,
  )
)

config_monophone_fh_v1 = dict(
  model_opts=dict(
    feature_extraction=None,
    feature_dimension=50,  # gammatone
    target_dimension=84,
    # blank_or_sil_idx=0,  # will change hash but 0 is the default anyway so we don't need to set it
  )
)

config_diphone_fh_v1 = dict(
  model_opts=dict(
    feature_extraction=None,
    feature_dimension=50,  # gammatone
    target_dimension=84,
    left_target_dimension=42,
    # blank_or_sil_idx=0,  # will change hash but 0 is the default anyway so we don't need to set it
  )
)
