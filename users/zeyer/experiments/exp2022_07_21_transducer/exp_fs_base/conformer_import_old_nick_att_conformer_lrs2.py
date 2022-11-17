"""
Starting point, 2022-10-12

Import model old_nick_att_conformer_lrs2,
which uses a Conformer from an old pure-RETURNN net dict config,
into a new RETURNN-common `nn.Conformer`.
"""

from __future__ import annotations
from typing import Optional, Any, Tuple, Dict, Sequence
import contextlib
from sisyphus import tk
from returnn_common import nn
from returnn_common.nn.encoder.blstm import BlstmEncoder
from returnn_common.asr.specaugment import specaugment_v2
from i6_core.returnn.training import Checkpoint

from i6_experiments.users.zeyer.datasets.switchboard_2020.task import get_switchboard_task_bpe1k
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef, ModelWithCheckpoint
from i6_experiments.users.zeyer.recog import recog_model
from i6_experiments.users.zeyer.returnn.convert_ckpt import ConvertCheckpointJob


# Can be excluded now. The test was successful. This version exactly reproduces the old model.
# More specifically, `test_import` passes exactly, and the recog is within a reasonable range.
# One small diff is that we use update_sample_only_in_training=True here but not in the original model.
_exclude_me = True


def sis_run_with_prefix(prefix_name: str):
    """run the exp"""
    if _exclude_me:
        return
    task = get_switchboard_task_bpe1k()

    epoch = 300
    new_chkpt = ConvertCheckpointJob(
        checkpoint=Checkpoint(index_path=tk.Path(
            f"/u/zeyer/setups/combined/2021-05-31"
            f"/alias/exp_fs_base/old_nick_att_conformer_lrs2/train/output/models/epoch.{epoch:03}.index")),
        make_model_func=MakeModel(
            extern_data_dict=task.train_dataset.get_extern_data(),
            default_input_key=task.train_dataset.get_default_input(),
            default_target_key=task.train_dataset.get_default_target(),
        ),
        map_func=map_param_func_v2,
    ).out_checkpoint
    model_with_checkpoint = ModelWithCheckpoint(definition=from_scratch_model_def, checkpoint=new_chkpt)

    res = recog_model(task, model_with_checkpoint, model_recog)
    tk.register_output(prefix_name + f"/recog_results_per_epoch/{epoch:03}", res.output)


class MakeModel:
    """for import"""

    def __init__(self, *, extern_data_dict, default_input_key, default_target_key):
        self.extern_data_dict = extern_data_dict
        self.default_input_key = default_input_key
        self.default_target_key = default_target_key

    def __call__(self) -> Model:
        data = nn.Data(name=self.default_input_key, **self.extern_data_dict[self.default_input_key])
        targets = nn.Data(name=self.default_target_key, **self.extern_data_dict[self.default_target_key])
        in_dim = data.feature_dim_or_sparse_dim
        target_dim = targets.feature_dim_or_sparse_dim
        return self.make_model(in_dim, target_dim)

    @classmethod
    def make_model(cls, in_dim: nn.Dim, target_dim: nn.Dim, *, num_enc_layers: int = 12) -> Model:
        """make"""
        return Model(
            in_dim,
            num_enc_layers=num_enc_layers,
            enc_input_allow_pool_last=True,
            enc_model_dim=nn.FeatureDim("enc", 512),
            enc_ff_dim=nn.FeatureDim("enc-ff", 2048),
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
                )
            ),
            nb_target_dim=target_dim,
            wb_target_dim=target_dim + 1,
            blank_idx=target_dim.dimension,
            bos_idx=_get_bos_idx(target_dim),
        )


_ParamMapping = {}  # type: Dict[str,str]


def _add_params():
    # frontend
    for layer_idx in [0, 1]:
        for direction in ["fw", "bw"]:
            for param_name in ["W", "W_re", "b"]:
                _ParamMapping[f"encoder.input_layer.layers.{layer_idx}.{direction}.param_{param_name}"] = \
                    f"encoder/lstm{layer_idx}_{direction}/rec/{param_name}"
    _ParamMapping.update({
        "encoder.input_projection.weight": "encoder/source_linear/W",
    })
    # conformer
    for layer_idx in range(12):
        # FF
        for sub in [1, 2]:
            _ParamMapping[f"encoder.layers.{layer_idx}.ffn{sub}.linear_ff.weight"] = \
                f"encoder/conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ff1/W"
            _ParamMapping[f"encoder.layers.{layer_idx}.ffn{sub}.linear_ff.bias"] = \
                f"encoder/conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ff1/b"
            _ParamMapping[f"encoder.layers.{layer_idx}.ffn{sub}.linear_out.weight"] = \
                f"encoder/conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ff2/W"
            _ParamMapping[f"encoder.layers.{layer_idx}.ffn{sub}.linear_out.bias"] = \
                f"encoder/conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ff2/b"
            _ParamMapping[f"encoder.layers.{layer_idx}.ffn{sub}_layer_norm.scale"] = \
                f"encoder/conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ln/scale"
            _ParamMapping[f"encoder.layers.{layer_idx}.ffn{sub}_layer_norm.bias"] = \
                f"encoder/conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ln/bias"
        # conv
        _ParamMapping[f"encoder.layers.{layer_idx}.conv_block.positionwise_conv1.weight"] = \
            f"encoder/conformer_block_{layer_idx + 1:02d}_conv_mod_pointwise_conv1/W"
        _ParamMapping[f"encoder.layers.{layer_idx}.conv_block.positionwise_conv1.bias"] = \
            f"encoder/conformer_block_{layer_idx + 1:02d}_conv_mod_pointwise_conv1/b"
        _ParamMapping[f"encoder.layers.{layer_idx}.conv_block.depthwise_conv.filter"] = \
            f"encoder/conformer_block_{layer_idx + 1:02d}_conv_mod_depthwise_conv2/W"
        _ParamMapping[f"encoder.layers.{layer_idx}.conv_block.depthwise_conv.bias"] = \
            f"encoder/conformer_block_{layer_idx + 1:02d}_conv_mod_depthwise_conv2/bias"
        _ParamMapping[f"encoder.layers.{layer_idx}.conv_block.positionwise_conv2.weight"] = \
            f"encoder/conformer_block_{layer_idx + 1:02d}_conv_mod_pointwise_conv2/W"
        _ParamMapping[f"encoder.layers.{layer_idx}.conv_block.positionwise_conv2.bias"] = \
            f"encoder/conformer_block_{layer_idx + 1:02d}_conv_mod_pointwise_conv2/b"
        _ParamMapping[f"encoder.layers.{layer_idx}.conv_block.norm.running_mean"] = \
            f"encoder/conformer_block_{layer_idx + 1:02d}_conv_mod_bn/batch_norm/v2_mean"
        _ParamMapping[f"encoder.layers.{layer_idx}.conv_block.norm.running_variance"] = \
            f"encoder/conformer_block_{layer_idx + 1:02d}_conv_mod_bn/batch_norm/v2_variance"
        _ParamMapping[f"encoder.layers.{layer_idx}.conv_block.norm.beta"] = \
            f"encoder/conformer_block_{layer_idx + 1:02d}_conv_mod_bn/batch_norm/v2_beta"
        _ParamMapping[f"encoder.layers.{layer_idx}.conv_block.norm.gamma"] = \
            f"encoder/conformer_block_{layer_idx + 1:02d}_conv_mod_bn/batch_norm/v2_gamma"
        _ParamMapping[f"encoder.layers.{layer_idx}.conv_layer_norm.scale"] = \
            f"encoder/conformer_block_{layer_idx + 1:02d}_conv_mod_ln/scale"
        _ParamMapping[f"encoder.layers.{layer_idx}.conv_layer_norm.bias"] = \
            f"encoder/conformer_block_{layer_idx + 1:02d}_conv_mod_ln/bias"
        # self-att
        _ParamMapping[f"encoder.layers.{layer_idx}.self_att.qkv.weight"] = \
            f"encoder/conformer_block_{layer_idx + 1:02d}_self_att/QKV"
        _ParamMapping[f"encoder.layers.{layer_idx}.self_att.proj.weight"] = \
            f"encoder/conformer_block_{layer_idx + 1:02d}_self_att_linear/W"
        _ParamMapping[f"encoder.layers.{layer_idx}.self_att_layer_norm.scale"] = \
            f"encoder/conformer_block_{layer_idx + 1:02d}_self_att_ln/scale"
        _ParamMapping[f"encoder.layers.{layer_idx}.self_att_layer_norm.bias"] = \
            f"encoder/conformer_block_{layer_idx + 1:02d}_self_att_ln/bias"
        _ParamMapping[f"encoder.layers.{layer_idx}.self_att.learned_pos_emb.pos_emb"] = \
            f"encoder/conformer_block_{layer_idx + 1:02d}_self_att_ln_rel_pos_enc/encoding_matrix"
        # final layer norm
        _ParamMapping[f"encoder.layers.{layer_idx}.final_layer_norm.scale"] = \
            f"encoder/conformer_block_{layer_idx + 1:02d}_ln/scale"
        _ParamMapping[f"encoder.layers.{layer_idx}.final_layer_norm.bias"] = \
            f"encoder/conformer_block_{layer_idx + 1:02d}_ln/bias"


_add_params()


def map_param_func_v2(reader, name, var):
    """map params"""
    import tensorflow as tf
    from tensorflow.python.training.py_checkpoint_reader import CheckpointReader
    assert isinstance(reader, CheckpointReader)
    assert isinstance(var, tf.Variable)
    if reader.has_tensor(var.op.name):
        return reader.get_tensor(var.op.name)
    if name in _ParamMapping:
        var_name = _ParamMapping[name]
        assert reader.has_tensor(var_name)
        value = reader.get_tensor(var_name)
        assert value.shape == tuple(var.shape.as_list())
        assert value.dtype == var.dtype.as_numpy_dtype
        return value
    raise NotImplementedError(f"cannot map {name!r} {var}")


def test_import():
    _layer_mapping = {
        "encoder/source_linear": "encoder/input_projection",
        "encoder/conformer_block_01_ffmod_1_drop2": "encoder/layers/0/ffn1",
        "encoder/conformer_block_01_ffmod_1_res": "encoder/layers/0/add",
        "encoder/conformer_block_01_self_att_res": "encoder/layers/0/add_0",
        "encoder/conformer_block_01_conv_mod_ln": "encoder/layers/0/conv_layer_norm",
        "encoder/conformer_block_01_conv_mod_glu": "encoder/layers/0/conv_block/gating",
        "encoder/conformer_block_01_conv_mod_depthwise_conv2": "encoder/layers/0/conv_block/depthwise_conv",
        "encoder/conformer_block_01_conv_mod_bn": "encoder/layers/0/conv_block/norm",
        "encoder/conformer_block_01_conv_mod_pointwise_conv2": "encoder/layers/0/conv_block",
        "encoder/conformer_block_01_conv_mod_res": "encoder/layers/0/add_1",
        "encoder/conformer_block_01_ffmod_2_ln": "encoder/layers/0/ffn2_layer_norm",
        "encoder/conformer_block_01_ffmod_2_ff1": "encoder/layers/0/ffn2/linear_ff",
        "encoder/conformer_block_01_ffmod_2_swish": "encoder/layers/0/ffn2/swish",
        "encoder/conformer_block_01_ffmod_2_drop2": "encoder/layers/0/ffn2",
        "encoder/conformer_block_01_ffmod_2_res": "encoder/layers/0/add_2",
        "encoder/conformer_block_01": "encoder/layers/0",
        "encoder": "encoder",
    }

    from i6_experiments.common.setups.returnn_common import serialization
    exec(serialization.PythonEnlargeStackWorkaroundNonhashedCode.code)

    in_dim = nn.FeatureDim("in", 40)
    time_dim = nn.SpatialDim("time")
    target_dim = nn.FeatureDim("target", 1030)
    target_dim.vocab = nn.Vocabulary.create_vocab_from_labels(
        [str(i) for i in range(target_dim.dimension)], eos_label=0)
    data = nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim])
    target_spatial_dim = nn.SpatialDim("target_spatial")
    target = nn.Data("target", dim_tags=[nn.batch_dim, target_spatial_dim], sparse_dim=target_dim)

    num_layers = 3
    from .old_nick_att_conformer_lrs2 import Model as OldModel, encoder_args as old_encoder_args
    old_encoder_args["num_blocks"] = num_layers
    old_encoder_args["batch_norm_opts"]["update_sample_only_in_training"] = True  # better for testing

    print("*** Create old model")
    nn.reset_default_root_name_ctx()
    old_model = OldModel(
        in_dim,
        nb_target_dim=target_dim,
        wb_target_dim=target_dim + 1,
        blank_idx=target_dim.dimension,
        bos_idx=_get_bos_idx(target_dim),
        epoch=300,
    )
    from_scratch_training(
        model=old_model, data=nn.get_extern_data(data), data_spatial_dim=time_dim,
        targets=nn.get_extern_data(target), targets_spatial_dim=target_spatial_dim)

    from returnn.config import Config
    config = Config(dict(log_verbositiy=5))
    config.update(nn.get_returnn_config().get_config_raw_dict(old_model))

    from returnn.tf.network import TFNetwork
    import tensorflow as tf
    import numpy.testing
    import tempfile
    import atexit
    import shutil
    ckpt_dir = tempfile.mkdtemp("returnn-import-test")
    atexit.register(lambda: shutil.rmtree(ckpt_dir))

    print("*** Construct TF graph for old model")
    tf1 = tf.compat.v1
    with tf1.Graph().as_default() as graph, tf1.Session(graph=graph).as_default() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(config.typed_dict["network"])
        print("*** Random init old model")
        net.initialize_params(session)
        print("*** Save old model to disk")
        net.save_params_to_file(ckpt_dir + "/old_model/model", session=session)

        print("*** Forwarding ...")
        from returnn_common.tests.returnn_helpers import make_feed_dict
        feed_dict = make_feed_dict(net.extern_data)
        fetches = net.get_fetches_dict()
        old_model_outputs_data = {}
        for old_layer_name, _ in _layer_mapping.items():
            layer = net.get_layer(old_layer_name)
            out = layer.output.copy_as_batch_major()
            old_model_outputs_data[old_layer_name] = out
            fetches["layer:" + old_layer_name] = out.placeholder
            for i, tag in enumerate(out.dim_tags):
                if tag.dyn_size_ext:
                    old_model_outputs_data[f"{old_layer_name}:size{i}"] = tag.dyn_size_ext
                    fetches[f"layer:{old_layer_name}:size{i}"] = tag.dyn_size_ext.placeholder
        old_model_outputs_fetch = session.run(fetches, feed_dict=feed_dict)

    def _make_new_model():
        return MakeModel.make_model(in_dim, target_dim, num_enc_layers=num_layers)

    print("*** Convert old model to new model")
    converter = ConvertCheckpointJob(
        checkpoint=Checkpoint(index_path=tk.Path(ckpt_dir + "/old_model/model.index")),
        make_model_func=_make_new_model,
        map_func=map_param_func_v2,
    )
    converter._out_model_dir = tk.Path(ckpt_dir + "/new_model")
    converter.out_checkpoint.index_path = tk.Path(ckpt_dir + "/new_model/model.index")
    converter.run()

    print("*** Create new model")
    nn.reset_default_root_name_ctx()
    new_model = _make_new_model()
    x = nn.get_extern_data(data)
    y = new_model.encode(x, in_spatial_dim=time_dim)[0]["enc"]
    y.mark_as_default_output()
    config.update(nn.get_returnn_config().get_config_raw_dict(new_model))
    with tf1.Graph().as_default() as graph, tf1.Session(graph=graph).as_default() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(config.typed_dict["network"])
        print("*** Load new model params from disk")
        net.load_params_from_file(ckpt_dir + "/new_model/model", session=session)

        old_model_outputs_data["encoder/source_linear"].get_time_dim_tag().declare_same_as(
            net.get_layer("encoder/input_projection").output.get_time_dim_tag())

        print("*** Forwarding ...")
        feed_dict = make_feed_dict(net.extern_data)
        fetches = net.get_fetches_dict()
        for old_layer_name, new_layer_name in _layer_mapping.items():
            layer = net.get_layer(new_layer_name)
            old_out = old_model_outputs_data[old_layer_name]
            assert old_out.batch_ndim == layer.output.batch_ndim
            mapped_axes = layer.output.find_matching_dim_map(
                old_out, list(range(old_out.batch_ndim)))
            out = layer.output.copy_transpose([mapped_axes[i] for i in range(old_out.batch_ndim)])
            fetches["layer:" + old_layer_name] = out.placeholder
            for i, tag in enumerate(out.dim_tags):
                if tag.dyn_size_ext:
                    old_model_outputs_data[f"{old_layer_name}:size{i}"] = tag.dyn_size_ext
                    fetches[f"layer:{old_layer_name}:size{i}"] = tag.dyn_size_ext.placeholder
        new_model_outputs_fetch = session.run(fetches, feed_dict=feed_dict)

        for old_layer_name, new_layer_name in _layer_mapping.items():
            out = old_model_outputs_data[old_layer_name]
            for i, tag in enumerate(out.dim_tags):
                if tag.dyn_size_ext:
                    old_v = old_model_outputs_fetch[f"layer:{old_layer_name}:size{i}"]
                    new_v = new_model_outputs_fetch[f"layer:{old_layer_name}:size{i}"]
                    numpy.testing.assert_equal(old_v, new_v, err_msg=f"{tag} mismatch")
            old_v = old_model_outputs_fetch["layer:" + old_layer_name]
            new_v = new_model_outputs_fetch["layer:" + old_layer_name]
            for i, tag in enumerate(out.dim_tags):
                if tag.dyn_size_ext:
                    assert tag.dyn_size_ext.dim_tags == (nn.batch_dim,)  # not implemented otherwise
                    assert out.batch_dim_axis == 0  # not implemented otherwise but should be ensured above
                    size_v = old_model_outputs_fetch[f"layer:{old_layer_name}:size{i}"]
                    for b in range(old_v.shape[0]):
                        idx = tuple([slice(b, b + 1)] + [slice(None, None)] * (i - 1) + [slice(size_v[b], None)])
                        old_v[idx] = 0
                        new_v[idx] = 0
            print("* Comparing", old_layer_name, "vs", new_layer_name)
            numpy.testing.assert_almost_equal(old_v, new_v, decimal=5)

    print("*** Done, all correct (!), exit now ***")
    raise SystemExit("done")


# `py` is the default sis config function name. so when running this directly, run the import test.
# So you can just run:
# `sis m recipe/i6_experiments/users/zeyer/experiments/exp2022_07_21_transducer/exp_fs_base/
#  conformer_import_old_nick_att_conformer_lrs2.py`
py = test_import


class Model(nn.Module):
    """Model definition"""

    def __init__(self, in_dim: nn.Dim, *,
                 num_enc_layers: int = 12,
                 nb_target_dim: nn.Dim,
                 wb_target_dim: nn.Dim,
                 blank_idx: int,
                 bos_idx: int,
                 enc_input_allow_pool_last: bool = False,
                 enc_model_dim: nn.Dim = nn.FeatureDim("enc", 512),
                 enc_ff_dim: nn.Dim = nn.FeatureDim("enc-ff", 2048),
                 enc_att_num_heads: int = 4,
                 enc_conformer_layer_opts: Optional[Dict[str, Any]] = None,
                 enc_key_total_dim: nn.Dim = nn.FeatureDim("enc_key_total_dim", 200),
                 att_num_heads: nn.Dim = nn.SpatialDim("att_num_heads", 1),
                 att_dropout: float = 0.1,
                 enc_dropout: float = 0.1,
                 enc_att_dropout: float = 0.1,
                 l2: float = 0.0001,
                 ):
        super(Model, self).__init__()
        if nn.ConformerEncoderLayer.use_dropout_after_self_att:
            nn.ConformerEncoderLayer.use_dropout_after_self_att = False
        self.in_dim = in_dim
        self.encoder = nn.ConformerEncoder(
            in_dim,
            enc_model_dim,
            ff_dim=enc_ff_dim,
            input_layer=BlstmEncoder(
                in_dim,
                nn.FeatureDim("pre-lstm", 512),
                num_layers=2, time_reduction=6,
                dropout=enc_dropout,
                allow_pool_last=enc_input_allow_pool_last,
            ),
            encoder_layer_opts=enc_conformer_layer_opts,
            num_layers=num_enc_layers,
            num_heads=enc_att_num_heads,
            dropout=enc_dropout,
            att_dropout=enc_att_dropout,
        )

        self.nb_target_dim = nb_target_dim
        self.wb_target_dim = wb_target_dim
        self.blank_idx = blank_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        self.enc_key_total_dim = enc_key_total_dim
        self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
        self.att_num_heads = att_num_heads
        self.att_dropout = att_dropout

        self.enc_ctx = nn.Linear(self.encoder.out_dim, enc_key_total_dim)
        self.enc_ctx_dropout = 0.2
        self.enc_win_dim = nn.SpatialDim("enc_win_dim", 5)
        self.att_query = nn.Linear(self.encoder.out_dim, enc_key_total_dim, with_bias=False)
        self.lm = DecoderLabelSync(nb_target_dim, l2=l2)
        self.readout_in_am = nn.Linear(2 * self.encoder.out_dim, nn.FeatureDim("readout", 1000), with_bias=False)
        self.readout_in_am_dropout = 0.1
        self.readout_in_lm = nn.Linear(self.lm.out_dim, self.readout_in_am.out_dim, with_bias=False)
        self.readout_in_lm_dropout = 0.1
        self.readout_in_bias = nn.Parameter([self.readout_in_am.out_dim])
        self.readout_reduce_num_pieces = 2
        self.readout_dim = self.readout_in_am.out_dim // self.readout_reduce_num_pieces
        self.out_nb_label_logits = nn.Linear(self.readout_dim, nb_target_dim)
        self.label_log_prob_dropout = 0.3
        self.out_emit_logit = nn.Linear(self.readout_dim, nn.FeatureDim("emit", 1))

        for p in self.encoder.parameters():
            p.weight_decay = l2
        for p in self.enc_ctx.parameters():
            p.weight_decay = l2

    def encode(self, source: nn.Tensor, *, in_spatial_dim: nn.Dim) -> (Dict[str, nn.Tensor], nn.Dim):
        """encode, and extend the encoder output for things we need in the decoder"""
        source = specaugment_v2(source, spatial_dim=in_spatial_dim, feature_dim=self.in_dim)
        enc, enc_spatial_dim = self.encoder(source, in_spatial_dim=in_spatial_dim)
        enc_ctx = self.enc_ctx(nn.dropout(enc, self.enc_ctx_dropout, axis=enc.feature_dim))
        enc_ctx_win, _ = nn.window(enc_ctx, spatial_dim=enc_spatial_dim, window_dim=self.enc_win_dim)
        enc_val_win, _ = nn.window(enc, spatial_dim=enc_spatial_dim, window_dim=self.enc_win_dim)
        return dict(enc=enc, enc_ctx_win=enc_ctx_win, enc_val_win=enc_val_win), enc_spatial_dim

    @staticmethod
    def encoder_unstack(ext: Dict[str, nn.Tensor]) -> Dict[str, nn.Tensor]:
        """
        prepare the encoder output for the loop (full-sum or time-sync)
        """
        # We might improve or generalize the interface later...
        # https://github.com/rwth-i6/returnn_common/issues/202
        loop = nn.NameCtx.inner_loop()
        return {k: loop.unstack(v) for k, v in ext.items()}

    def decoder_default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> Optional[nn.LayerState]:
        """Default initial state"""
        return nn.LayerState(lm=self.lm.default_initial_state(batch_dims=batch_dims))

    def decode(self, *,
               enc: nn.Tensor,  # single frame if axis is single step, or sequence otherwise ("am" before)
               enc_spatial_dim: nn.Dim,  # single step or time axis,
               enc_ctx_win: nn.Tensor,  # like enc
               enc_val_win: nn.Tensor,  # like enc
               all_combinations_out: bool = False,  # [...,prev_nb_target_spatial_dim,axis] out
               prev_nb_target: Optional[nn.Tensor] = None,  # non-blank
               prev_nb_target_spatial_dim: Optional[nn.Dim] = None,  # one longer than target_spatial_dim, due to BOS
               prev_wb_target: Optional[nn.Tensor] = None,  # with blank
               wb_target_spatial_dim: Optional[nn.Dim] = None,  # single step or align-label spatial axis
               state: Optional[nn.LayerState] = None,
               ) -> (ProbsFromReadout, nn.LayerState):
        """decoder step, or operating on full seq"""
        if state is None:
            assert enc_spatial_dim != nn.single_step_dim, "state should be explicit, to avoid mistakes"
            batch_dims = enc.batch_dims_ordered(
                remove=(enc.feature_dim, enc_spatial_dim)
                if enc_spatial_dim != nn.single_step_dim
                else (enc.feature_dim,))
            state = self.decoder_default_initial_state(batch_dims=batch_dims)
        state_ = nn.LayerState()

        att_query = self.att_query(enc)
        att_energy = nn.dot(enc_ctx_win, att_query, reduce=att_query.feature_dim)
        att_energy = att_energy * (att_energy.feature_dim.dimension ** -0.5)
        att_weights = nn.softmax(att_energy, axis=self.enc_win_dim)
        att_weights = nn.dropout(att_weights, dropout=self.att_dropout, axis=att_weights.shape_ordered)
        att = nn.dot(att_weights, enc_val_win, reduce=self.enc_win_dim)

        if all_combinations_out:
            assert prev_nb_target is not None and prev_nb_target_spatial_dim is not None
            assert prev_nb_target_spatial_dim in prev_nb_target.shape
            assert enc_spatial_dim != nn.single_step_dim
            lm_scope = contextlib.nullcontext()
            lm_input = prev_nb_target
            lm_axis = prev_nb_target_spatial_dim
        else:
            assert prev_wb_target is not None and wb_target_spatial_dim is not None
            assert wb_target_spatial_dim in {enc_spatial_dim, nn.single_step_dim}
            prev_out_emit = prev_wb_target != self.blank_idx
            lm_scope = nn.MaskedComputation(mask=prev_out_emit)
            lm_input = nn.reinterpret_set_sparse_dim(prev_wb_target, out_dim=self.nb_target_dim)
            lm_axis = wb_target_spatial_dim

        with lm_scope:
            lm, state_.lm = self.lm(lm_input, spatial_dim=lm_axis, state=state.lm)

            # We could have simpler code by directly concatenating the readout inputs.
            # However, for better efficiency, keep am/lm path separate initially.
            readout_in_lm_in = nn.dropout(lm, self.readout_in_lm_dropout, axis=lm.feature_dim)
            readout_in_lm = self.readout_in_lm(readout_in_lm_in)

        readout_in_am_in = nn.concat_features(enc, att)
        readout_in_am_in = nn.dropout(readout_in_am_in, self.readout_in_am_dropout, axis=readout_in_am_in.feature_dim)
        readout_in_am = self.readout_in_am(readout_in_am_in)
        readout_in = nn.combine_bc(readout_in_am, "+", readout_in_lm)
        readout_in += self.readout_in_bias
        readout = nn.reduce_out(
            readout_in, mode="max", num_pieces=self.readout_reduce_num_pieces, out_dim=self.readout_dim)

        return ProbsFromReadout(model=self, readout=readout), state_


class DecoderLabelSync(nn.Module):
    """
    Often called the (I)LM part, or prediction network.
    Runs label-sync, i.e. only on non-blank labels.
    """
    def __init__(self, in_dim: nn.Dim, *,
                 embed_dim: nn.Dim = nn.FeatureDim("embed", 256),
                 lstm_dim: nn.Dim = nn.FeatureDim("lstm", 1024),
                 dropout: float = 0.2,
                 l2: float = 0.0001,
                 ):
        super(DecoderLabelSync, self).__init__()
        self.embed = nn.Linear(in_dim, embed_dim)
        self.dropout = dropout
        self.lstm = nn.LSTM(self.embed.out_dim, lstm_dim)
        self.out_dim = self.lstm.out_dim
        for p in self.parameters():
            p.weight_decay = l2

    def default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> Optional[nn.LayerState]:
        """init"""
        return self.lstm.default_initial_state(batch_dims=batch_dims)

    def __call__(self, source: nn.Tensor, *, spatial_dim: nn.Dim, state: nn.LayerState
                 ) -> Tuple[nn.Tensor, nn.LayerState]:
        embed = self.embed(source)
        embed = nn.dropout(embed, self.dropout, axis=embed.feature_dim)
        lstm, state = self.lstm(embed, spatial_dim=spatial_dim, state=state)
        return lstm, state


class ProbsFromReadout:
    """
    functions to calculate the probabilities from the readout
    """
    def __init__(self, *, model: Model, readout: nn.Tensor):
        self.model = model
        self.readout = readout

    def get_label_logits(self) -> nn.Tensor:
        """label log probs"""
        label_logits_in = nn.dropout(self.readout, self.model.label_log_prob_dropout, axis=self.readout.feature_dim)
        label_logits = self.model.out_nb_label_logits(label_logits_in)
        return label_logits

    def get_label_log_probs(self) -> nn.Tensor:
        """label log probs"""
        label_logits = self.get_label_logits()
        label_log_prob = nn.log_softmax(label_logits, axis=label_logits.feature_dim)
        return label_log_prob

    def get_emit_logit(self) -> nn.Tensor:
        """emit logit"""
        emit_logit = self.model.out_emit_logit(self.readout)
        return emit_logit

    def get_wb_label_log_probs(self) -> nn.Tensor:
        """align label log probs"""
        label_log_prob = self.get_label_log_probs()
        emit_logit = self.get_emit_logit()
        emit_log_prob = nn.log_sigmoid(emit_logit)
        blank_log_prob = nn.log_sigmoid(-emit_logit)
        label_emit_log_prob = label_log_prob + nn.squeeze(emit_log_prob, axis=emit_log_prob.feature_dim)
        assert self.model.blank_idx == label_log_prob.feature_dim.dimension  # not implemented otherwise
        output_log_prob = nn.concat_features(label_emit_log_prob, blank_log_prob)
        return output_log_prob


def _get_bos_idx(target_dim: nn.Dim) -> int:
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


def from_scratch_model_def(*, epoch: int, in_dim: nn.Dim, target_dim: nn.Dim) -> Model:
    """Function is run within RETURNN."""
    epoch  # noqa
    return MakeModel.make_model(in_dim, target_dim)


from_scratch_model_def: ModelDef[Model]
from_scratch_model_def.behavior_version = 14


def from_scratch_training(*,
                          model: Model,
                          data: nn.Tensor, data_spatial_dim: nn.Dim,
                          targets: nn.Tensor, targets_spatial_dim: nn.Dim
                          ):
    """Function is run within RETURNN."""
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    prev_targets, prev_targets_spatial_dim = nn.prev_target_seq(
        targets, spatial_dim=targets_spatial_dim, bos_idx=model.bos_idx, out_one_longer=True)
    probs, _ = model.decode(
        **enc_args,
        enc_spatial_dim=enc_spatial_dim,
        all_combinations_out=True,
        prev_nb_target=prev_targets,
        prev_nb_target_spatial_dim=prev_targets_spatial_dim)
    out_log_prob = probs.get_wb_label_log_probs()
    loss = nn.transducer_time_sync_full_sum_neg_log_prob(
        log_probs=out_log_prob,
        labels=targets,
        input_spatial_dim=enc_spatial_dim,
        labels_spatial_dim=targets_spatial_dim,
        blank_index=model.blank_idx)
    loss.mark_as_loss("full_sum")


from_scratch_training: TrainDef[Model]
from_scratch_training.learning_rate_control_error_measure = "dev_score_full_sum"


def model_recog(*,
                model: Model,
                data: nn.Tensor, data_spatial_dim: nn.Dim,
                targets_dim: nn.Dim,  # noqa
                ) -> nn.Tensor:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return: recog results including beam
    """
    batch_dims = data.batch_dims_ordered((data_spatial_dim, data.feature_dim))
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    beam_size = 12

    loop = nn.Loop(axis=enc_spatial_dim)  # time-sync transducer
    loop.max_seq_len = nn.dim_value(enc_spatial_dim) * 2
    loop.state.decoder = model.decoder_default_initial_state(batch_dims=batch_dims)
    loop.state.target = nn.constant(model.blank_idx, shape=batch_dims, sparse_dim=model.wb_target_dim)
    with loop:
        enc = model.encoder_unstack(enc_args)
        probs, loop.state.decoder = model.decode(
            **enc,
            enc_spatial_dim=nn.single_step_dim,
            wb_target_spatial_dim=nn.single_step_dim,
            prev_wb_target=loop.state.target,
            state=loop.state.decoder)
        log_prob = probs.get_wb_label_log_probs()
        loop.state.target = nn.choice(
            log_prob, input_type="log_prob",
            target=None, search=True, beam_size=beam_size,
            length_normalization=False)
        res = loop.stack(loop.state.target)

    assert model.blank_idx == targets_dim.dimension  # added at the end
    res.feature_dim.vocab = nn.Vocabulary.create_vocab_from_labels(
        targets_dim.vocab.labels + ["<blank>"], user_defined_symbols={"<blank>": model.blank_idx})
    return res


# RecogDef API
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = False
