"""
Replicating the pipeline of my 2020 transducer work:
https://github.com/rwth-i6/returnn-experiments/tree/master/2020-rnn-transducer

Note: this file is loaded in different contexts:

- as a sisyphus config file. In this case, the function `py` is called.
- via the generated RETURNN configs. In this case, all the Sisyphus stuff is ignored,
  and only selected functions will run.

The reason for this is that we want to have all the relevant code for the experiment
to be in one place, such that we can make a copy of the file as a base for another separate experiment.

Note on the hash of the model definition:
This is explicit, via the version object below,
and via the module name (__name__; this includes the package name),
and via the model def function name.

Note on the motivation for the interface:
- should be flexible to different tasks (datasets)
- should be simple (obviously!)
- for training and recognition (alignment)
- need to use dynamic get_network because we don't want to run the net code in the root config

"""

from sisyphus import tk
from .task import get_switchboard_task
from .train import train
from .recog import recog
from .align import align
from returnn_common import nn
from returnn_common.nn.encoder.blstm_cnn_specaug import BlstmCnnSpecAugEncoder


# version is used for the hash of the model definition,
# together with the model def function name together with the module name (__name__).
assert __name__.startswith("i6_experiments.")  # just a sanity check
version = 1
extra_hash = (version,)


def sis_config_main():
    """sis config function"""
    task = get_switchboard_task()

    step1_model = train(
        task=task, model_def=from_scratch_model_def, train_def=from_scratch_training, extra_hash=extra_hash)
    step2_alignment = align(task=task, model=step1_model)
    # use step1 model params; different to the paper
    step3_model = train(
        task=task, model_def=extended_model_def, train_def=extended_model_training, extra_hash=extra_hash,
        alignment=step2_alignment, init_params=step1_model.checkpoint)
    step4_model = train(
        task=task, model_def=extended_model_def, train_def=extended_model_training, extra_hash=extra_hash,
        alignment=step2_alignment, init_params=step3_model.checkpoint)

    tk.register_output('step1', recog(task, step1_model).main_measure_value)
    tk.register_output('step3', recog(task, step3_model).main_measure_value)
    tk.register_output('step4', recog(task, step4_model).main_measure_value)


py = sis_config_main  # `py` is the default sis config function name


class Model(nn.Module):
    """Model definition"""

    def __init__(self, *, num_enc_layers=6):
        super(Model, self).__init__()
        self.encoder = BlstmCnnSpecAugEncoder(num_layers=num_enc_layers)
        # TODO decoder...


class Decoder(nn.Module):
    """Decoder"""
    def __init__(self, *,
                 enc_key_total_dim: nn.Dim,
                 enc_key_per_head_dim: nn.Dim,
                 attention_dropout: float,
                 ):
        super(Decoder, self).__init__()
        # TODO
        self.enc_key_total_dim = enc_key_total_dim
        self.enc_key_per_head_dim = enc_key_per_head_dim
        self.attention_dropout = attention_dropout

        self.att_query = nn.Linear(enc_key_total_dim)

    def encoder_ext(self, enc_out: nn.Tensor, enc_spatial_dim: nn.Dim) -> Dict[str, nn.Tensor]:
        """Extend the encoder output"""
        # TODO
        return enc_out

    def __call__(self, ):
        pass

        # TODO we should not access the loop
        #  - for encoder/enc_ctx_win/enc_val_win,
        #    we need a generalization of UnstackLayer with state (non-batch scalar index)
        loop = nn.NameCtx.inner_loop()

        _ = {

    "am": {"class": "copy", "from": "data:source"},

    "enc_ctx_win": {"class": "gather_nd", "from": "base:enc_ctx_win", "position": ":i"},  # [B,W,D]
    "enc_val_win": {"class": "gather_nd", "from": "base:enc_val_win", "position": ":i"},  # [B,W,D]
    "att_query": {"class": "linear", "from": "am", "activation": None, "with_bias": False, "n_out": self.enc_key_total_dim},
    'att_energy': {"class": "dot", "red1": "f", "red2": "f", "var1": "static:0", "var2": None,
                   "from": ['enc_ctx_win', 'att_query']},  # (B, W)
    'att_weights0': {"class": "softmax_over_spatial", "axis": "static:0", "from": 'att_energy',
                     "energy_factor": self.enc_key_per_head_dim.dimension ** -0.5},  # (B, W)
    'att_weights1': {"class": "dropout", "dropout_noise_shape": {"*": None},
                     "from": 'att_weights0', "dropout": self.attention_dropout},
    "att_weights": {"class": "merge_dims", "from": "att_weights1", "axes": "except_time"},
    'att': {"class": "dot", "from": ['att_weights', 'enc_val_win'],
            "red1": "static:0", "red2": "static:0", "var1": None, "var2": "f"},  # (B, V)

    "prev_out_non_blank": {
        "class": "reinterpret_data", "from": "prev:output", "set_sparse_dim": target_num_labels},
    # "class": "reinterpret_data", "from": "prev:output_wo_b", "set_sparse_dim": target_num_labels},  # [B,]
    "lm_masked": {"class": "masked_computation",
                  "mask": "prev:output_emit",
                  "from": "prev_out_non_blank",  # in decoding
                  "masked_from": "base:lm_input" if task == "train" else None,  # enables optimization if used

                  "unit": {
                      "class": "subnetwork", "from": "data", "trainable": True,
                      "subnetwork": {
                          "input_embed": {"class": "linear", "n_out": 256, "activation": "identity", "trainable": True,
                                          "L2": l2, "from": "data"},
                          "lstm0": {"class": "rec", "unit": "nativelstm2", "dropout": 0.2, "n_out": 1024, "L2": l2,
                                    "from": "input_embed", "trainable": True},
                          "output": {"class": "copy", "from": "lstm0"}
                          # "output": {"class": "linear", "from": "lstm1", "activation": "softmax", "dropout": 0.2, "n_out": target_num_labels, "trainable": False}
                      }}},
    # "lm_embed_masked": {"class": "linear", "activation": None, "n_out": 256, "from": "lm_masked"},
    # "lm_unmask": {"class": "unmask", "from": "lm_masked", "mask": "prev:output_emit"},
    # "lm_embed_unmask": {"class": "unmask", "from": "lm_embed_masked", "mask": "prev:output_emit"},
    "lm": {"class": "copy", "from": "lm_embed_unmask"},  # [B,L]

    # joint network: (W_enc h_{enc,t} + W_pred * h_{pred,u} + b)
    # train : (T-enc, B, F|2048) ; (U+1, B, F|256)
    # search: (B, F|2048) ; (B, F|256)
    "readout_in": {"class": "linear", "from": ["am", "att", "lm_masked"], "activation": None, "n_out": 1000, "L2": l2,
                   "dropout": 0.2,
                   "out_type": {"batch_dim_axis": 2 if task == "train" else 0,
                                "shape": (None, None, 1000) if task == "train" else (1000,),
                                "time_dim_axis": 0 if task == "train" else None}},  # (T, U+1, B, 1000)

    "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": "readout_in"},

    "label_log_prob": {
        "class": "linear", "from": "readout", "activation": "log_softmax", "dropout": 0.3,
        "n_out": target_num_labels},  # (B, T, U+1, 1030)
    "emit_prob0": {"class": "linear", "from": "readout", "activation": None, "n_out": 1,
                   "is_output_layer": True},  # (B, T, U+1, 1)
    "emit_log_prob": {"class": "activation", "from": "emit_prob0", "activation": "log_sigmoid"},  # (B, T, U+1, 1)
    "blank_log_prob": {"class": "eval", "from": "emit_prob0", "eval": "tf.log_sigmoid(-source(0))"},  # (B, T, U+1, 1)
    "label_emit_log_prob": {"class": "combine", "kind": "add",
                            "from": ["label_log_prob", "emit_log_prob"]},  # (B, T, U+1, 1), scaling factor in log-space
    "output_log_prob": {"class": "copy", "from": ["blank_log_prob", "label_emit_log_prob"]},  # (B, T, U+1, 1031)

    "output_prob": {
        "class": "eval", "from": ["output_log_prob", "base:data:" + target, "base:encoder"], "eval": rna_loss,
        "out_type": rna_loss_out, "loss": "as_is",
    },

    # this only works when the loop has been optimized, i.e. log-probs are (B, T, U, V)
    "rna_alignment": {"class": "eval", "from": ["output_log_prob", "base:data:" + target, "base:encoder"],
                      "eval": rna_alignment, "out_type": rna_alignment_out,
                      "is_output_layer": True} if task == "train"  # (B, T)
    else {"class": "copy", "from": "output_log_prob"},

    # During training   : targetb = "target"  (RNA-loss)
    # During recognition: targetb = "targetb"
    'output': {
        'class': 'choice', 'target': targetb, 'beam_size': beam_size,
        'from': "output_log_prob", "input_type": "log_prob",
        "initial_output": 0,
        "cheating": "exclusive" if task == "train" else None,
        # "explicit_search_sources": ["prev:u"] if task == "train" else None,
        # "custom_score_combine": targetb_recomb_train if task == "train" else None
        "explicit_search_sources": ["prev:out_str", "prev:output"] if task == "search" else None,
        "custom_score_combine": targetb_recomb_recog if task == "search" else None
    },

    "out_str": {
        "class": "eval", "from": ["prev:out_str", "output_emit", "output"],
        "initial_output": None, "out_type": {"shape": (), "dtype": "string"},
        "eval": out_str},

    "output_is_not_blank": {"class": "compare", "from": "output", "value": targetb_blank_idx, "kind": "not_equal",
                            "initial_output": True},

    # initial state=True so that we are consistent to the training and the initial state is correctly set.
    "output_emit": {"class": "copy", "from": "output_is_not_blank", "initial_output": True, "is_output_layer": True},

    "const0": {"class": "constant", "value": 0, "collocate_with": ["du", "dt"]},
    "const1": {"class": "constant", "value": 1, "collocate_with": ["du", "dt"]},
        }


def from_scratch_model_def(*, epoch: int) -> Model:
    """Function is run within RETURNN."""
    pass  # TODO


def from_scratch_training(*,
                          model: Model,
                          data: nn.Data, data_spatial_dim: nn.Dim,
                          targets: nn.Data, targets_spatial_dim: nn.Dim
                          ):
    """Function is run within RETURNN."""
    # TODO feed through model, define full sum loss, mark_as_loss


def extended_model_def(*, epoch: int) -> Model:
    """Function is run within RETURNN."""
    pass  # TODO


def extended_model_training(*,
                            model: Model,
                            data: nn.Data, data_spatial_dim: nn.Dim,
                            align_targets: nn.Data, align_targets_spatial_dim: nn.Dim
                            ):
    """Function is run within RETURNN."""
    pass  # TODO


def model_recog(*,
                model: Model,
                data: nn.Data, data_spatial_dim: nn.Dim,
                target_vocab: nn.Dim,
                ):
    """Function is run within RETURNN."""
    pass  # TODO
    # TODO probably this should be moved over to the recog module, as this will probably be the same always
