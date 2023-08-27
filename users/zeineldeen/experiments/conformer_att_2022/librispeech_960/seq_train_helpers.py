import copy


def get_trans_lstm_lm():
    """
    Returns subnetwork of a LM trained only on transcripts. The LM architecture is close to ASR decoder architecture.
    PPL = 76 (train) ; 112 (dev clean+other)
    """
    return {
        "class": "subnetwork",
        "from": ["prev:output"],
        "load_on_init": "/work/asr4/michel/setups-data/language_modelling/librispeech/neurallm/decoder_sized_transcripts_only/net-model/network.007",
        "n_out": 10025,
        "trainable": False,
        "subnetwork": {
            "input": {
                "class": "linear",
                "n_out": 128,
                "activation": "identity",
                "trainable": False,
            },
            "lstm0": {
                "class": "rnn_cell",
                "unit": "LSTMBlock",
                "n_out": 1000,
                "unit_opts": {"forget_bias": 0.0},
                "from": ["input"],
                "trainable": False,
            },
            "output": {
                "class": "linear",
                "from": ["lstm0"],
                "activation": "identity",
                "use_transposed_weights": False,
                "n_out": 10025,
                "trainable": False,
            },
        },
    }


def add_double_softmax(net, loss_scale, ce_scale, am_scale, lm_scale, ce_label_smoothing=0.0):
    """
    Add a double softmax training criterion.
    The loss is defined as:
        P = P_asr (w|..) * P_lm(w) / sum_w' P_asr(w'|..) * P_lm(w')
        log P = log P_asr (w|..) + log P_lm(w) - log sum_w' P_asr(w'|..) * P_lm(w')
        Loss = - log P
    Scales are applied to P_asr and P_lm.

    All layers here are added to the decoder subnetwork.
    """
    subnet = net["output"]["unit"]

    # Add transcription LM to the network
    subnet["lm_output"] = get_trans_lstm_lm()
    subnet["lm_output_prob"] = {
        "class": "activation",
        "activation": "softmax",
        "from": ["lm_output"],
        "target": "bpe_labels",
    }  # [B,V]

    # Add the double softmax loss
    subnet["asr_log_prob"] = {"class": "activation", "activation": "safe_log", "from": "output_prob"}
    subnet["scaled_asr_log_prob"] = {"class": "eval", "from": "asr_log_prob", "eval": f"source(0) * {am_scale}"}
    subnet["lm_log_prob"] = {"class": "activation", "activation": "safe_log", "from": "lm_output_prob"}
    subnet["scaled_lm_log_prob"] = {"class": "eval", "from": "lm_log_prob", "eval": f"source(0) * {lm_scale}"}
    subnet["numerator"] = {
        "class": "combine",
        "kind": "add",
        "from": ["scaled_asr_log_prob", "scaled_lm_log_prob"],
    }  # [B,V]
    subnet["denominator"] = {"class": "reduce", "mode": "logsumexp", "from": "numerator", "axis": "f"}  # [B]
    subnet["double_softmax_log_prob"] = {"class": "combine", "kind": "sub", "from": ["numerator", "denominator"]}
    subnet["double_softmax_loss"] = {
        "class": "eval",
        "eval": "tf.exp(source(0))",
        "from": "double_softmax_log_prob",
        "loss": "ce",
        "loss_scale": loss_scale,
        "target": "bpe_labels",
    }

    # CE loss
    if "loss_opts" in subnet["output_prob"]:
        subnet["output_prob"]["loss_opts"]["scale"] = ce_scale
        subnet["output_prob"]["loss_opts"]["label_smoothing"] = ce_label_smoothing
    else:
        subnet["output_prob"].pop("loss_scale", None)
        subnet["output_prob"]["loss_opts"] = {"scale": ce_scale, "label_smoothing": ce_label_smoothing}


def add_min_wer(net, loss_scale, ce_scale, am_scale, lm_scale, beam_size, ce_label_smoothing=0.0):
    """
    Add minimum WER training criterion. For reference: https://arxiv.org/abs/1712.01818
    """
    subnet = net["output"]["unit"]

    # add LM
    subnet["lm_output"] = get_trans_lstm_lm()
    subnet["lm_output_prob"] = {
        "class": "activation",
        "activation": "softmax",
        "from": ["lm_output"],
        "target": "bpe_labels",
    }  # [B,V]

    # add extra search subnet to return the N-best list
    net["extra.search:output"] = copy.deepcopy(net["output"])
    extra_search_subnet = net["extra.search:output"]["unit"]

    extra_search_subnet["output_prob"].pop("loss", None)
    extra_search_subnet["output_prob"].pop("loss_opts", None)
    extra_search_subnet["output_prob"].pop("loss_scale", None)

    extra_search_subnet["asr_output_log_prob"] = {
        "class": "activation",
        "activation": "safe_log",
        "from": "output_prob",
    }
    extra_search_subnet["scaled_asr_output_log_prob"] = {
        "class": "eval",
        "from": "asr_output_log_prob",
        "eval": f"source(0) * {am_scale}",
    }
    extra_search_subnet["lm_output_log_prob"] = {
        "class": "activation",
        "activation": "safe_log",
        "from": "lm_output_prob",
    }
    extra_search_subnet["scaled_lm_output_log_prob"] = {
        "class": "eval",
        "from": "lm_output_log_prob",
        "eval": f"source(0) * {lm_scale}",
    }
    extra_search_subnet["combined_log_prob"] = {
        "class": "combine",
        "kind": "add",
        "from": ["scaled_asr_output_log_prob", "scaled_lm_output_log_prob"],
    }
    extra_search_subnet["output"]["from"] = "combined_log_prob"  # asr + lm
    extra_search_subnet["output"]["length_normalization"] = False
    extra_search_subnet["output"]["cheating"] = "exclusive"
    extra_search_subnet["output"]["beam_size"] = beam_size
    extra_search_subnet["output"]["input_type"] = "log_prob"

    # CE loss
    if "loss_opts" in subnet["output_prob"]:
        subnet["output_prob"]["loss_opts"]["scale"] = ce_scale
        subnet["output_prob"]["loss_opts"]["label_smoothing"] = ce_label_smoothing
    else:
        subnet["output_prob"].pop("loss_scale", None)
        subnet["output_prob"]["loss_opts"] = {"scale": ce_scale, "label_smoothing": ce_label_smoothing}

    # MinWER loss
    net["min_wer"] = {
        "class": "copy",
        "from": ["extra.search:output"],
        "loss": "expected_loss",
        "loss_opts": {
            "divide_beam_size": False,
            "loss": {"class": "edit_distance"},
            "loss_kind": "error",
            "norm_scores_stop_gradient": False,
            "subtract_average_loss": False,
            "scale": loss_scale,
        },
        "target": "bpe_labels",
    }


def add_mmi(net, loss_scale, ce_scale, am_scale, lm_scale, beam_size, ce_label_smoothing=0.0):
    """
    Add Maximum Mutual Information training criterion.
    """
    assert ce_label_smoothing==0.0, "label smoothing not implemented correctly"
    subnet = net["output"]["unit"]

    # add LM
    subnet["lm_output"] = get_trans_lstm_lm()
    subnet["lm_output_prob"] = {
        "class": "activation",
        "activation": "softmax",
        "from": ["lm_output"],
        "target": "bpe_labels",
    }  # [B,V]

    # add extra search subnet to return the N-best list
    net["extra.search:output"] = copy.deepcopy(net["output"])
    extra_search_subnet = net["extra.search:output"]["unit"]

    extra_search_subnet["output_prob"].pop("loss", None)
    extra_search_subnet["output_prob"].pop("loss_opts", None)
    extra_search_subnet["output_prob"].pop("loss_scale", None)

    extra_search_subnet["asr_output_log_prob"] = {
        "class": "activation",
        "activation": "safe_log",
        "from": "output_prob",
    }
    extra_search_subnet["scaled_asr_output_log_prob"] = {
        "class": "eval",
        "from": "asr_output_log_prob",
        "eval": f"source(0) * {am_scale}",
    }
    extra_search_subnet["lm_output_log_prob"] = {
        "class": "activation",
        "activation": "safe_log",
        "from": "lm_output_prob",
    }
    extra_search_subnet["scaled_lm_output_log_prob"] = {
        "class": "eval",
        "from": "lm_output_log_prob",
        "eval": f"source(0) * {lm_scale}",
    }
    extra_search_subnet["combined_log_prob"] = {
        "class": "combine",
        "kind": "add",
        "from": ["scaled_asr_output_log_prob", "scaled_lm_output_log_prob"],
    }
    extra_search_subnet["output"]["from"] = "combined_log_prob"  # asr + lm
    extra_search_subnet["output"]["length_normalization"] = False
    extra_search_subnet["output"]["cheating"] = "exclusive"
    extra_search_subnet["output"]["beam_size"] = beam_size
    extra_search_subnet["output"]["input_type"] = "log_prob"

    # CE loss
    if "loss_opts" in subnet["output_prob"]:
        subnet["output_prob"]["loss_opts"]["scale"] = loss_scale + ce_scale
        subnet["output_prob"]["loss_opts"]["label_smoothing"] = ce_label_smoothing
    else:
        subnet["output_prob"].pop("loss_scale", None)
        subnet["output_prob"]["loss_opts"] = {"scale": loss_scale + ce_scale, "label_smoothing": ce_label_smoothing}

    # MMI loss
    net["get_scores"] = {"class": "choice_get_beam_scores", "from" : ["extra.search:output"] }
    net["split_scores"] = {"class": "split_batch_beam", "from": ["get_scores"] }
    net["denominator_score"] = {
        "class": "reduce",
        "mode": "logsumexp",
        "axes": "F",
        "from": ["split_scores"] ,
        "loss": "as_is", 
        "loss_opts": {"scale": loss_scale}
    }
