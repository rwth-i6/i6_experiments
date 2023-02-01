import copy
from IPython import embed

from sisyphus import *

import i6_core.returnn as returnn
import i6_core.rasr as sp

from i6_core.rasr.command import RasrCommand

from i6_experiments.users.raissi.setups.common.helpers.pipeline_data import (
    ContextEnum,
    ContextMapper
)

context_mapper = ContextMapper()

def get_embedding_layer(source, dim, l2=0.01):
    return {"with_bias": False, "L2": l2, "class": "linear", "activation": None, "from": [f"data:{source}"], "n_out": dim}

def blstm_config(network, partition_epochs, lr=5e-4, batch_size=10000, max_seqs=100, chunking="64:32", **kwargs):
    key_interval = "learning_rate_control_min_num_epochs_per_new_lr"
    if key_interval in kwargs:
        update_interval = kwargs[key_interval]
    else:
        update_interval = partition_epochs["dev"]
    result = {"batch_size": batch_size,
              "max_seqs": max_seqs,
              "cache_size": "0",
              "batching": "random",
              "chunking": chunking,
              "learning_rate": lr,
              "learning_rate_control": "newbob_multi_epoch",
              "newbob_multi_num_epochs": partition_epochs["train"],
              "newbob_multi_update_interval": update_interval,
              "use_tensorflow": True,
              "multiprocessing": True,
              "network": network}
    result.update(**kwargs)
    
    return result


def blstm_network(layers=6 * [512], dropout=0.1, l2=0.1, unit_type="nativelstm2", specaugment=False):
    num_layers = len(layers)
    assert num_layers > 0
    
    result = {}
    
    if specaugment:
        result["source"] = {"class": "eval",
                            "eval": "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)"}
        input_first_layer = "source"
    else:
        input_first_layer = ["data"]
    
    for l, size in enumerate(layers):
        l += 1  # start counting from 1
        for direction, name in [(1, "fwd"), (-1, "bwd")]:
            if l == 1:
                from_layers = input_first_layer
            else:
                from_layers = ["fwd_%d" % (l - 1), "bwd_%d" % (l - 1)]
            result["%s_%d" % (name, l)] = {"class": "rec",
                                           "unit": unit_type,
                                           "direction": direction,
                                           "n_out": size,
                                           "dropout": dropout,
                                           "L2": l2,
                                           "from": from_layers}
    
    return result


def add_delta_blstm_(network, name, l2=0.01, source_layer=None):
    if source_layer is None:
        source_layer = 'encoder-output'
    
    network["fwd_delta"] = {'L2': l2, 'class': 'rec', 'direction': 1, 'dropout': 0.1,
                            'from': source_layer, 'n_out': 512, 'unit': 'nativelstm2'}
    
    network["bwd_delta"] = {'L2': l2, 'class': 'rec', 'direction': -1, 'dropout': 0.1,
                            'from': source_layer, 'n_out': 512, 'unit': 'nativelstm2'}
    
    network[name] = {'class': 'copy', 'from': ['fwd_delta', 'bwd_delta']}
    
    return network


def get_common_subnetwork_for_targets_with_blstm(layers, dropout, l2, use_boundary_classes=True, n_contexts=47, n_states_per_phone=3,
                                                 unit_type="nativelstm2",
                                                 specaugment=False, is_min_duration=False, use_word_end_classes=False):
    acousticNet = blstm_network(layers, dropout, l2, unit_type=unit_type, specaugment=specaugment)
    assert (not (use_boundary_classes and use_word_end_classes))
    
    if use_boundary_classes:
        labelingInput = "popBoundry"
        acousticNet["boundryClass"] = {"class": "eval", "from": "data:classes", "eval": "tf.math.floormod(source(0),%d)" % 4,
                                       "out_type": {'dim': 4, 'dtype': 'int32', 'sparse': True}}
        acousticNet["popBoundry"] = {"class": "eval", "from": ["data:classes"], "eval": "tf.math.floordiv(source(0),%d)" % 4,
                                     "out_type": {'dim': (n_contexts ** 3) * n_states_per_phone, 'dtype': 'int32',
                                                  'sparse': True}}
    
    elif use_word_end_classes:
        labelingInput = "popWordEnd"
        acousticNet["wordEndClass"] = {"class": "eval", "from": "data:classes", "eval": "tf.math.floormod(source(0),%d)" % 2,
                                       "out_type": {'dim': 2, 'dtype': 'int32', 'sparse': True}}
        acousticNet["popWordEnd"] = {"class": "eval", "from": ["data:classes"], "eval": "tf.math.floordiv(source(0),%d)" % 2,
                                     "out_type": {'dim': (n_contexts ** 3) * n_states_per_phone,
                                                  'dtype': 'int32',
                                                  'sparse': True}}
    
    else:
        labelingInput = "data:classes"

    acousticNet["futureLabel"] = {"class": "eval", "from": labelingInput,
                                  "eval": "tf.math.floormod(source(0),%d)" % n_contexts,
                                  "register_as_extern_data": "futureLabel",
                                  "out_type": {'dim': n_contexts, 'dtype': 'int32', 'sparse': True}}
    acousticNet["popFutureLabel"] = {"class": "eval", "from": labelingInput,
                                     "eval": "tf.math.floordiv(source(0),%d)" % n_contexts,
                                     "out_type": {'dim': (n_contexts ** 2 * n_states_per_phone), 'dtype': 'int32', 'sparse': True}}

    acousticNet["pastLabel"] = {"class": "eval", "from": "popFutureLabel",
                                "eval": "tf.math.floormod(source(0),%d)" % n_contexts,
                                "register_as_extern_data": "pastLabel",
                                "out_type": {'dim': n_contexts, 'dtype': 'int32', 'sparse': True}}
    acousticNet["popPastLabel"] = {"class": "eval", "from": "popFutureLabel",
                                   "eval": "tf.math.floordiv(source(0),%d)" % n_contexts,
                                   "out_type": {'dim': n_contexts * n_states_per_phone, 'dtype': 'int32', 'sparse': True}}

    acousticNet["stateId"] = {"class": "eval", "from": "popPastLabel",
                              "eval": "tf.math.floormod(source(0),%d)" % n_states_per_phone,
                              "out_type": {'dim': n_states_per_phone, 'dtype': 'int32', 'sparse': True}}

    acousticNet["centerPhoneme"] = {"class": "eval", "from": "popPastLabel",
                                    "eval": "tf.math.floordiv(source(0),%d)" % n_states_per_phone,
                                    "out_type": {'dim': n_contexts, 'dtype': 'int32', 'sparse': True}}
    
    if is_min_duration:
        if use_word_end_classes:
            acousticNet["centerState"] = {"class": "eval", "from": ["centerPhoneme", "wordEndClass"],
                                          "eval": "(source(0)*2)+source(1)",
                                          "register_as_extern_data": "centerState",
                                          "out_type": {'dim': n_contexts * 2, 'dtype': 'int32', 'sparse': True}}
        else:
            acousticNet["centerPhoneme"]["register_as_extern_data"] = "centerState"
    else:
        if use_word_end_classes:
            acousticNet["centerState"] = {"class": "eval", "from": ["centerPhoneme", "stateId", "wordEndClass"],
                                          "eval": f"(((source(0)*{n_states_per_phone})+source(2))*2)+source(1)",
                                          "register_as_extern_data": "centerState",
                                          "out_type": {'dim': n_contexts * n_states_per_phone * 2, 'dtype': 'int32',
                                                       'sparse': True}}
        else:
            acousticNet["centerState"] = {"class": "eval", "from": ["centerPhoneme", "stateId"],
                                          "eval": f"(source(0)*{n_states_per_phone})+source(1)", "register_as_extern_data": "centerState",
                                          "out_type": {'dim': n_contexts * n_states_per_phone, 'dtype': 'int32', 'sparse': True}}
    
    return acousticNet


def make_config(context_type, partition_epochs,
                python_prolog=None, python_epilog="",
                n_states_per_phone=3, n_contexts=47,
                use_boundary_classes=False, is_min_duration=False, use_word_end_classes=False,
                layers=6 * [500], l2=0.01, mlp_l2=0.01, dropout=0.1,
                ph_emb_size=64, st_emb_size=256, focal_loss_factor=2.0, label_smoothing=0.0,
                add_mlps=False, use_multi_task=True, final_context_type=None, eval_dense_label=False,
                unit_type="nativelstm2", specaugment=False, shared_delta_encoder=False, **kwargs):
    if eval_dense_label:
        shared_network = get_common_subnetwork_for_targets_with_blstm(layers,
                                                                     dropout,
                                                                     l2,
                                                                     use_boundary_classes=use_boundary_classes,
                                                                     n_contexts=n_contexts,
                                                                     n_states_per_phone=n_states_per_phone,
                                                                     unit_type=unit_type,
                                                                     specaugment=specaugment,
                                                                     is_min_duration=is_min_duration,
                                                                     use_word_end_classes=use_word_end_classes)
    else:
        shared_network = blstm_network(layers, dropout, l2, unit_type=unit_type, specaugment=specaugment)
    
    config = get_config_for_context_type(context_type, partition_epochs,shared_network,
                                         use_multi_task=use_multi_task, add_mlps=add_mlps,
                                         final_context_type=final_context_type, shared_delta_encoder=shared_delta_encoder,
                                         st_emb_size=st_emb_size, ph_emb_size=ph_emb_size,
                                         focal_loss_factor=focal_loss_factor, label_smoothing=label_smoothing,
                                         l2=mlp_l2, **kwargs)
    
    returnnConfig = returnn.ReturnnConfig(config, python_prolog=python_prolog, python_epilog=python_epilog)
    
    return returnnConfig


def get_graph_from_returnn_config(returnnConfig, python_prolog=None, python_epilog=None):
    if isinstance(returnnConfig, returnn.ReturnnConfig):
        tf_returnn_config = copy.copy(returnnConfig.config)
    else:
        tf_returnn_config = copy.copy(returnnConfig)
    
    tf_returnn_config["train"] = {"class": "ExternSprintDataset",
                               "partitionEpoch": 1,
                               "sprintConfigStr": "",
                               "sprintTrainerExecPath": None}
    
    tf_returnn_config["dev"] = {"class": "ExternSprintDataset",
                             "partitionEpoch": 1,
                             "sprintConfigStr": "",
                             "sprintTrainerExecPath": None}
    

    
    conf = returnn.ReturnnConfig(tf_returnn_config, python_prolog=python_prolog, python_epilog=python_epilog)
    returnn_config_file = returnn.WriteReturnnConfigJob(conf).out_returnn_config_file
    compiledGraphJob = returnn.CompileTFGraphJob(returnn_config_file)
    
    return compiledGraphJob.out_graph


def get_config_for_context_type(context_type, partition_epochs, shared_network,
                                add_mlps=False, use_multi_task=True, final_context_type=None, shared_delta_encoder=False,
                                ph_emb_size=64, st_emb_size=256,
                                focal_loss_factor=2.0, label_smoothing=0.2, l2=0.01, **kwargs):
    ###
    # This function is the entry point for strating the training, which means for context-dependent models
    # separate functions for the multi-stage training
    ###

    mono_network = get_monophone_net(shared_network,
                                    add_mlps=add_mlps,
                                    use_multi_task=use_multi_task,
                                    final_ctx_type=final_context_type,
                                    ph_emb_size=ph_emb_size,
                                    st_emb_size=st_emb_size,
                                    focal_loss_factor=focal_loss_factor,
                                    label_smoothing=label_smoothing,
                                    l2=l2,
                                    shared_delta_encoder=shared_delta_encoder)

    if context_type.value == context_mapper.get_enum(1):
        config = blstm_config(mono_network, partition_epochs, **kwargs)
        return config
    
    elif context_type.value == context_mapper.get_enum(2):
        network = get_diphone_net(mono_network, use_multi_task=use_multi_task, label_smoothing=label_smoothing, l2=l2, ph_emb_size=ph_emb_size, st_emb_size=st_emb_size)
    elif context_type.value == context_mapper.get_enum(4):
        network = get_forward_net(mono_network, l2=l2, ph_emb_size=ph_emb_size, st_emb_size=st_emb_size)

    else:
        assert(False, "Network type not implemented")
        sys.exit()

    #ToDo: implement others
    """
    elif context_type.value == context_mapper.get_enum(3):
        network = get_symmetric_net(mono_network)



    elif context_type.value == context_mapper.get_enum(5):
        network = get_backward_net(mono_network)
    """
    #for ctx-dep networks you always need MLPs and will use dense label
    if context_type != context_mapper.get_enum(1):
        assert (add_mlps, "for context-dependent models you need MLP layers")
        network['center-output']['target'] = 'centerState'

    #ToDo: once you have your optimal setting add it here using bin_ce_weight
    config = blstm_config(network, partition_epochs, **kwargs)
    return config


def set_Mlp_component(network, layerName, outputSize, sourceLayer="encoder-output", l2=None):
    lOneName = ("-").join(["linear1", layerName])
    lTwoName = ("-").join(["linear2", layerName])
    
    network[lOneName] = {"class": "linear",
                         "activation": "relu",
                         "from": sourceLayer,
                         "n_out": outputSize}
    
    network[lTwoName] = {"class": "linear",
                         "activation": "relu",
                         "from": lOneName,
                         "n_out": outputSize}
    
    if l2 is not None:
        network[lOneName]["L2"] = l2
        network[lTwoName]["L2"] = l2
    
    return network


def get_monophone_net(shared_network, add_mlps=False, use_multi_task=True,
                      final_ctx_type=None, ph_emb_size=64, st_emb_size=512,
                      focal_loss_factor=2.0, label_smoothing=0.0, l2=None, shared_delta_encoder=False):
    network = copy.copy(shared_network)
    network["encoder-output"] = {"class": "copy", "from": ["fwd_6", "bwd_6"]}
    
    encoder_out_len = shared_network['fwd_1']['n_out'] * 2
    assert final_ctx_type is not None
    
    lossOpts = {}
    if focal_loss_factor > 0.0:
        lossOpts["focal_loss_factor"] = focal_loss_factor
    if label_smoothing > 0.0:
        lossOpts["label_smoothing"] = label_smoothing
    
    if add_mlps:
        if final_ctx_type.value == context_mapper.get_enum(3):
            triOut = encoder_out_len + ph_emb_size + ph_emb_size
            set_Mlp_component(network, "triphone", triOut, l2=l2)

            network["center-output"] = {"class": "softmax",
                                        "from": "linear2-triphone",
                                        "target": "classes",
                                        "loss": "ce",
                                        "loss_opts": copy.copy(lossOpts)}

            if use_multi_task:
                set_Mlp_component(network, "contexts", encoder_out_len, l2=l2)
                network["right-output"] = {"class": "softmax",
                                           "from": "linear2-contexts",
                                           "target": "futureLabel",
                                           "loss": "ce",
                                           "loss_opts": copy.copy(lossOpts)}
                network["left-output"] = {"class": "softmax",
                                          "from": "linear2-contexts",
                                          "target": "pastLabel",
                                          "loss": "ce",
                                          "loss_opts": copy.copy(lossOpts)}
                network["center-output"]["target"] = "centerState"


        
        elif final_ctx_type.value == context_mapper.get_enum(4):
            diOut = encoder_out_len + ph_emb_size
            set_Mlp_component(network, "diphone", diOut, l2=l2)
            network["center-output"] = {"class": "softmax",
                                        "from": "linear2-diphone",
                                        "target": "classes",
                                        "loss": "ce",
                                        "loss_opts": copy.copy(lossOpts)}

            if use_multi_task:
                triOut = encoder_out_len + ph_emb_size + st_emb_size
                set_Mlp_component(network, "leftContext", encoder_out_len, l2=l2)
                set_Mlp_component(network, "triphone", triOut, l2=l2)
                network["left-output"] = {"class": "softmax",
                                          "from": "linear2-leftContext",
                                          "target": "pastLabel",
                                          "loss": "ce",
                                          "loss_opts": copy.copy(lossOpts)}

                network["right-output"] = {"class": "softmax",
                                           "from": "linear2-triphone",
                                           "target": "futureLabel",
                                           "loss": "ce",
                                           "loss_opts": copy.copy(lossOpts)}
                network["center-output"]["target"] = "centerState"
            

        
        elif final_ctx_type.value == context_mapper.get_enum(5):
            assert (use_multi_task, "it is not possible to have a monophone backward without multitask")
            set_Mlp_component(network, "centerState", encoder_out_len, l2=l2)
            set_Mlp_component(network, "diphone", 1030, l2=l2)
            set_Mlp_component(network, "triphone", 1040, l2=l2)
            network["center-output"] = {"class": "softmax",
                                        "from": "linear2-centerState",
                                        "target": "centerState",
                                        "loss": "ce",
                                        "loss_opts": copy.copy(lossOpts)}
            network["left-output"] = {"class": "softmax",
                                      "from": "linear2-triphone",
                                      "target": "pastLabel",
                                      "loss": "ce",
                                      "loss_opts": copy.copy(lossOpts)}

            network["right-output"] = {"class": "softmax",
                                       "from": "linear2-diphone",
                                       "target": "futureLabel",
                                       "loss": "ce",
                                       "loss_opts": copy.copy(lossOpts)}

        elif final_ctx_type.value == context_mapper.get_enum(6):
            delta_blstm_n = "deltaEncoder-output"
            diOut = encoder_out_len + ph_emb_size
            if shared_delta_encoder:
                add_delta_blstm_(network, name=delta_blstm_n, l2=l2, source_layer=['fwd_6', 'bwd_6'])
                set_Mlp_component(network, "diphone", diOut, sourceLayer=delta_blstm_n, l2=l2)
            else:
                add_delta_blstm_(network, name=delta_blstm_n, l2=l2)
                set_Mlp_component(network, "diphone", diOut, l2=l2)
            network["center-output"] = {"class": "softmax",
                                        "from": "linear2-diphone",
                                        "target": "classes",
                                        "loss": "ce",
                                        "loss_opts": copy.copy(lossOpts)}


            if use_multi_task:
                triOut = encoder_out_len + ph_emb_size + st_emb_size
                set_Mlp_component(network, "leftContext", encoder_out_len, l2=l2)
                if shared_delta_encoder:
                    set_Mlp_component(network, "triphone", triOut, sourceLayer=delta_blstm_n, l2=l2)
                else:
                    set_Mlp_component(network, "triphone", triOut, sourceLayer=delta_blstm_n, l2=l2)

                network["left-output"] = {"class": "softmax",
                                          "from": "linear2-leftContext",
                                          "target": "pastLabel",
                                          "loss": "ce",
                                          "loss_opts": copy.copy(lossOpts)}

                network["right-output"] = {"class": "softmax",
                                           "from": "linear2-triphone",
                                           "target": "futureLabel",
                                           "loss": "ce",
                                           "loss_opts": copy.copy(lossOpts)}
                network["center-output"]["target"] = "centerState"
    else:
        network["center-output"] = {"class": "softmax",
                                    "from": "encoder-output",
                                    "target": "classes",
                                    "loss": "ce",
                                    "loss_opts": copy.copy(lossOpts)}
        if use_multi_task:
            network["left-output"] = {"class": "softmax",
                                         "from": "encoder-output",
                                         "target": "pastLabel",
                                         "loss": "ce",
                                         "loss_opts": copy.copy(lossOpts)}

            network["right-output"] = {"class": "softmax",
                                        "from": "encoder-output",
                                        "target": "futureLabel",
                                        "loss": "ce",
                                        "loss_opts": copy.copy(lossOpts)}
            network["center-output"]["target"] = "centerState"


    
    return network


def get_diphone_net(shared_network, use_multi_task, l2, label_smoothing, ph_emb_size=64, st_emb_size=256):
    network = copy.copy(shared_network)
    network["pastEmbed"] = get_embedding_layer(source="pastLabel", dim=ph_emb_size, l2=l2)
    network["linear1-diphone"]["from"] = ["encoder-output", "pastEmbed"]
    if use_multi_task:
        network["currentState"] = get_embedding_layer(source="centerState", dim=st_emb_size, l2=l2)
        network["linear1-triphone"]["from"] = ["encoder-output", "currentState"]
    else:
        encoder_out_len = shared_network['fwd_1']['n_out'] * 2
        loss_opts = network["center-output"]["loss_opts"]
        loss_opts["label_smoothing"] = label_smoothing
        set_Mlp_component(network, "leftContext", encoder_out_len, l2=l2)
        network["left-output"] = {"class": "softmax",
                                  "from": "linear2-leftContext",
                                  "target": "pastLabel",
                                  "loss": "ce",
                                  "loss_opts": copy.copy(loss_opts)}

    return network


def get_forward_net(shared_network, l2, ph_emb_size=64, st_emb_size=256):
    network = copy.copy(shared_network)

    # Embeddings
    network["pastEmbed"]    = get_embedding_layer(source="pastLabel", dim=ph_emb_size, l2=l2)
    network["currentState"] = get_embedding_layer(source="centerState", dim=st_emb_size, l2=l2)

    network["linear1-diphone"]["from"] = ["encoder-output", "pastEmbed"]
    network["linear1-triphone"]["from"] = ["encoder-output", "currentState"]

    return network




def get_symmetric_net(shared_network):
    network = copy.copy(shared_network)
    
    network["encoder-output"] = {"class": "copy", "from": ["fwd_6", "bwd_6"]}
    
    # Embeddings
    network["pastEmbed"] = {"class": "linear", "activation": None, "from": ["data:lastLabel"], "n_out": 10}
    network["futureEmbed"] = {"class": "linear", "activation": None, "from": ["data:futureLabel"], "n_out": 10}
    
    # triphone output
    network["linear1-triphone"] = {"class": "linear",
                                   "activation": "relu",
                                   "from": ["encoder-output", "pastEmbed", "futureEmbed"],
                                   "n_out": 1020}
    
    network["linear2-triphone"] = {"class": "linear",
                                   "activation": "relu",
                                   "from": "linear1-triphone",
                                   "n_out": 1020}
    
    network["triphone-output"] = {"class": "softmax",
                                  "from": "linear2-triphone",
                                  "target": "alignment",
                                  "loss": "ce",
                                  "loss_opts": {"focal_loss_factor": 2.0}}
    
    network["future-output"] = {"class": "softmax",
                                "from": "encoder-output",
                                "target": "futureLabel",
                                "loss": "ce",
                                "loss_opts": {"focal_loss_factor": 2.0}}
    # change this to past-output
    network["context-output"] = {"class": "softmax",
                                 "from": "encoder-output",
                                 "target": "lastLabel",
                                 "loss": "ce",
                                 "loss_opts": {"focal_loss_factor": 2.0}}
    
    return network



def get_backward_net(shared_network):
    network = copy.copy(shared_network)
    
    network["encoder-output"] = {"class": "copy", "from": ["fwd_6", "bwd_6"]}
    
    # Embeddings
    network["futureEmbed"] = {"class": "linear", "activation": None, "from": ["data:futureLabel"], "n_out": 10}
    network["currentState"] = {"class": "linear", "activation": None, "from": ["data:alignment"], "n_out": 30}
    
    # change this to past-output
    network["currentState-output"] = {"class": "softmax",
                                      "from": "encoder-output",
                                      "target": "alignment",
                                      "loss": "ce",
                                      "loss_opts": {"focal_loss_factor": 2.0}}
    
    # diphone output
    network["linear1-diphone"] = {"class": "linear",
                                  "activation": "relu",
                                  "from": ["encoder-output", "currentState"],
                                  "n_out": 1030}
    
    network["linear2-diphone"] = {"class": "linear",
                                  "activation": "relu",
                                  "from": "linear1-diphone",
                                  "n_out": 1030}
    
    network["diphone-output"] = {"class": "softmax",
                                 "from": "linear2-diphone",
                                 "target": "futureLabel",
                                 "loss": "ce",
                                 "loss_opts": {"focal_loss_factor": 2.0}}
    
    # triphone output
    network["linear1-triphone"] = {"class": "linear",
                                   "activation": "relu",
                                   "from": ["encoder-output", "currentState", "futureEmbed"],
                                   "n_out": 1040}
    
    network["linear2-triphone"] = {"class": "linear",
                                   "activation": "relu",
                                   "from": "linear1-triphone",
                                   "n_out": 1040}
    
    network["triphone-output"] = {"class": "softmax",
                                  "from": "linear2-triphone",
                                  "target": "lastLabel",
                                  "loss": "ce",
                                  "loss_opts": {"focal_loss_factor": 2.0}}
    
    return network


############ BW alignment for factorized hybrid #################
def get_bw_params_for_cartfree(csp, returnn_config, output_names, loss_wrt_to_act_in=False,
                               am_scale=1.0,
                               import_model=None, exp_average=0.001,
                               prior_scale=1.0, tdp_scale=1.0,
                               mappedSilence=False,
                               extra_config=None, extra_post_config=None):
    if returnn_config.config['use_tensorflow']:
        inputs = []
        for out in output_names:
            out_denot = out.split("-")[0]
            # prior calculation
            accu_name = ("_").join(['acc-prior', out_denot])
            returnn_config.config['network'][accu_name] = {'class': 'accumulate_mean',
                                                        'exp_average': exp_average,
                                                        'from': out,
                                                        'is_prob_distribution': True}

            comb_name = ("_").join(['comb-prior', out_denot])
            inputs.append(comb_name)
            returnn_config.config['network'][comb_name] = {'class': 'combine',
                                                        'kind': 'eval',
                                                        'eval': 'am_scale*( safe_log(source(0)) - (safe_log(source(1)) * prior_scale) )',
                                                        'eval_locals': {'am_scale': am_scale,
                                                                        'prior_scale': prior_scale},
                                                        'from': [out, accu_name]}

            bw_out = ("_").join(['output-bw', out_denot])
            returnn_config.config['network'][bw_out] = {'class': 'copy',
                                                     'from': out,
                                                     'loss': 'via_layer',
                                                     'loss_opts': {'align_layer': ("/").join(['fast-bw', out_denot]),
                                                                   'loss_wrt_to_act_in': loss_wrt_to_act_in},
                                                     'loss_scale': 1.0}

        returnn_config.config['network']['fast-bw'] = {'class': 'fast_bw_factorized',
                                                    'align_target': 'monophone',
                                                    'from': inputs,
                                                    'tdp_scale': tdp_scale,
                                                    'mappedSilence': mappedSilence}

        returnn_config.config['network']["fast-bw"]['sprint_opts'] = {
            "sprintExecPath": RasrCommand.select_exe(csp.nn_trainer_exe, 'nn-trainer'),
            "sprintConfigStr": "--config=fastbw.config",
            "sprintControlConfig": {"verbose": True},
            "usePythonSegmentOrder": False,
            "numInstances": 1}


    else:  # Use Theano
        assert False, "Theano implementation of bw training not supportet yet."

    if 'chunking' in returnn_config.config:
        del returnn_config.config['chunking']
    if 'pretrain' in returnn_config.config and import_model is not None:
        del returnn_config.config['pretrain']

    # start training from existing model
    if import_model is not None:
        returnn_config.config['import_model_train_epoch1'] = str(import_model)[
                                                          :-5 if returnn_config.config['use_tensorflow'] else None]

    # Create additional Sprint config file to compute losses
    mapping = {'corpus': 'neural-network-trainer.corpus',
               'lexicon': [
                   'neural-network-trainer.alignment-fsa-exporter.model-combination.lexicon'],
               'acoustic_model': [
                   'neural-network-trainer.alignment-fsa-exporter.model-combination.acoustic-model']
               }
    config, post_config = sp.build_config_from_mapping(csp, mapping)
    post_config['*'].output_channel.file = 'fastbw.log'

    # Define action
    config.neural_network_trainer.action = 'python-control'
    # neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder
    config.neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder.orthographic_parser.allow_for_silence_repetitions = False
    config.neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder.orthographic_parser.normalize_lemma_sequence_scores = True
    # neural_network_trainer.alignment_fsa_exporter.alignment-fsa-exporter
    config.neural_network_trainer.alignment_fsa_exporter.alignment_fsa_exporter.model_combination.acoustic_model.fix_allophone_context_at_word_boundaries = True
    config.neural_network_trainer.alignment_fsa_exporter.alignment_fsa_exporter.model_combination.acoustic_model.transducer_builder_filter_out_invalid_allophones = True

    # additional config
    config._update(extra_config)
    post_config._update(extra_post_config)

    config["neural-network-trainer"]["*"]["seed"] = 29
    config["neural-network-trainer"]["*"]["corpus"]["select-partition"] = 5

    additional_sprint_config_files = {'fastbw': config}
    additional_sprint_post_config_files = {'fastbw': post_config}

    return returnn_config, additional_sprint_config_files, additional_sprint_post_config_files


def get_bw_params_for_monophone(csp, returnn_config, loss_wrt_to_act_in=False,
                                am_scale=1.0, prior_scale=1.0, tdp_scale=1.0,
                                import_model=None, exp_average=0.001,
                                out='center-output', fix_tdp_bug=False, fixed_prior=None, normalize_lemma_scores=True,
                                extra_config=None, extra_post_config=None):
    if returnn_config.config['use_tensorflow']:
        inputs = []
        out_denot = out.split('-')[0]
        # prior calculation
        accu_name = ("_").join(['acc-prior', out_denot])
        if fixed_prior is None:
            returnn_config.config['network'][accu_name] = {'class': 'accumulate_mean',
                                                        'exp_average': exp_average,
                                                        'from': out,
                                                        'is_prob_distribution': True}
        else:
            returnn_config.config['network'][accu_name] = {'class': 'constant',
                                                        'dtype': 'float32',
                                                        'value': fixed_prior}

        comb_name = ("_").join(['comb-prior', out_denot])
        inputs.append(comb_name)
        returnn_config.config['network'][comb_name] = {'class': 'combine',
                                                    'kind': 'eval',
                                                    'eval': 'am_scale*( safe_log(source(0)) - (safe_log(source(1)) * prior_scale) )',
                                                    'eval_locals': {'am_scale': am_scale,
                                                                    'prior_scale': prior_scale},
                                                    'from': [out, accu_name]}

        returnn_config.config['network']['output_bw'] = {'class': 'copy',
                                                      'from': out,
                                                      'loss': 'via_layer',
                                                      'loss_opts': {'align_layer': 'fast_bw',
                                                                    'loss_wrt_to_act_in': loss_wrt_to_act_in},
                                                      'loss_scale': 1.0}
        returnn_config.config['network']['fast_bw'] = {'class': 'fast_bw',
                                                    'align_target': 'sprint',
                                                    'from': inputs,
                                                    'tdp_scale': tdp_scale}
        returnn_config.config['network']["fast_bw"]['sprint_opts'] = {
            "sprintExecPath": RasrCommand.select_exe(csp.nn_trainer_exe, 'nn-trainer'),
            "sprintConfigStr": "--config=fastbw.config",
            "sprintControlConfig": {"verbose": True},
            "usePythonSegmentOrder": False,
            "numInstances": 1}


    else:  # Use Theano
        assert False, "Please set use_tensorflow to True in your config."

    if 'chunking' in returnn_config.config:
        del returnn_config.config['chunking']
    if 'pretrain' in returnn_config.config and import_model is not None:
        del returnn_config.config['pretrain']

    # start training from existing model
    if import_model is not None:
        returnn_config.config['import_model_train_epoch1'] = import_model

    # Create additional Sprint config file to compute losses
    mapping = {'corpus': 'neural-network-trainer.corpus',
               'lexicon': [
                   'neural-network-trainer.alignment-fsa-exporter.model-combination.lexicon'],
               'acoustic_model': [
                   'neural-network-trainer.alignment-fsa-exporter.model-combination.acoustic-model']
               }
    config, post_config = sp.build_config_from_mapping(csp, mapping)
    post_config['*'].output_channel.file = 'fastbw.log'

    # Define action
    config.neural_network_trainer.action = 'python-control'
    # neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder
    config.neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder.orthographic_parser.allow_for_silence_repetitions = False
    config.neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder.orthographic_parser.normalize_lemma_sequence_scores = normalize_lemma_scores
    # neural_network_trainer.alignment_fsa_exporter.alignment-fsa-exporter
    config.neural_network_trainer.alignment_fsa_exporter.alignment_fsa_exporter.model_combination.acoustic_model.fix_allophone_context_at_word_boundaries = True
    config.neural_network_trainer.alignment_fsa_exporter.alignment_fsa_exporter.model_combination.acoustic_model.transducer_builder_filter_out_invalid_allophones = True
    config.neural_network_trainer.alignment_fsa_exporter.alignment_fsa_exporter.model_combination.acoustic_model.fix_tdp_leaving_epsilon_arc = fix_tdp_bug

    # additional config
    config._update(extra_config)
    post_config._update(extra_post_config)

    additional_sprint_config_files = {'fastbw': config}
    additional_sprint_post_config_files = {'fastbw': post_config}

    return returnn_config, additional_sprint_config_files, additional_sprint_post_config_files



def get_bw_params_for_monophone_debugged(csp, returnn_config, loss_wrt_to_act_in=False,
                                am_scale=1.0, prior_scale=1.0, tdp_scale=1.0,
                                import_model=None, exp_average=0.001,
                                out='center-output', fix_tdp_bug=False, fixed_prior=None, normalize_lemma_scores=True,
                                extra_config=None, extra_post_config=None):
    if returnn_config.config['use_tensorflow']:
        inputs = []
        out_denot = out.split('-')[0]
        # prior calculation
        accu_name = ("_").join(['acc-prior', out_denot])
        if fixed_prior is None:
            returnn_config.config['network'][accu_name] = {'class': 'accumulate_mean',
                                                        'exp_average': exp_average,
                                                        'from': out,
                                                        'is_prob_distribution': True}
        else:
            returnn_config.config['network'][accu_name] = {'class': 'constant',
                                                        'dtype': 'float32',
                                                        'value': fixed_prior}

        comb_name = ("_").join(['comb-prior', out_denot])
        inputs.append(comb_name)
        returnn_config.config['network'][comb_name] = {'class': 'combine',
                                                    'kind': 'eval',
                                                    'eval': 'am_scale*( safe_log(source(0)) - (safe_log(source(1)) * prior_scale) )',
                                                    'eval_locals': {'am_scale': am_scale,
                                                                    'prior_scale': prior_scale},
                                                    'from': [out, accu_name]}

        returnn_config.config['network']['output_bw'] = {'class': 'copy',
                                                      'from': out,
                                                      'loss': 'via_layer',
                                                      'loss_opts': {'align_layer': 'fast_bw',
                                                                    'loss_wrt_to_act_in': loss_wrt_to_act_in},
                                                      'loss_scale': 1.0}
        returnn_config.config['network']['fast_bw'] = {'class': 'fast_bw',
                                                    'align_target': 'sprint',
                                                    'from': inputs,
                                                    'tdp_scale': tdp_scale}
        returnn_config.config['network']["fast_bw"]['sprint_opts'] = {
            "sprintExecPath": RasrCommand.select_exe(csp.nn_trainer_exe, 'nn-trainer'),
            "sprintConfigStr": "--config=fastbw.config",
            "sprintControlConfig": {"verbose": True},
            "usePythonSegmentOrder": False,
            "numInstances": 1}


    else:  # Use Theano
        assert False, "Please set use_tensorflow to True in your config."

    if 'chunking' in returnn_config.config:
        del returnn_config.config['chunking']
    if 'pretrain' in returnn_config.config and import_model is not None:
        del returnn_config.config['pretrain']

    # start training from existing model
    if import_model is not None:
        returnn_config.config['import_model_train_epoch1'] = import_model

    # Create additional Sprint config file to compute losses
    mapping = {'corpus': 'neural-network-trainer.corpus',
               'lexicon': [
                   'neural-network-trainer.alignment-fsa-exporter.model-combination.lexicon'],
               'acoustic_model': [
                   'neural-network-trainer.alignment-fsa-exporter.model-combination.acoustic-model']
               }
    config, post_config = sp.build_config_from_mapping(csp, mapping)
    post_config['*'].output_channel.file = 'fastbw.log'

    # Define action
    config.neural_network_trainer.action = 'python-control'
    # neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder
    config.neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder.orthographic_parser.allow_for_silence_repetitions = False
    config.neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder.orthographic_parser.normalize_lemma_sequence_scores = normalize_lemma_scores
    # neural_network_trainer.alignment_fsa_exporter.alignment-fsa-exporter
    config.neural_network_trainer.alignment_fsa_exporter.model_combination.acoustic_model.fix_allophone_context_at_word_boundaries = True
    config.neural_network_trainer.alignment_fsa_exporter.model_combination.acoustic_model.transducer_builder_filter_out_invalid_allophones = True
    config.neural_network_trainer.alignment_fsa_exporter.model_combination.acoustic_model.fix_tdp_leaving_epsilon_arc = fix_tdp_bug

    # additional config
    config._update(extra_config)
    post_config._update(extra_post_config)

    additional_sprint_config_files = {'fastbw': config}
    additional_sprint_post_config_files = {'fastbw': post_config}

    return returnn_config, additional_sprint_config_files, additional_sprint_post_config_files




def get_bw_params_for_monophone_noprior(csp, crnn_config, loss_wrt_to_act_in=False,
                                am_scale=1.0, tdp_scale=1.0,
                                import_model=None, out='center-output', fix_tdp_bug=False, normalize_lemma_scores=True,
                                extra_config=None, extra_post_config=None):
    if crnn_config.config['use_tensorflow']:
        inputs = []
        out_denot = out.split('-')[0]
        # prior calculation

        comb_name = ("_").join(['multiply-scale', out_denot])
        inputs.append(comb_name)
        crnn_config.config['network'][comb_name] = {'class': 'combine',
                                                    'kind': 'eval',
                                                    'eval': 'am_scale*(safe_log(source(0)))',
                                                    'eval_locals': {'am_scale': am_scale},
                                                    'from': [out]}

        crnn_config.config['network']['output_bw'] = {'class': 'copy',
                                                      'from': out,
                                                      'loss': 'via_layer',
                                                      'loss_opts': {'align_layer': 'fast_bw',
                                                                    'loss_wrt_to_act_in': loss_wrt_to_act_in},
                                                      'loss_scale': 1.0}
        crnn_config.config['network']['fast_bw'] = {'class': 'fast_bw',
                                                    'align_target': 'sprint',
                                                    'from': inputs,
                                                    'tdp_scale': tdp_scale}
        crnn_config.config['network']["fast_bw"]['sprint_opts'] = {
            "sprintExecPath": SprintCommand.select_exe(csp.nn_trainer_exe, 'nn-trainer'),
            "sprintConfigStr": "--config=fastbw.config",
            "sprintControlConfig": {"verbose": True},
            "usePythonSegmentOrder": False,
            "numInstances": 1}


    else:  # Use Theano
        assert False, "Please set use_tensorflow to True in your config."

    if 'chunking' in crnn_config.config:
        del crnn_config.config['chunking']
    if 'pretrain' in crnn_config.config and import_model is not None:
        del crnn_config.config['pretrain']

    # start training from existing model
    if import_model is not None:
        crnn_config.config['import_model_train_epoch1'] = import_model

    # Create additional Sprint config file to compute losses
    mapping = {'corpus': 'neural-network-trainer.corpus',
               'lexicon': [
                   'neural-network-trainer.alignment-fsa-exporter.model-combination.lexicon'],
               'acoustic_model': [
                   'neural-network-trainer.alignment-fsa-exporter.model-combination.acoustic-model']
               }
    config, post_config = sp.build_config_from_mapping(csp, mapping)
    post_config['*'].output_channel.file = 'fastbw.log'

    # Define action
    config.neural_network_trainer.action = 'python-control'
    # neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder
    config.neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder.orthographic_parser.allow_for_silence_repetitions = False
    config.neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder.orthographic_parser.normalize_lemma_sequence_scores = normalize_lemma_scores
    # neural_network_trainer.alignment_fsa_exporter.alignment-fsa-exporter
    config.neural_network_trainer.alignment_fsa_exporter.alignment_fsa_exporter.model_combination.acoustic_model.fix_allophone_context_at_word_boundaries = True
    config.neural_network_trainer.alignment_fsa_exporter.alignment_fsa_exporter.model_combination.acoustic_model.transducer_builder_filter_out_invalid_allophones = True
    config.neural_network_trainer.alignment_fsa_exporter.alignment_fsa_exporter.model_combination.acoustic_model.fix_tdp_leaving_epsilon_arc = fix_tdp_bug

    # additional config
    config._update(extra_config)
    post_config._update(extra_post_config)

    # config["neural-network-trainer"]["*"]["seed"] = 29
    # config["neural-network-trainer"]["*"]["corpus"]["select-partition"] = 5

    additional_sprint_config_files = {'fastbw': config}
    additional_sprint_post_config_files = {'fastbw': post_config}

    return crnn_config, additional_sprint_config_files, additional_sprint_post_config_files




