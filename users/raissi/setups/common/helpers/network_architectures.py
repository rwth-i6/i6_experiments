import i6_core.returnn as returnn
import i6_core.rasr as sp
from i6_core.rasr.command import RasrCommand
from sisyphus import *

import copy
from IPython import embed




def blstm_config(network, partition_epochs, lr=5e-4, batch_size=10000, max_seqs=100, chunking="64:32", **kwargs):
    result = {"batch_size": batch_size,
              "max_seqs": max_seqs,
              "cache_size": "0",
              "batching": "random",
              "chunking": chunking,
              "learning_rate": lr,
              "learning_rate_control": "newbob_multi_epoch",
              "newbob_multi_num_epochs": partition_epochs["train"],
              "newbob_multi_update_interval": partition_epochs["dev"],
              "learning_rate_control_relative_error_relative_lr": True,
              "learning_rate_control_min_num_epochs_per_new_lr": 3,
              "use_tensorflow": True,
              "multiprocessing": True,
    
              "network": network}
    result.update(**kwargs)
    
    return result


def blstm_network(layers=6 * [512], dropout=0.1, l2=0.1, unit_type="lstmp", specaugment=False):
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


"""
        u32 result = e;
        u32 boundaryClass = result % 4;
        result = (result - boundaryClass)/4;
        u32 stateClass = result % 3;
        result = (result - stateClass)/3;
        u32 futureLabel = result % 47;
        result = (result - futureLabel)/ 47;
        u32 pastLabel = result % 47;
        u32 centerPhoneme = (result - pastLabel)/ 47;
"""


def add_delta_blstm_(network, name, l2=0.01, source_layer=None):
    if source_layer is None:
        source_layer = 'encoder-output'
    
    network["fwd_delta"] = {'L2': l2, 'class': 'rec', 'direction': 1, 'dropout': 0.1,
                            'from': source_layer, 'n_out': 512, 'unit': 'nativelstm2'}
    
    network["bwd_delta"] = {'L2': l2, 'class': 'rec', 'direction': -1, 'dropout': 0.1,
                            'from': source_layer, 'n_out': 512, 'unit': 'nativelstm2'}
    
    network[name] = {'class': 'copy', 'from': ['fwd_delta', 'bwd_delta']}
    
    return network


def get_common_subnetwork_for_targets_with_blstm(layers, dropout, l2, isBoundary=True, nContexts=47, nStates=3,
                                                 unit_type="lstmp",
                                                 specaugment=False, isMinDuration=False, isWordEnd=False):
    acousticNet = blstm_network(layers, dropout, l2, unit_type=unit_type, specaugment=specaugment)
    
    assert (not (isBoundary and isWordEnd))
    
    if isBoundary:
        stateInput = "popBoundry"
        acousticNet["boundryClass"] = {"class": "eval", "from": "data:classes", "eval": "tf.floormod(source(0),%d)" % 4,
                                       "out_type": {'dim': 4, 'dtype': 'int32', 'sparse': True}}
        acousticNet["popBoundry"] = {"class": "eval", "from": ["data:classes"], "eval": "tf.floordiv(source(0),%d)" % 4,
                                     "out_type": {'dim': (nContexts ** 3) * nStates, 'dtype': 'int32',
                                                  'sparse': True}}
    
    elif isWordEnd:
        stateInput = "popWordEnd"
        acousticNet["wordEndClass"] = {"class": "eval", "from": "data:classes", "eval": "tf.floormod(source(0),%d)" % 2,
                                       "out_type": {'dim': 2, 'dtype': 'int32', 'sparse': True}}
        acousticNet["popWordEnd"] = {"class": "eval", "from": ["data:classes"], "eval": "tf.floordiv(source(0),%d)" % 2,
                                     "out_type": {'dim': (nContexts ** 3) * nStates, 'dtype': 'int32',
                                                  'sparse': True}}
    
    else:
        stateInput = "data:classes"
    
    acousticNet["stateId"] = {"class": "eval", "from": stateInput, "eval": "tf.floormod(source(0),%d)" % nStates,
                              "out_type": {'dim': nStates, 'dtype': 'int32', 'sparse': True}}
    acousticNet["popStateId"] = {"class": "eval", "from": stateInput, "eval": "tf.floordiv(source(0),%d)" % nStates,
                                 "out_type": {'dim': (nContexts ** 3), 'dtype': 'int32',
                                              'sparse': True}}
    acousticNet["futureLabel"] = {"class": "eval", "from": "popStateId",
                                  "eval": "tf.floormod(source(0),%d)" % nContexts,
                                  "register_as_extern_data": "futureLabel",
                                  "out_type": {'dim': nContexts, 'dtype': 'int32', 'sparse': True}}
    acousticNet["popFutureLabel"] = {"class": "eval", "from": "popStateId",
                                     "eval": "tf.floordiv(source(0),%d)" % nContexts,
                                     "out_type": {'dim': (nContexts ** 2), 'dtype': 'int32', 'sparse': True}}
    
    acousticNet["pastLabel"] = {"class": "eval", "from": "popFutureLabel",
                                "eval": "tf.floormod(source(0),%d)" % nContexts, "register_as_extern_data": "pastLabel",
                                "out_type": {'dim': nContexts, 'dtype': 'int32', 'sparse': True}}
    
    acousticNet["centerPhoneme"] = {"class": "eval", "from": "popFutureLabel",
                                    "eval": "tf.floordiv(source(0),%d)" % nContexts,
                                    "out_type": {'dim': nContexts, 'dtype': 'int32', 'sparse': True}}
    
    if isMinDuration:
        if isWordEnd:
            acousticNet["centerState"] = {"class": "eval", "from": ["centerPhoneme", "wordEndClass"],
                                          "eval": "(source(0)*2)+source(1)",
                                          "register_as_extern_data": "centerState",
                                          "out_type": {'dim': nContexts * 2, 'dtype': 'int32', 'sparse': True}}
        else:
            acousticNet["centerPhoneme"]["register_as_extern_data"] = "centerState"
    else:
        if isWordEnd:
            acousticNet["centerState"] = {"class": "eval", "from": ["centerPhoneme", "stateId", "wordEndClass"],
                                          "eval": "(((source(0)*2)+source(2))*3)+source(1)",
                                          "register_as_extern_data": "centerState",
                                          "out_type": {'dim': nContexts * nStates * 2, 'dtype': 'int32',
                                                       'sparse': True}}
        else:
            acousticNet["centerState"] = {"class": "eval", "from": ["centerPhoneme", "stateId"],
                                          "eval": "(source(0)*3)+source(1)", "register_as_extern_data": "centerState",
                                          "out_type": {'dim': nContexts * nStates, 'dtype': 'int32', 'sparse': True}}
    
    return acousticNet


def make_config(contextType, contextMapper, python_prolog, python_epilog, partition_epochs,
                num_input=40, nStates=3, nContexts=47,
                isBoundary=False, isMinDuration=False, isWordEnd=False,
                layers=6 * [500], l2=0.01, mlpL2=0.01, dropout=0.1,
                ctxEmbSize=10, stateEmbSize=30, focalLossFactor=2.0, labelSmoothing=0.0,
                addMLPs=False, finalContextType=None, sprint=False,
                unit_type="lstmp", specaugment=False, sharedDeltaEncoder=False, **kwargs):
    if sprint:
        sharedNetwork = get_common_subnetwork_for_targets_with_blstm(layers,
                                                                     dropout,
                                                                     l2,
                                                                     isBoundary=isBoundary,
                                                                     nContexts=nContexts,
                                                                     nStates=nStates,
                                                                     unit_type=unit_type,
                                                                     specaugment=specaugment,
                                                                     isMinDuration=isMinDuration,
                                                                     isWordEnd=isWordEnd)
    else:
        sharedNetwork = blstm_network(layers, dropout, l2, unit_type=unit_type, specaugment=specaugment)
    
    config = get_config_for_context_type(contextType, contextMapper, partition_epochs,
                                         sharedNetwork,
                                         stateEmbSize=stateEmbSize,
                                         focalLossFactor=focalLossFactor, labelSmoothing=labelSmoothing,
                                         addMLPs=addMLPs, finalContextType=finalContextType, l2=mlpL2,
                                         sharedDeltaEncoder=sharedDeltaEncoder, **kwargs)
    
    returnnConfig = returnn.ReturnnConfig(config, python_prolog=python_prolog, python_epilog=python_epilog)
    
    return returnnConfig


def get_graph_from_returnn_config(returnnConfig, python_prolog, python_epilog):
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


def get_config_for_context_type(contextType, contextMapper, partition_epochs,
                                sharedNetwork, ctxEmbSize=10, stateEmbSize=30,
                                focalLossFactor=2.0, labelSmoothing=0.2,
                                addMLPs=False, finalContextType=None, l2=0.01, sharedDeltaEncoder=False, **kwargs):
    if contextType.value == contextMapper.get_enum(1):
        network = get_monophone_net(sharedNetwork,
                                    addMLPs=addMLPs,
                                    finalCtxType=finalContextType,
                                    contextMapper=contextMapper,
                                    ctxEmbSize=ctxEmbSize,
                                    stateEmbSize=stateEmbSize,
                                    focalLossFactor=focalLossFactor,
                                    labelSmoothing=labelSmoothing,
                                    l2=l2,
                                    sharedDeltaEncoder=sharedDeltaEncoder)
    
    
    elif contextType.value == contextMapper.get_enum(2):
        network = get_diphone_net(sharedNetwork)
    
    elif contextType.value == contextMapper.get_enum(3):
        network = get_symmetric_net(sharedNetwork)
    
    elif contextType.value == contextMapper.get_enum(4):
        network = get_forward_net(sharedNetwork)
    
    elif contextType.value == contextMapper.get_enum(5):
        network = get_backward_net(sharedNetwork)
    
    else:
        return None

    # ToDo: once you hvae your optimal setting add it here using bin_ce_weight
    # if finalCtxType.value == contextMapper.get_enum(6):
    
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


def get_monophone_net(sharedNetwork, addMLPs=False,
                      finalCtxType=None, contextMapper=None, ctxEmbSize=10, stateEmbSize=30,
                      focalLossFactor=2.0, labelSmoothing=0.0, l2=None, sharedDeltaEncoder=False):
    network = copy.copy(sharedNetwork)
    network["encoder-output"] = {"class": "copy", "from": ["fwd_6", "bwd_6"]}
    
    encoder_out_len = sharedNetwork['fwd_1']['n_out'] * 2
    
    lossOpts = {}
    if focalLossFactor > 0.0:
        lossOpts["focal_loss_factor"] = focalLossFactor
    if labelSmoothing > 0.0:
        lossOpts["label_smoothing"] = labelSmoothing
    
    if addMLPs:

        assert finalCtxType is not None
        assert contextMapper is not None
        # ToDo: complete the options
        if finalCtxType.value == contextMapper.get_enum(3):
            set_Mlp_component(network, "contexts", encoder_out_len, l2=l2)
            set_Mlp_component(network, "triphone", 1020, l2=l2)
            
            network["left-output"] = {"class": "softmax",
                                      "from": "linear2-contexts",
                                      "target": "lastLabel",
                                      "loss": "ce",
                                      "loss_opts": copy.copy(lossOpts)}
            
            network["right-output"] = {"class": "softmax",
                                       "from": "linear2-contexts",
                                       "target": "futureLabel",
                                       "loss": "ce",
                                       "loss_opts": copy.copy(lossOpts)}
            
            network["center-output"] = {"class": "softmax",
                                        "from": "linear2-triphone",
                                        "target": "alignment",
                                        "loss": "ce",
                                        "loss_opts": copy.copy(lossOpts)}
        
        elif finalCtxType.value == contextMapper.get_enum(4):
            diOut = encoder_out_len + ctxEmbSize
            triOut = encoder_out_len + ctxEmbSize + stateEmbSize
            set_Mlp_component(network, "leftContext", encoder_out_len, l2=l2)
            set_Mlp_component(network, "diphone", diOut, l2=l2)
            set_Mlp_component(network, "triphone", triOut, l2=l2)
            network["left-output"] = {"class": "softmax",
                                      "from": "linear2-leftContext",
                                      "target": "lastLabel",
                                      "loss": "ce",
                                      "loss_opts": copy.copy(lossOpts)}
            
            network["right-output"] = {"class": "softmax",
                                       "from": "linear2-triphone",
                                       "target": "futureLabel",
                                       "loss": "ce",
                                       "loss_opts": copy.copy(lossOpts)}
            
            network["center-output"] = {"class": "softmax",
                                        "from": "linear2-diphone",
                                        "target": "alignment",
                                        "loss": "ce",
                                        "loss_opts": copy.copy(lossOpts)}
        
        elif finalCtxType.value == contextMapper.get_enum(5):
            set_Mlp_component(network, "centerState", encoder_out_len, l2=l2)
            set_Mlp_component(network, "diphone", 1030, l2=l2)
            set_Mlp_component(network, "triphone", 1040, l2=l2)
            network["left-output"] = {"class": "softmax",
                                      "from": "linear2-triphone",
                                      "target": "lastLabel",
                                      "loss": "ce",
                                      "loss_opts": copy.copy(lossOpts)}
            
            network["right-output"] = {"class": "softmax",
                                       "from": "linear2-diphone",
                                       "target": "futureLabel",
                                       "loss": "ce",
                                       "loss_opts": copy.copy(lossOpts)}
            
            network["center-output"] = {"class": "softmax",
                                        "from": "linear2-centerState",
                                        "target": "alignment",
                                        "loss": "ce",
                                        "loss_opts": copy.copy(lossOpts)}
        
        
        elif finalCtxType.value == contextMapper.get_enum(6):
            diOut = encoder_out_len + ctxEmbSize
            triOut = encoder_out_len + ctxEmbSize + stateEmbSize
            delta_blstm_n = "deltaEncoder-output"
            
            set_Mlp_component(network, "leftContext", encoder_out_len, l2=l2)
            if sharedDeltaEncoder:
                add_delta_blstm_(network, name=delta_blstm_n, l2=l2, source_layer=['fwd_6', 'bwd_6'])
                set_Mlp_component(network, "diphone", diOut, sourceLayer=delta_blstm_n, l2=l2)
                set_Mlp_component(network, "triphone", triOut, sourceLayer=delta_blstm_n, l2=l2)
            else:
                add_delta_blstm_(network, name=delta_blstm_n, l2=l2)
                set_Mlp_component(network, "diphone", diOut, l2=l2)
                set_Mlp_component(network, "triphone", triOut, sourceLayer=delta_blstm_n, l2=l2)
            
            network["left-output"] = {"class": "softmax",
                                      "from": "linear2-leftContext",
                                      "target": "lastLabel",
                                      "loss": "ce",
                                      "loss_opts": copy.copy(lossOpts)}
            
            network["right-output"] = {"class": "softmax",
                                       "from": "linear2-triphone",
                                       "target": "futureLabel",
                                       "loss": "ce",
                                       "loss_opts": copy.copy(lossOpts)}
            
            network["center-output"] = {"class": "softmax",
                                        "from": "linear2-diphone",
                                        "target": "alignment",
                                        "loss": "ce",
                                        "loss_opts": copy.copy(lossOpts)}
    else:
        network["left-output"] = {"class": "softmax",
                                     "from": "encoder-output",
                                     "target": "lastLabel",
                                     "loss": "ce",
                                     "loss_opts": copy.copy(lossOpts)}

        network["right-output"] = {"class": "softmax",
                                    "from": "encoder-output",
                                    "target": "futureLabel",
                                    "loss": "ce",
                                    "loss_opts": copy.copy(lossOpts)}

        network["center-output"] = {"class": "softmax",
                                      "from": "encoder-output",
                                      "target": "alignment",
                                      "loss": "ce",
                                      "loss_opts": copy.copy(lossOpts)}
    
    return network


def get_diphone_net(sharedNetwork):
    network = copy.copy(sharedNetwork)
    
    network["encoder-output"] = {"class": "copy", "from": ["fwd_6", "bwd_6"]}
    network["pastEmbed"] = {"class": "linear", "activation": None, "from": ["data:lastLabel"], "n_out": 10}
    
    network["linear1"] = {"class": "linear",
                          "activation": "relu",
                          "from": ["encoder-output", "pastEmbed"],
                          "n_out": 1010}
    
    network["linear2"] = {"class": "linear",
                          "activation": "relu",
                          "from": "linear1",
                          "n_out": 1010}
    
    network["center-output"] = {"class": "softmax",
                                "from": "linear2",
                                "target": "alignment",
                                "loss": "ce",
                                "loss_opts": {"focal_loss_factor": 2.0}}
    
    network["left-output"] = {"class": "softmax",
                              "from": "encoder-output",
                              "target": "lastLabel",
                              "loss": "ce",
                              "loss_opts": {"focal_loss_factor": 2.0}}
    
    return network


def get_symmetric_net(sharedNetwork):
    network = copy.copy(sharedNetwork)
    
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


def get_forward_net(sharedNetwork):
    network = copy.copy(sharedNetwork)
    
    network["encoder-output"] = {"class": "copy", "from": ["fwd_6", "bwd_6"]}
    
    # Embeddings
    network["pastEmbed"] = {"class": "linear", "activation": None, "from": ["data:lastLabel"], "n_out": 10}
    network["currentState"] = {"class": "linear", "activation": None, "from": ["data:alignment"], "n_out": 30}
    
    # triphone output
    network["linear1-triphone"] = {"class": "linear",
                                   "activation": "relu",
                                   "from": ["encoder-output", "currentState", "pastEmbed"],
                                   "n_out": 1040}
    
    network["linear2-triphone"] = {"class": "linear",
                                   "activation": "relu",
                                   "from": "linear1-triphone",
                                   "n_out": 1040}
    
    network["triphone-output"] = {"class": "softmax",
                                  "from": "linear2-triphone",
                                  "target": "futureLabel",
                                  "loss": "ce",
                                  "loss_opts": {"focal_loss_factor": 2.0}}
    
    # diphone output
    network["linear1"] = {"class": "linear",
                          "activation": "relu",
                          "from": ["encoder-output", "pastEmbed"],
                          "n_out": 1010}
    
    network["linear2"] = {"class": "linear",
                          "activation": "relu",
                          "from": "linear1",
                          "n_out": 1010}
    
    network["diphone-output"] = {"class": "softmax",
                                 "from": "linear2",
                                 "target": "alignment",
                                 "loss": "ce",
                                 "loss_opts": {"focal_loss_factor": 2.0}}
    
    network["context-output"] = {"class": "softmax",
                                 "from": "encoder-output",
                                 "target": "lastLabel",
                                 "loss": "ce",
                                 "loss_opts": {"focal_loss_factor": 2.0}}
    
    return network


def get_backward_net(sharedNetwork):
    network = copy.copy(sharedNetwork)
    
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




