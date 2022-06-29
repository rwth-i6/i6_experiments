from i6_core.returnn.config import ReturnnConfig


def get_specific_returnn_config(returnn_config, epoch=None):
    """
    converts a config with a staged network into a config for a specific epoch

    :param ReturnnConfig returnn_config:
    :param epoch: epoch, if None use last one
    :return:
    """
    if not returnn_config.staged_network_dict:
        return returnn_config
    training_returnn_config= returnn_config
    config_dict = returnn_config.config.copy()
    # TODO: only last network for now, fix with epoch
    if epoch:
        index = 0
        raise NotImplementedError
    else:
        index = max(training_returnn_config.staged_network_dict.keys())
    config_dict['network'] = training_returnn_config.staged_network_dict[index]
    returnn_config = ReturnnConfig(config=config_dict,
                                   post_config=training_returnn_config.post_config,
                                   staged_network_dict=None,
                                   python_prolog=training_returnn_config.python_prolog,
                                   python_epilog=training_returnn_config.python_epilog,
                                   python_epilog_hash=training_returnn_config.python_epilog_hash,
                                   python_prolog_hash=training_returnn_config.python_prolog_hash)
    return returnn_config


class ExtendedReturnnConfig(ReturnnConfig):
    """
    Placeholder for old setups from Mohammad
    """
    pass