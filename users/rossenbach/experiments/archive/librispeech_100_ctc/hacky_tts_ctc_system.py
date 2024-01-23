import copy

from .ctc_system import CtcSystem


from i6_core.returnn.config import ReturnnConfig


class HackyTTSCTCSystem(CtcSystem):

    @classmethod
    def get_specific_returnn_config(cls, returnn_config, epoch=None, log_activation=False):
        """
        converts a config with a staged network into a config for a specific epoch

        :param ReturnnConfig returnn_config:
        :param epoch: epoch, if None use last one
        :return:
        """
        if not returnn_config.staged_network_dict:
            # THIS IS WRONG! the log_activation fix is missing!
            return returnn_config
        training_returnn_config = returnn_config
        config_dict = copy.deepcopy(returnn_config.config)
        # TODO: only last network for now, fix with epoch
        if epoch:
            index = 0
            raise NotImplementedError
        else:
            index = max(training_returnn_config.staged_network_dict.keys())
        config_dict['network'] = copy.deepcopy(training_returnn_config.staged_network_dict[index])
        if log_activation:
            config_dict['network']['output'] = {'class': 'activation', 'from': 'output_0', 'activation': 'log'}

        for tts_layer in [
            "speaker_embedding_raw",
            'fw0',
            'bw0',
            'fw1',
            'bw1',
            'audio_target',
        ]:
            config_dict['network'].pop(tts_layer)
        returnn_config = ReturnnConfig(config=config_dict,
                                       post_config=training_returnn_config.post_config,
                                       staged_network_dict=None,
                                       python_prolog=training_returnn_config.python_prolog,
                                       python_epilog=training_returnn_config.python_epilog,
                                       python_epilog_hash=training_returnn_config.python_epilog_hash,
                                       python_prolog_hash=training_returnn_config.python_prolog_hash)
        return returnn_config