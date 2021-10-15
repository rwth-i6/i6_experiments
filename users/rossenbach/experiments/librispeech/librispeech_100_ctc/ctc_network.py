def make_network_config(fdim, num_classes, dropout=0.1, **kwargs):
    # -- encoder (default: 6 * 512 BLSTM) -- #
    encoder_layers = kwargs.get('encoder_layers', 6)
    encoder_size   = kwargs.get('encoder_size', 512)
    if kwargs.get('subsampling', True):
        maxpool_pos = kwargs.get('maxpool_pos', 'middle')
        if maxpool_pos == 'bottom': max_pool = [2]
        elif maxpool_pos == 'middle': max_pool = [1]*int(encoder_layers/2-1) + [2]
        elif maxpool_pos == 'top': max_pool = [1]*int(encoder_layers-1) + [2]
        elif isinstance(maxpool_pos, list):
            assert len(maxpool_pos) <= encoder_layers, 'invalid maxpool_pos'
            max_pool = maxpool_pos
        else: assert False, 'unknown maxpool_pos %s' %maxpool_pos
    else: max_pool = []
    network, fromList = nn_setup.build_encoder_network(num_layers=encoder_layers, size=encoder_size, max_pool=max_pool, dropout=dropout)


class BLSTMCTCModel(Module):

    def __init__(self, num_layers, size, max_pool, dropout):
        super().__init__()

        self.blstm_layers = []
        for i in range(num_layers):
