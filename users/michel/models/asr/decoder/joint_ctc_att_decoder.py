"""One pass joint decoder for CTC and attention based models."""


class CTCAttJointDecoder:
    def __init__(self, encoder_net, att_scale, ctc_scale):
        self.encoder_net = encoder_net
        self.att_scale = att_scale
        self.ctc_scale = ctc_scale

    def create_att_subnetwork(self):
        pass
