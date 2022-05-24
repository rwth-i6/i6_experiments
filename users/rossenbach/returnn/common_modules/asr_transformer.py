from typing import *

import time

from returnn_common.nn.transformer import Transformer

from returnn_common import nn


class BLSTMPoolModule(nn.Module):

    def __init__(self, hidden_size, pool=None, dropout=None):
        super().__init__()
        self.lstm_out_dim = nn.FeatureDim("lstm_out_dim", dimension=hidden_size)
        self.out_feature_dim = self.lstm_out_dim + self.lstm_out_dim
        self.fw_rec = nn.LSTM(out_dim=self.lstm_out_dim)
        self.bw_rec = nn.LSTM(out_dim=self.lstm_out_dim)
        self.pool = pool
        self.dropout = dropout

    @nn.scoped
    def __call__(self, inp, time_axis):
        fw_out, _ = self.fw_rec(inp, direction=1, axis=time_axis)
        bw_out, _ = self.bw_rec(inp, direction=-1, axis=time_axis)
        c = nn.concat((fw_out, self.lstm_out_dim), (bw_out, self.lstm_out_dim))
        if self.pool is not None and self.pool > 1:
            pool, pool_spatial_dim = nn.pool1d(c, mode="max", pool_size=self.pool, padding="same", in_spatial_dim=time_axis)
            inp = nn.dropout(pool, self.dropout, axis=self.out_feature_dim)
            out_time_dim = pool_spatial_dim
        else:
            inp = nn.dropout(c, self.dropout, axis=self.out_feature_dim)
            out_time_dim = time_axis
        return inp, out_time_dim


class BLSTMDownsamplingTransformerASR(nn.Module):
    """
    Standard Transformer Module
    """
    def __init__(self,
                 audio_feature_dim: nn.Dim,
                 target_vocab: nn.Dim,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.downsampling_1 = BLSTMPoolModule(hidden_size=256, pool=2, dropout=0.1)
        self.downsampling_2 = BLSTMPoolModule(hidden_size=256, pool=2, dropout=0.1)
        self.transformer = Transformer(model_dim=self.downsampling_2.out_feature_dim, target_dim=target_vocab, **kwargs)

        #self.input_linear = nn.Linear(out_dim=self.transformer.model_dim, in_dim=audio_feature_dim)

    @nn.scoped
    def __call__(self,
                 *,
                 audio_features: nn.Tensor,
                 labels: nn.Tensor,
                 audio_time_dim: nn.Dim,
                 label_time_dim: nn.Dim,
                 label_dim: nn.Dim,
                 ):
        pool1_out, pool1_time_axis = self.downsampling_1(audio_features, time_axis=audio_time_dim)
        pool2_out, pool2_time_axis = self.downsampling_2(pool1_out, time_axis=pool1_time_axis)

        encoder_out, out_logits, out_labels, _ = self.transformer(
            pool2_out,
            source_spatial_axis=pool2_time_axis,
            target=labels,
            target_spatial_axis=label_time_dim
        )

        loss = nn.sparse_softmax_cross_entropy_with_logits(
            logits=out_logits,
            targets=labels,
            axis=label_dim,
        )
        loss.mark_as_loss()

        return out_logits