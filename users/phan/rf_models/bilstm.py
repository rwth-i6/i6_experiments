from typing import Tuple, Sequence

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim
from returnn.frontend.rec import LstmState


class BiLSTM(rf.Module):
    """
    Bidirectional LSTM implemented on top of rf.LSTM,
    works by having two rf.LSTM layers for scanning the sequence in 2 directions
    """

    def __init__(
        self,
        in_dim: Dim,
        out_dim: Dim,
        *,
        with_bias: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fwd_lstm = rf.LSTM(in_dim, out_dim, with_bias=with_bias)
        self.bwd_lstm = rf.LSTM(in_dim, out_dim, with_bias=with_bias)
    
    def __call__(
        self,
        source: Tensor,
        *,
        states: Tuple[LstmState, LstmState],
        spatial_dim: Dim,
    ):
        """
        This will not (and should not) work for single_step_dim

        :param source: Tensor of size {...,spatial_dim,in_dim}.
        :param state: States of the LSTM. First is forward state, second is backward state.
            Both h and c are of shape {...,out_dim}.
        :return: output of shape {...,spatial_dim,2*out_dim},
            and new state of the LSTM.
        """
        fwd_state, bwd_state = states
        fwd_out, new_fwd_state = self.fwd_lstm(source, state=fwd_state, spatial_dim=spatial_dim)
        # rf.array_.reverse_sequence will fail when raw_tensor of
        # source and axis are not on the same device.
        # Maybe fixed by using the latest returnn
        gpu = source.raw_tensor.device
        lengths_raw_tensor = spatial_dim.dyn_size_ext.raw_tensor
        cpu = lengths_raw_tensor.device
        spatial_dim.dyn_size_ext.raw_tensor = lengths_raw_tensor.to(gpu)
        source_rev = rf.array_.reverse_sequence(source, axis=spatial_dim)
        # lstm expects axis raw tensor to be on cpu...
        spatial_dim.dyn_size_ext.raw_tensor = lengths_raw_tensor.to(cpu)
        bwd_out, new_bwd_state = self.bwd_lstm(source_rev, state=bwd_state, spatial_dim=spatial_dim)
        spatial_dim.dyn_size_ext.raw_tensor = lengths_raw_tensor.to(gpu)
        bwd_out_rev = rf.array_.reverse_sequence(bwd_out, axis=spatial_dim)
        spatial_dim.dyn_size_ext.raw_tensor = lengths_raw_tensor.to(cpu)
        out, concat_out_dim = rf.concat((fwd_out, self.out_dim), (bwd_out_rev, self.out_dim))
        return out, (new_fwd_state, new_bwd_state)

    
    def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> Tuple[LstmState, LstmState]:
        """
        Initial states. First is forward state, second is backward state.
        """
        fwd_state = self.fwd_lstm.default_initial_state(batch_dims=batch_dims)
        bwd_state = self.bwd_lstm.default_initial_state(batch_dims=batch_dims)
        return fwd_state, bwd_state

