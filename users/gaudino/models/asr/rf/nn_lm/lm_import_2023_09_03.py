"""Import LM from TF checkpoint to RETURNN frontend model with PT backend."""

from typing import Optional, Any, Tuple, Dict, Sequence, List
import tree
from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
import copy as _copy


class LSTM_LM_Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: Dim,
        target_dim: Dim,
        *,
        num_layers: int = 4,
        lstm_input_dim: int = 128 ,
        lstm_model_dim:  int = 2048,
        # enc_att_num_heads: int = 4,
        # enc_conformer_layer_opts: Optional[Dict[str, Any]] = None,
        # enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
        # att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
        # att_dropout: float = 0.1,
        # enc_dropout: float = 0.1,
        # enc_att_dropout: float = 0.1,
        # l2: float = 0.0001,
        search_args: Optional[Dict[str, Any]] = None,
    ):
        super(LSTM_LM_Model, self).__init__()
        self.in_dim = in_dim
        self.target_dim = target_dim
        lstm_input_dim = Dim(name="lstm-input", dimension=lstm_input_dim)
        lstm_model_dim = Dim(name="lstm-model", dimension=lstm_model_dim)

        self.input = rf.Embedding(in_dim, lstm_input_dim)
        self.input_bias = rf.Parameter((lstm_input_dim,))

        first_lstm_layer = rf.LSTM(lstm_input_dim, lstm_model_dim, with_bias=True)
        lstm_layer = rf.LSTM(lstm_model_dim, lstm_model_dim, with_bias=True)

        self.layers = rf.Sequential(first_lstm_layer, *[_copy.deepcopy(lstm_layer) for _ in range(num_layers - 1)])

        # self.lstm_0 = rf.LSTM(lstm_input_dim, lstm_model_dim, with_bias=True)
        # self.lstm_1 = rf.LSTM(lstm_model_dim, lstm_model_dim, with_bias=True)
        # self.lstm_2 = rf.LSTM(lstm_model_dim, lstm_model_dim, with_bias=True)
        # self.lstm_3 = rf.LSTM(lstm_model_dim, lstm_model_dim, with_bias=True)

        self.output = rf.Linear(lstm_model_dim, target_dim)

    def __call__(self, prev_target, state, spatial_dim: Dim, collected_outputs: Optional[Dict[str, Tensor]] = None):
        """loop step"""
        lm_state = rf.State()
        input = self.input(prev_target)
        input += self.input_bias
        # breakpoint()
        decoded = input

        for layer_name, layer in self.layers.items():
            layer: rf.LSTM  # or similar
            decoded, lm_state[layer_name] = layer(
                decoded, spatial_dim=spatial_dim, state=state[layer_name]
            )
            # if layer_name in ["0"]: # "0"
            #     breakpoint()
            if collected_outputs is not None:
                collected_outputs[layer_name] = decoded
        # lstm_0, lstm_0_state = self.lstm_0(input, state=state.lstm_0, spatial_dim=single_step_dim)
        # lm_state.lstm_0 = lstm_0_state
        # lstm_1, lstm_1_state = self.lstm_1(lstm_0, state=state.lstm_1, spatial_dim=single_step_dim)
        # lm_state.lstm_1 = lstm_1_state
        # lstm_2, lstm_2_state = self.lstm_2(lstm_1, state=state.lstm_2, spatial_dim=single_step_dim)
        # lm_state.lstm_2 = lstm_2_state
        # lstm_3, lstm_3_state = self.lstm_3(lstm_2, state=state.lstm_3, spatial_dim=single_step_dim)
        # lm_state.lstm_3 = lstm_3_state
        output = self.output(decoded)
        return {"output": output, "state": lm_state}

    def default_initial_state(self, *, batch_dims: Sequence[Dim]
    ) -> rf.State:
        """Default initial state"""
        state = rf.State({k: v.default_initial_state(batch_dims=batch_dims) for k, v in self.layers.items()})
        # state = rf.State(
        #     lstm_0=self.lstm_0.default_initial_state(batch_dims=batch_dims),
        #     lstm_1=self.lstm_1.default_initial_state(batch_dims=batch_dims),
        #     lstm_2=self.lstm_2.default_initial_state(batch_dims=batch_dims),
        #     lstm_3=self.lstm_3.default_initial_state(batch_dims=batch_dims),
        # )
        return state

    def select_state(self, state: rf.State, backrefs) -> rf.State:
        state = tree.map_structure(
            lambda s: rf.gather(s, indices=backrefs), state
        )
        return state

class MakeModel:
    """for import"""

    def __init__(
        self,
        in_dim: int,
        target_dim: int,
        *,
        eos_label: int = 0,
        # num_enc_layers: int = 12,
        model_args: Optional[Dict[str, Any]] = None,
    ):
        self.in_dim = in_dim
        self.target_dim = target_dim

        self.eos_label = eos_label
        self.model_args =model_args


    def __call__(self) -> LSTM_LM_Model:
        from returnn.datasets.util.vocabulary import Vocabulary

        in_dim = Dim(name="in", dimension=self.in_dim, kind=Dim.Types.Feature)
        target_dim = Dim(
            name="target", dimension=self.target_dim, kind=Dim.Types.Feature
        )
        target_dim.vocab = Vocabulary.create_vocab_from_labels(
            [str(i) for i in range(target_dim.dimension)], eos_label=self.eos_label
        )

        return self.make_model(in_dim, target_dim, model_args=self.model_args)

    @classmethod
    def make_model(
        cls,
        in_dim: Dim,
        target_dim: Dim,
        *,
        model_args: Optional[Dict[str, Any]] = None,
        # search_args: Optional[Dict[str, Any]],
        # num_enc_layers: int = 12,
    ) -> LSTM_LM_Model:
        """make"""
        return LSTM_LM_Model(
            in_dim,
            # num_enc_layers=num_enc_layers,
            target_dim=target_dim,
            **(model_args if model_args else {}),
        )