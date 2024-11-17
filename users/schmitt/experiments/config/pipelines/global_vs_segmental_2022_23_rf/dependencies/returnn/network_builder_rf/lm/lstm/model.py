from typing import Optional, Any, Tuple, Dict, Sequence, List
import tree
from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf


class LSTM_LM_Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: Dim,
        target_dim: Dim,
        *,
        num_enc_layers: int = 12,

        lstm_input_dim: Dim = Dim(name="lstm-input", dimension=128),
        lstm_model_dim: Dim = Dim(name="lstm-model", dimension=2048),
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

        self.input = rf.Embedding(in_dim, lstm_input_dim)
        self.input_bias = rf.Parameter((lstm_input_dim,))

        self.lstm_0 = rf.LSTM(lstm_input_dim, lstm_model_dim, with_bias=True)
        self.lstm_1 = rf.LSTM(lstm_model_dim, lstm_model_dim, with_bias=True)
        self.lstm_2 = rf.LSTM(lstm_model_dim, lstm_model_dim, with_bias=True)
        self.lstm_3 = rf.LSTM(lstm_model_dim, lstm_model_dim, with_bias=True)

        self.output = rf.Linear(lstm_model_dim, target_dim)

    def loop_step(self, prev_target, prev_state):
        """loop step"""
        lm_state = rf.State()
        input = self.input(prev_target)
        input += self.input_bias
        # breakpoint()
        lstm_0, lstm_0_state = self.lstm_0(input, state=prev_state.lstm_0, spatial_dim=single_step_dim)
        lm_state.lstm_0 = lstm_0_state
        lstm_1, lstm_1_state = self.lstm_1(lstm_0, state=prev_state.lstm_1, spatial_dim=single_step_dim)
        lm_state.lstm_1 = lstm_1_state
        lstm_2, lstm_2_state = self.lstm_2(lstm_1, state=prev_state.lstm_2, spatial_dim=single_step_dim)
        lm_state.lstm_2 = lstm_2_state
        lstm_3, lstm_3_state = self.lstm_3(lstm_2, state=prev_state.lstm_3, spatial_dim=single_step_dim)
        lm_state.lstm_3 = lstm_3_state
        output = self.output(lstm_3)
        return {"output": output}, lm_state

    def lm_default_initial_state(self, *, batch_dims: Sequence[Dim]
    ) -> rf.State:
        """Default initial state"""
        state = rf.State(
            lstm_0=self.lstm_0.default_initial_state(batch_dims=batch_dims),
            lstm_1=self.lstm_1.default_initial_state(batch_dims=batch_dims),
            lstm_2=self.lstm_2.default_initial_state(batch_dims=batch_dims),
            lstm_3=self.lstm_3.default_initial_state(batch_dims=batch_dims),
        )
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
    ):
        self.in_dim = in_dim
        self.target_dim = target_dim

        self.eos_label = eos_label

    def __call__(self) -> LSTM_LM_Model:
        from returnn.datasets.util.vocabulary import Vocabulary

        in_dim = Dim(name="in", dimension=self.in_dim, kind=Dim.Types.Feature)
        target_dim = Dim(
            name="target", dimension=self.target_dim, kind=Dim.Types.Feature
        )
        target_dim.vocab = Vocabulary.create_vocab_from_labels(
            [str(i) for i in range(target_dim.dimension)], eos_label=self.eos_label
        )

        return self.make_model(in_dim, target_dim)

    @classmethod
    def make_model(
        cls,
        in_dim: Dim,
        target_dim: Dim,
    ) -> LSTM_LM_Model:
        """make"""
        return LSTM_LM_Model(
            in_dim,
            # num_enc_layers=num_enc_layers,
            target_dim=target_dim,
        )


def from_scratch_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> LSTM_LM_Model:
  """Function is run within RETURNN."""
  from returnn.config import get_global_config

  in_dim, epoch  # noqa
  config = get_global_config()  # noqa

  return MakeModel.make_model(
    in_dim,
    target_dim,
  )


def _returnn_v2_get_model(*, epoch: int, **_kwargs_unused):
  from returnn.tensor import Tensor
  from returnn.config import get_global_config

  config = get_global_config()
  default_input_key = config.typed_value("default_input")
  default_target_key = config.typed_value("target")
  extern_data_dict = config.typed_value("extern_data")
  data = Tensor(name=default_input_key, **extern_data_dict[default_input_key])

  if default_target_key in extern_data_dict:
    targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
  else:
    non_blank_target_dimension = config.typed_value("non_blank_target_dimension", None)
    vocab = config.typed_value("vocab", None)
    assert non_blank_target_dimension and vocab
    target_dim = Dim(description="non_blank_target_dim", dimension=non_blank_target_dimension, kind=Dim.Types.Spatial)
    targets = Tensor(name=default_target_key, sparse_dim=target_dim, vocab=vocab)

  model_def = config.typed_value("_model_def")
  model = model_def(epoch=epoch, in_dim=data.feature_dim, target_dim=targets.sparse_dim)
  return model
