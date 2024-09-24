from __future__ import annotations

from typing import TYPE_CHECKING, List
import math

import torch
import returnn.frontend as rf
from returnn.tensor import Dim, Tensor, batch_dim

from i6_experiments.users.phan.rf_models.bilstm import BiLSTM
from i6_experiments.users.phan.rf_models.trafo_lm_luca import Trafo_LM_Model

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
    from i6_experiments.users.zeyer.model_with_checkpoints import (
        ModelWithCheckpoints,
        ModelWithCheckpoint,
    )

_log_mel_feature_dim = 80

class BiLSTMEncoder(rf.Module):
    """
    Encoder using biLSTM
    Referrence: /u/atanas.gruev/setups/librispeech/2023-08-08-zhou-conformer-transducer/alias/0_bpe-5k-ss4_blstm-ctc/train_bpe-5k_gt50_ss4_blstm_OCLR_30ep/output/crnn.config
    Has:
        - <n_blstm_layers> BiLSTM layers
        - log softmax output from last blstm
        - max pooling after certain blstm (Zhou's config)
    """
    def __init__(
        self,
        in_dim: Dim,
        out_dim: Dim,
        hidden_dim: int,
        n_blstm_layers: int,
        max_pool_after: List[int],
        target_dim: Dim,
        bos_idx: int,
        eos_idx: int,
        blank_idx: int,
        external_language_model: Optional[dict] = None,
    ):
        """
        Pooling default to max pool with same padding, pool_size=(2,)

        :param in_dim:
        :param out_dim: Should be vocab dimension + blank here
        :param n_blstm_layers:
        :param max_pool_after: After which blstm layers there is pooling?
            (Zhou'config: 2, 3) [1 is the first]
            Not possible after last blstm
        :param external_language_model: dict config for constructing the external LM in search
        :param target_dim: dim used for the LM
        """
        super(BiLSTMEncoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.max_pool_after = max_pool_after
        self.blstm_hidden_dim = Dim(name="blstm_hidden_dim", dimension=hidden_dim)
        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)
        self.use_specaugment = config.typed_value("use_specaugment", False)
        
        self._specaugment_opts = {
            "steps": config.typed_value("specaugment_steps") or (0, 1000, 2000),
            "max_consecutive_spatial_dims": config.typed_value(
                "specaugment_max_consecutive_spatial_dims"
            )
            or 20,
            "max_consecutive_feature_dims": config.typed_value(
                "specaugment_max_consecutive_feature_dims"
            )
            or (_log_mel_feature_dim // 5),
            "num_spatial_mask_factor": config.typed_value(
                "specaugment_num_spatial_mask_factor"
            )
            or 100,
        }
        
        blstm_layers = []
        if n_blstm_layers == 1:
            blstm_layers.append(
                BiLSTM(self.in_dim, self.out_dim)
            )
        else:
            blstm_in_dim = self.in_dim
            for i in range(n_blstm_layers):
                if i > 0:
                    blstm_in_dim = 2*self.blstm_hidden_dim

                # if i-1 in max_pool_after:
                #     blstm_in_dim = self.after_pool_dim

                # if i == n_blstm_layers-1:
                #     blstm_out_dim = self.out_dim
                
                blstm_layers.append(
                    BiLSTM(blstm_in_dim, self.blstm_hidden_dim)
                )
        self.blstm_layers = rf.ModuleList(*blstm_layers)
        self.final_linear = rf.Linear(2*self.blstm_hidden_dim, self.out_dim)

        # these are important for search
        self.target_dim = target_dim
        self.target_dim_w_blank = self.out_dim
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx

        if external_language_model:
            lm_cls = external_language_model.pop("class", None)
            if lm_cls == "Trafo_LM_Model":
                self.language_model = Trafo_LM_Model(
                    target_dim,
                    target_dim,
                    **external_language_model,
                )
            else:
                raise NotImplementedError("Only the Kazuki Trafo LM is supported!!!!!")

    def __call__(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
    ):
        """
        Output logits
        :param source: Tensor (B, T, F)
        """
        # log mel filterbank features
        source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
            source,
            in_spatial_dim=in_spatial_dim,
            out_dim=self.in_dim,
            sampling_rate=16_000,
            log_base=math.exp(2.3026),  # almost 10.0 but not exactly...
        )
        if self.use_specaugment:
            # SpecAugment
            source = rf.audio.specaugment(
                source,
                spatial_dim=in_spatial_dim,
                feature_dim=self.in_dim,
                **self._specaugment_opts,
            )
        out = source
        spatial_dim = in_spatial_dim
        # print(f"Log mel features: {source}")
        for i, blstm in enumerate(self.blstm_layers):
            blstm: BiLSTM
            states = blstm.default_initial_state(batch_dims=[batch_dim])
            out, new_states = blstm(out, states=states, spatial_dim=spatial_dim)
            # print(f"blstm {i}: {out}")
            if i in self.max_pool_after and i != len(self.blstm_layers)-1:
                out, out_spatial_dims = rf.max_pool( # hopefully behave the same as Zhou's config
                    out, 
                    pool_size=(2,),
                    padding="same",
                    in_spatial_dims=[spatial_dim],
                )
                spatial_dim = out_spatial_dims[0]
                # print(f"max_pool {i}: {out}")
        out = self.final_linear(out)
        # print(f"final_linear: {out}")
        return out, spatial_dim

    def encode(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim
    ):
        return self.__call__(source, in_spatial_dim=in_spatial_dim)

# where the model is defined
def from_scratch_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim, **kwargs) -> BiLSTMEncoder:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
    external_language_model = config.typed_value("external_language_model")
    return BiLSTMEncoder(
        in_dim,
        target_dim+1, # for the blank
        hidden_dim=512,
        n_blstm_layers=6,
        max_pool_after=[1, 2], # downsampling factor 4 here
        target_dim=target_dim,
        bos_idx=0,
        eos_idx=0,
        blank_idx=10025,
        external_language_model=external_language_model,
    )


from_scratch_model_def: ModelDef[Model]
from_scratch_model_def.behavior_version = 21 #16
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = 160



# train step
def from_scratch_training(
    *,
    model: BiLSTMEncoder,
    data: rf.Tensor,
    data_spatial_dim: Dim,
    targets: rf.Tensor,
    targets_spatial_dim: Dim,
):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    # import for training only, will fail on CPU servers
    # from i6_native_ops import warp_rnnt

    config = get_global_config()  # noqa
    logits, downsampled_in_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    ctc_loss = rf.ctc_loss(
        logits=logits,
        targets=targets,
        input_spatial_dim=downsampled_in_spatial_dim,
        targets_spatial_dim=targets_spatial_dim,
        blank_index=10025 # fuck it
    )
    rf.get_run_ctx().mark_as_loss(
        ctc_loss,
        "ctc",
        custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
    )


from_scratch_training: TrainDef[Model]
from_scratch_training.learning_rate_control_error_measure = "dev_score_full_sum"
