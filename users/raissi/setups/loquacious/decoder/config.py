import dataclasses
from dataclasses import dataclass
import typing

from sisyphus import tk

from i6_experiments.users.raissi.setups.common.util.tdp import (
    TDP,
)

from i6_experiments.users.raissi.setups.common.data.factored_label import (
    PhoneticContext,
)

from i6_experiments.users.raissi.setups.common.decoder.config import (
    SearchParameters,
    PriorInfo,
)
from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import DecodingTensorMap


BLSTM_FH_DECODING_TENSOR_CONFIG_TF2 = dataclasses.replace(
    DecodingTensorMap.default(),
    in_encoder_output="concat_lstm_fwd_6_lstm_bwd_6/concat_sources/concat",
    in_seq_length="extern_data/placeholders/data/data_dim0_size",
    out_encoder_output="encoder__output/output_batch_major",
    out_right_context="right__output/output_batch_major",
    out_left_context="left__output/output_batch_major",
    out_center_state="center__output/output_batch_major",
    out_joint_diphone="output/output_batch_major",

)


@dataclass(eq=True, frozen=True)
class LQCSearchParameters(SearchParameters):
    @classmethod
    def default_monophone(cls, *, priors: PriorInfo, frame_rate: int = 1) -> "LBSSearchParameters":
        return cls(
            beam=22,
            beam_limit=500_000,
            lm_scale=1.0 if frame_rate > 1 else 4.0,
            tdp_scale=0.1 if frame_rate > 1 else 0.4,
            prior_info=priors.with_scale(0.2),
            tdp_speech=(8.0, 0.0, "infinity", 0.0) if frame_rate > 1 else (3.0, 0.0, "infinity", 0.0),
            tdp_silence=(11.0, 0.0, "infinity", 20.0) if frame_rate > 1 else (0.0, 3.0, "infinity", 20.0),
            tdp_nonword=(11.0, 0.0, "infinity", 20.0) if frame_rate > 1 else (0.0, 3.0, "infinity", 20.0),
            non_word_phonemes="[UNKNOWN]",
            we_pruning=0.5,
            we_pruning_limit=10000,
        )

    @classmethod
    def default_diphone(cls, *, priors: PriorInfo, frame_rate: int = 1) -> "LBSSearchParameters":
        return cls(
            beam=20,
            beam_limit=500_000,
            lm_scale=1.8 if frame_rate > 1 else 6.0,
            tdp_scale=0.1 if frame_rate > 1 else 0.4,
            prior_info=priors.with_scale(center=0.2, left=0.1),
            tdp_speech=(8.0, 0.0, "infinity", 0.0) if frame_rate > 1 else (3.0, 0.0, "infinity", 0.0),
            tdp_silence=(11.0, 0.0, "infinity", 20.0) if frame_rate > 1 else (0.0, 3.0, "infinity", 20.0),
            tdp_nonword=(11.0, 0.0, "infinity", 20.0) if frame_rate > 1 else (0.0, 3.0, "infinity", 20.0),
            non_word_phonemes="[UNKNOWN]",
            we_pruning=0.5,
            we_pruning_limit=10000,
        )

    @classmethod
    def default_triphone(cls, *, priors: PriorInfo, frame_rate: int = 1) -> "LBSSearchParameters":
        return cls(
            beam=20,
            beam_limit=500_000,
            lm_scale=11.0,
            tdp_scale=0.6,
            prior_info=priors.with_scale(center=0.2, left=0.1, right=0.1),
            tdp_speech=(8.0, 0.0, "infinity", 0.0) if frame_rate > 1 else (3.0, 0.0, "infinity", 0.0),
            tdp_silence=(11.0, 0.0, "infinity", 20.0) if frame_rate > 1 else (0.0, 3.0, "infinity", 20.0),
            tdp_nonword=(11.0, 0.0, "infinity", 20.0) if frame_rate > 1 else (0.0, 3.0, "infinity", 20.0),
            non_word_phonemes="[UNKNOWN]",
            we_pruning=0.5,
            we_pruning_limit=10000,
        )

    @classmethod
    def default_joint_diphone(cls, *, priors: PriorInfo, frame_rate: int = 1) -> "SearchParameters":
        return cls(
            beam=20,
            beam_limit=500_000,
            lm_scale=1.8 if frame_rate > 1 else 6.0,
            tdp_scale=0.1 if frame_rate > 1 else 0.4,
            prior_info=priors.with_scale(diphone=0.4),
            tdp_speech=(8.0, 0.0, "infinity", 0.0) if frame_rate > 1 else (3.0, 0.0, "infinity", 0.0),
            tdp_silence=(11.0, 0.0, "infinity", 20.0) if frame_rate > 1 else (0.0, 3.0, "infinity", 20.0),
            tdp_nonword=(11.0, 0.0, "infinity", 20.0) if frame_rate > 1 else (0.0, 3.0, "infinity", 20.0),
            non_word_phonemes="[UNKNOWN]",
            we_pruning=0.5,
            we_pruning_limit=10000,
        )
