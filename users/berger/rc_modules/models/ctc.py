from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
from returnn_common.asr.specaugment import specaugment_v2
from returnn_common.nn.encoder.base import ISeqDownsamplingEncoder, ISeqFramewiseEncoder
from ..feature_extraction import FeatureType, make_features
from ..encoder import EncoderType, make_encoder
from ..data_augmentation import legacy_specaugment
from returnn_common import nn
import i6_core.rasr as rasr
from i6_core.am.config import acoustic_model_config


def make_ctc_rasr_loss_config(
    loss_corpus_path: str,
    loss_lexicon_path: str,
    am_args: Dict,
    allow_label_loop: bool = True,
    min_duration: int = 1,
    extra_config: Optional[rasr.RasrConfig] = None,
    extra_post_config: Optional[rasr.RasrConfig] = None,
):

    # Make crp and set loss_corpus and lexicon
    loss_crp = rasr.CommonRasrParameters()
    rasr.crp_add_default_output(loss_crp)

    loss_crp.corpus_config = rasr.RasrConfig()  # type: ignore
    loss_crp.corpus_config.file = loss_corpus_path  # type: ignore
    loss_crp.corpus_config.remove_corpus_name_prefix = "loss-corpus/"  # type: ignore

    loss_crp.lexicon_config = rasr.RasrConfig()  # type: ignore
    loss_crp.lexicon_config.file = loss_lexicon_path  # type: ignore

    loss_crp.acoustic_model_config = acoustic_model_config(**am_args)  # type: ignore
    loss_crp.acoustic_model_config.allophones.add_all = True  # type: ignore
    loss_crp.acoustic_model_config.allophones.add_from_lexicon = False  # type: ignore

    # Make config from crp
    mapping = {
        "acoustic_model": "*.model-combination.acoustic-model",
        "corpus": "*.corpus",
        "lexicon": "*.model-combination.lexicon",
    }
    config, post_config = rasr.build_config_from_mapping(
        loss_crp,
        mapping,
        parallelize=False,
    )
    config.action = "python-control"
    config.python_control_loop_type = "python-control-loop"
    config.extract_features = False

    # Allophone state transducer
    config["*"].transducer_builder_filter_out_invalid_allophones = True  # type: ignore
    config["*"].fix_allophone_context_at_word_boundaries = True  # type: ignore

    # Automaton manipulation
    if allow_label_loop:
        topology = "ctc"
    else:
        topology = "rna"
    config["*"].allophone_state_graph_builder.topology = topology  # type: ignore

    if min_duration > 1:
        config["*"].allophone_state_graph_builder.label_min_duration = min_duration  # type: ignore

    # maybe not needed
    config["*"].allow_for_silence_repetitions = False  # type: ignore

    config._update(extra_config)
    post_config._update(extra_post_config)

    return config, post_config


def make_rasr_ctc_loss_opts(
    rasr_binary_path: str, rasr_arch: str = "linux-x86_64-standard", num_instances: int = 2, **kwargs
):
    trainer_exe = Path(rasr_binary_path) / f"nn-trainer.{rasr_arch}"

    config, post_config = make_ctc_rasr_loss_config(**kwargs)

    loss_opts = {
        "sprint_opts": {
            "sprintExecPath": trainer_exe.as_posix(),
            "sprintConfigStr": f"{config} {post_config} --*.LOGFILE=nn-trainer.loss.log --*.TASK=1",
            "minPythonControlVersion": 4,
            "numInstances": num_instances,
            "usePythonSegmentOrder": False,
        },
        "tdp_scale": 0.0,
    }
    return loss_opts


class CTCModel(nn.Module):
    def __init__(
        self,
        num_outputs: int,
        feature_type: FeatureType = FeatureType.Gammatone,
        encoder_type: EncoderType = EncoderType.Blstm,
        specaug_args: dict = {},
        feature_args: dict = {},
        encoder_args: dict = {},
        loss_args: dict = {},
        legacy_specaug: bool = False,
    ) -> None:
        self.features, self.feature_dim = make_features(feature_type, **feature_args)

        self.specaug_args = specaug_args

        self.encoder = make_encoder(encoder_type, in_dim=self.feature_dim, **encoder_args)
        self.out_dim = nn.FeatureDim("ctc_out", dimension=num_outputs)
        self.out_projection = nn.Linear(self.encoder.out_dim, self.out_dim)

        self.loss_args = loss_args

        self.legacy_specaug = legacy_specaug

    def __call__(
        self,
        source: nn.Tensor,
        *,
        train: bool = False,
    ) -> nn.Tensor:
        assert source.data is not None
        assert source.data.time_dim_axis is not None
        in_spatial_dim = source.data.dim_tags[source.data.time_dim_axis]

        x = source

        if source.data.feature_dim_or_sparse_dim is not None:
            x = nn.squeeze(x, axis=source.data.feature_dim_or_sparse_dim)

        x = self.features(x, in_spatial_dim=in_spatial_dim)
        assert isinstance(x, tuple)
        x, spatial_dim = x
        assert isinstance(spatial_dim, nn.Dim)

        if self.legacy_specaug:
            x = legacy_specaugment(x, spatial_dim=spatial_dim, feature_dim=self.feature_dim, **self.specaug_args)
        else:
            x = specaugment_v2(x, spatial_dim=spatial_dim, feature_dim=self.feature_dim, **self.specaug_args)

        if isinstance(self.encoder, ISeqFramewiseEncoder):
            x = self.encoder(x, spatial_dim=spatial_dim)
        elif isinstance(self.encoder, ISeqDownsamplingEncoder):
            x, spatial_dim = self.encoder(x, in_spatial_dim=spatial_dim)
        else:
            raise TypeError(f"unsupported encoder type {type(self.encoder)}")

        x = self.out_projection(x)

        if train:
            x = nn.make_layer(
                name="ctc_loss",
                layer_dict={
                    "class": "activation",
                    "activation": "softmax",
                    "from": x,
                    "loss": "fast_bw",
                    "loss_opts": make_rasr_ctc_loss_opts(**self.loss_args),
                },
            )
        else:
            x = nn.log_softmax(x, axis=self.out_dim)

        return x


def construct_net_with_data(epoch: int, train: bool, audio_data: nn.Data, **kwargs) -> CTCModel:
    net = CTCModel(**kwargs)

    output = net(source=nn.get_extern_data(audio_data), train=train)
    output.mark_as_default_output()

    return net
