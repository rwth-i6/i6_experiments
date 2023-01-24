__all__ = ["FactoredHybridFeatureScorer"]


import i6_core.rasr as rasr

from ..factored import PhoneticContext
from .config import default_posterior_scales, PriorInfo


class FactoredHybridFeatureScorer(rasr.FeatureScorer):
    def __init__(
        self,
        fs_tf_config,
        context_type: PhoneticContext,
        prior_mixtures,
        prior_info: PriorInfo,
        num_states_per_phone: int,
        num_label_contexts: int,
        silence_id: int,
        num_encoder_output: int,
        posterior_scales=None,
        loop_scale=1.0,
        forward_scale=1.0,
        silence_loop_penalty=0.0,
        silence_forward_penalty=0.0,
        use_estimated_tdps=False,
        state_dependent_tdp_file=None,
        is_min_duration=False,
        use_word_end_classes=False,
        use_boundary_classes=False,
        is_multi_encoder_output=False,
    ):
        super().__init__()

        """
        Both prior_scales and posterior_scales are a dictionary with three keys for each output
        """
        if posterior_scales is None:
            posterior_scales = default_posterior_scales()

        self.config = rasr.RasrConfig()
        self.config.feature_scorer_type = "tf-factored-hybrid-scorer"
        self.config.context_type = context_type
        self.config.file = prior_mixtures
        self.config.num_states_per_phone = num_states_per_phone
        self.config.num_label_contexts = num_label_contexts
        self.config.silence_id = silence_id
        self.config.num_encoder_output = num_encoder_output
        self.config.left_context_scale = posterior_scales["left-context-scale"]
        self.config.center_state_scale = posterior_scales["center-state-scale"]
        self.config.right_context_scale = posterior_scales["right-context-scale"]
        if prior_info.left_context_prior is not None:
            self.config.left_context_prior_scale = prior_info.left_context_prior.scale
            self.config.left_context_prior_file = prior_info.left_context_prior.file
        if prior_info.center_state_prior is not None:
            self.config.center_state_prior_scale = prior_info.center_state_prior.scale
            self.config.center_state_prior_file = prior_info.center_state_prior.file
        if prior_info.right_context_prior is not None:
            self.config.right_context_prior_scale = prior_info.right_context_prior.scale
            self.config.right_context_prior_file = prior_info.right_context_prior.file
        self.config.loop_scale = loop_scale
        self.config.forward_scale = forward_scale
        self.config.silence_loop_penalty = silence_loop_penalty
        self.config.silence_forward_penalty = silence_forward_penalty
        self.config.use_estimated_tdps = use_estimated_tdps
        if use_estimated_tdps:
            assert state_dependent_tdp_file is not None
        self.config.state_dependent_tdp_file = state_dependent_tdp_file
        self.config.is_min_duration = is_min_duration
        self.config.use_word_end_classes = use_word_end_classes
        self.config.use_boundary_classes = use_boundary_classes
        self.config.is_multi_encoder_output = is_multi_encoder_output
        self.config.loader = fs_tf_config.loader
        self.config.input_map = fs_tf_config.input_map
        self.config.output_map = fs_tf_config.output_map


class FactoredHybridFeatureScorerV2(rasr.FeatureScorer):
    def __init__(
        self,
        fs_tf_config,
        context_type,
        transition_type,
        prior_mixtures,
        prior_info,
        num_states_per_phone,
        num_label_contexts,
        silence_id,
        num_encoder_output=1024,
        posterior_scales=None,
        loop_scale=1.0,
        forward_scale=1.0,
        silence_loop_penalty=0.0,
        silence_forward_penalty=0.0,
        use_estimated_tdps=False,
        state_dependent_tdp_file=None,
        is_min_duration=False,
        use_word_end_classes=False,
        use_boundary_classes=False,
        is_multi_encoder_output=False,
    ):
        super().__init__()

        """
        Both prior_scales and posterior_scales are a dictionary with three keys for each output
        """
        if posterior_scales is None:
            posterior_scales = dict(
                zip(
                    [
                        f"{k}-scale"
                        for k in [
                            "left-context",
                            "center-state",
                            "right-context",
                        ]
                    ],
                    [1.0] * 3,
                )
            )

        self.config = rasr.RasrConfig()
        self.config.feature_scorer_type = "tf-factored-hybrid-scorer"
        self.config.context_type = context_type
        self.config.transition_type = transition_type
        self.config.file = prior_mixtures
        self.config.num_states_per_phone = num_states_per_phone
        self.config.num_label_contexts = num_label_contexts
        self.config.silence_id = silence_id
        self.config.num_encoder_output = num_encoder_output
        self.config.left_context_scale = posterior_scales["left-context-scale"]
        self.config.center_state_scale = posterior_scales["center-state-scale"]
        self.config.right_context_scale = posterior_scales["right-context-scale"]
        if prior_info["left-context-prior"]["file"] is not None:
            self.config.left_context_prior_scale = prior_info["left-context-prior"][
                "scale"
            ]
            self.config.left_context_prior_file = prior_info["left-context-prior"][
                "file"
            ]
        if prior_info["center-state-prior"]["file"] is not None:
            self.config.center_state_prior_scale = prior_info["center-state-prior"][
                "scale"
            ]
            self.config.center_state_prior_file = prior_info["center-state-prior"][
                "file"
            ]
        if prior_info["right-context-prior"]["file"] is not None:
            self.config.right_context_prior_scale = prior_info["right-context-prior"][
                "scale"
            ]
            self.config.right_context_prior_file = prior_info["right-context-prior"][
                "file"
            ]
        self.config.loop_scale = loop_scale
        self.config.forward_scale = forward_scale
        self.config.silence_loop_penalty = silence_loop_penalty
        self.config.silence_forward_penalty = silence_forward_penalty
        self.config.use_estimated_tdps = use_estimated_tdps
        if use_estimated_tdps:
            assert state_dependent_tdp_file is not None
        self.config.state_dependent_tdp_file = state_dependent_tdp_file
        self.config.is_min_duration = is_min_duration
        self.config.use_word_end_classes = use_word_end_classes
        self.config.use_boundary_classes = use_boundary_classes
        self.config.is_multi_encoder_output = is_multi_encoder_output
        self.config.loader = fs_tf_config.loader
        self.config.input_map = fs_tf_config.input_map
        self.config.output_map = fs_tf_config.output_map
