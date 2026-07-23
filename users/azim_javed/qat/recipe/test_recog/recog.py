import os
import sys
from typing import List, Optional, Tuple
import numpy.typing as npt
import torch
import torchaudio

sys.path.insert(
    0, "/work/asr4/berger/rasr_dev/label_scorer/pr/fix_python_scorer_allowed_transitions/arch/linux-x86_64-standard"
)

from ctc_model import get_ctc_model

from librasr import Configuration, SearchAlgorithm, register_label_scorer_type, LabelScorer, TransitionType


class PytorchCTCLabelScorer(LabelScorer):
    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        print(f"RASR initialized the label scorer with test-parameter={config['test-parameter']}")
        self._ctc_model = get_ctc_model()
        self._feature_buffer = []
        self._scores = torch.zeros(0, 0)

    def allowed_transition_types(self) -> List[TransitionType]:
        return [
            TransitionType.LABEL_TO_LABEL,
            TransitionType.LABEL_LOOP,
            TransitionType.LABEL_TO_BLANK,
            TransitionType.BLANK_TO_LABEL,
            TransitionType.BLANK_LOOP,
            TransitionType.INITIAL_LABEL,
            TransitionType.INITIAL_BLANK,
        ]

    def reset(self) -> None:
        self._feature_buffer.clear()
        self._scores = torch.zeros(0, 0)

    def signal_no_more_features(self) -> None:
        # All features are available -> Run model forwarding
        features = torch.concat(self._feature_buffer, dim=0)  # [T, 1]
        log_probs, _ = self._ctc_model.forward(
            audio_features=features.reshape(1, -1, 1), audio_features_len=torch.tensor([features.size(0)])
        )  # [1, T, V]
        self._scores = -log_probs.squeeze(0)  # [T, V]

    def get_initial_scoring_context(self) -> int:
        # Context is just timestep
        return 0

    def extended_scoring_context(self, context: int, next_token: int, transition_type: int) -> int:
        return context + 1

    def add_inputs(self, inputs: npt.NDArray) -> None:
        self._feature_buffer.append(torch.tensor(inputs))

    def compute_scores_with_times(self, contexts: List[int]) -> List[Optional[Tuple[List[float], int]]]:
        result = []
        for context in contexts:
            if context >= self._scores.size(0):
                result.append(None)
                continue
            result.append((self._scores[context].tolist(), context))

        return result


register_label_scorer_type("pytorch-ctc", PytorchCTCLabelScorer)

config = Configuration()
config.set_from_file("recognition.config")

search_algorithm = SearchAlgorithm(config=config)

for audio_file in [
    "/work/asr4/berger/rasr_dev/label_scorer/setup/dependencies/audio/1272-128104-0000.wav",
    "/work/asr4/berger/rasr_dev/label_scorer/setup/dependencies/audio/1272-128104-0001.wav",
    "/work/asr4/berger/rasr_dev/label_scorer/setup/dependencies/audio/1272-128104-0002.wav",
]:
    with torch.no_grad():
        waveform, _ = torchaudio.load(audio_file)  # [1, T] # type: ignore
    result = " ".join(item.lemma for item in search_algorithm.recognize_segment(features=waveform.reshape(-1, 1)))
    result = result.replace(" _ ", " ")
    result = result.replace("_ ", "")
    result = result.replace(" _", "")
    result = result.replace("@@ ", "")
    print(f"Recognition result for {os.path.basename(audio_file)}: {result}")
    print()
