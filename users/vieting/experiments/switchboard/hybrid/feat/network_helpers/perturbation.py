import functools
import torchaudio
import torch
import numpy as np
import random
from typing import List, Dict, Optional, Any

class PerturbationFactor:
    """
    Class to wrap perturbation factors, e.g. for speed or tempo perturbation.
    """

    def __init__(self, prob, minimum, maximum):
        self.prob = prob
        self.min = minimum
        self.max = maximum


class WaveformPerturbation:
    """
    This class enables the perturbation of audio waveforms by applying a variety of transformations such as speed and tempo modification,
    SoX effects, codec application, and pre-emphasis filtering. 

    :param speed: A dictionary that specifies the parameters for speed perturbation. 
        - 'prob' (float): The probability of applying speed perturbation. 
        - 'minimum' (float): The minimum factor by which the audio speed will be decreased.
        - 'maximum' (float): The maximum factor by which the audio speed will be increased.
        Example: {"prob": 0.6, "minimum": 0.88, "maximum": 1.12}

    :param tempo: A dictionary specifying the parameters for tempo perturbation.
        - 'prob' (float): The probability of applying tempo perturbation. 
        - 'minimum' (float): The minimum factor by which the audio tempo will be decreased.
        - 'maximum' (float): The maximum factor by which the audio tempo will be increased.
        Example: {"prob": 0.6, "minimum": 0.83, "maximum": 1.17}

    :param sox_effects: A list of dictionaries, each dictionary representing a SoX effect.

    :param codecs: A list of dictionaries where each dictionary represents a codec with its parameters.
        - 'format' (str): The audio format such as 'wav', 'vorbis' etc.
        - 'encoding' or 'compression' (str/float): The encoding or compression technique and its level to be used.
        - 'prob' (float): The probability of applying this specific codec.
        Example: [{"format": "wav", "encoding": "ULAW", "prob": 0.4}]

    :param preemphasis: A dictionary containing parameters for the preemphasis filter.
        - 'prob' (float): The probability of applying the preemphasis filter.
        - 'minimum' (float): The minimum preemphasis factor.
        - 'maximum' (float): The maximum preemphasis factor.
        Example: {"prob": 0.9, "minimum": 0.9, "maximum": 1.0}
        
    The parameters `speed`, `tempo`, `codecs`, and `preemphasis` contain a 'prob' key
    which determines the probability that the corresponding transformation is applied. 
    """

    def __init__(
        self,
        speed: Optional[Dict[str, Any]] = None,
        tempo: Optional[Dict[str, Any]] = None,
        sox_effects: Optional[List[Dict[str, Any]]] = None,
        codecs: Optional[List[Dict[str, Any]]] = None,
        preemphasis: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes an instance of a class.
        :param speed: A dictionary containing parameters for the speed perturbation.
            Expected keys:
            - factor (float): The factor by which the audio speed will be perturbed.
        :param tempo: A dictionary containing parameters for the tempo perturbation.
            Expected keys:
            - factor (float): The factor by which the audio tempo will be perturbed.
        :param sox_effects: A list of dictionaries, each containing parameters for a specific SoX effect.
        :param codecs: A list of dictionaries, each containing parameters for applying codecs to the audio.
        :param preemphasis: A dictionary containing parameters for the preemphasis filter.
        """
        self._speed = PerturbationFactor(**speed) if speed else None
        self._tempo = PerturbationFactor(**tempo) if tempo else None
        self._perturbations = [functools.partial(self.sox, sox_effects=sox_effects)]
        if preemphasis:
            self._perturbations.append(functools.partial(self.preemphasis, factor=PerturbationFactor(**preemphasis)))
        if codecs:
            self._perturbations.append(functools.partial(self.apply_codecs, codecs=codecs))

    def run(self, audio, sample_rate, random_state):
        audio = torch.from_numpy(audio).unsqueeze(0).to(torch.float32)
        for perturbation in self._perturbations:
            audio = perturbation(audio, sample_rate, random_state)
        audio = audio.numpy().squeeze()
        assert isinstance(audio, np.ndarray)
        assert len(audio.shape) == 1
        return audio

    def sox(self, audio, sample_rate, random_state, sox_effects):
        sox_effects = sox_effects or []
        speed = False
        if self._speed is not None:
            if random_state.random() < self._speed.prob:
                factor = random_state.random() * (self._speed.max - self._speed.min) + self._speed.min
                sox_effects.append(["speed", str(factor)])
                sox_effects.append(["rate", str(sample_rate)])
                speed = True
        if self._tempo is not None:
            if random_state.random() < self._tempo.prob and not speed:
                factor = random_state.random() * (self._tempo.max - self._tempo.min) + self._tempo.min
                sox_effects.append(["tempo", str(factor)])
        audio, _ = torchaudio.sox_effects.apply_effects_tensor(audio, sample_rate, sox_effects)
        return audio

    @staticmethod
    def preemphasis(audio, sample_rate, random_state, factor):
        if random_state.random() < factor.prob:
            preemphasis_coefficient = random_state.random() * (factor.max - factor.min) + factor.min
            audio = torchaudio.functional.preemphasis(audio, coeff=preemphasis_coefficient)
        return audio

    @staticmethod
    def apply_codecs(audio, sample_rate, random_state, codecs):
        for codec in codecs:
            prob = codec.pop("prob", 1.0)
            if random_state.random() < prob:
                audio = torchaudio.functional.apply_codec(audio, sample_rate, **codec)
        return audio


def get_code_for_perturbation():
    classes = ["import torch", "import numpy as np", "import torchaudio", "import functools", "import random", "from typing import List, Dict, Any, Optional"]
    for cls_name, cls in list(globals().items()):
        if isinstance(cls, type):
            classes.append(cls)
    return classes
