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
    This class enables the perturbation of audio waveforms by applying a variety of transformations such
    as speed and tempo modification, SoX effects, codec application, and pre-emphasis filtering.
    The parameters `speed`, `tempo`, `codecs`, and `preemphasis` contain a 'prob' key
    which determines the probability that the corresponding transformation is applied.
    """

    def __init__(
        self,
        speed: Optional[Dict[str, Any]] = None,
        tempo: Optional[Dict[str, Any]] = None,
        sox_effects: Optional[List[List[str]]] = None,
        codecs: Optional[List[Dict[str, Any]]] = None,
        preemphasis: Optional[Dict[str, Any]] = None,
        non_linearities: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes an instance of a class.
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

        :param sox_effects: A list of Lists, each dictionary representing a SoX effect.

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
        :param non_linearity: A dictionary containing parameters for the non-linearity filter.
            - 'prob' (float): The probability of applying the non-linearity filter.
            - 'alpha' (float): The alpha value for the non-linearity filter.
        """
        import functools

        self._speed = PerturbationFactor(**speed) if speed else None
        self._tempo = PerturbationFactor(**tempo) if tempo else None
        self._sox_effects = sox_effects or []
        self._perturbations = []
        self.non_linearities = non_linearities
        if preemphasis:
            self._perturbations.append(functools.partial(self.preemphasis, factor=PerturbationFactor(**preemphasis)))
        if codecs:
            self._perturbations.append(functools.partial(self.apply_codecs, codecs=codecs))
        if non_linearities:
            self._perturbations.append(functools.partial(self.non_linearity, factor=PerturbationFactor(**non_linearities)))

    def run(self, audio, sample_rate, random_state):
        import numpy as np

        assert isinstance(audio, np.ndarray)
        assert len(audio.shape) == 1
        audio = audio.astype(np.float32)
        audio = self.sox(audio, sample_rate, random_state)
        for perturbation in self._perturbations:
            audio = perturbation(audio, sample_rate, random_state)
        return audio

    def sox(self, audio, sample_rate, random_state):
        import sox

        speed = False
        tfm = sox.Transformer()
        if self._speed is not None and random_state.random() < self._speed.prob:
            factor = random_state.random() * (self._speed.max - self._speed.min) + self._speed.min
            tfm.speed(factor)
            speed = True
        if self._tempo is not None and random_state.random() < self._tempo.prob and not speed:
            factor = random_state.random() * (self._tempo.max - self._tempo.min) + self._tempo.min
            tfm.stretch(factor)
        for effect in self._sox_effects:
            effect_name, *params = effect
            getattr(tfm, effect_name)(*params)
        audio = tfm.build_array(input_array=audio, sample_rate_in=sample_rate)
        return audio

    @staticmethod
    def preemphasis(audio, sample_rate, random_state, factor):
        import numpy as np

        def preemphasis_numpy(waveform, coeff=0.97):
            waveform = np.copy(waveform)
            waveform[..., 1:] -= coeff * waveform[..., :-1]
            return waveform

        if random_state.random() < factor.prob:
            preemphasis_coefficient = random_state.random() * (factor.max - factor.min) + factor.min
            audio = preemphasis_numpy(audio, coeff=preemphasis_coefficient)

        return audio

    @staticmethod
    def apply_codecs(audio, sample_rate, random_state, codecs):
        import sox

        tfm = sox.Transformer()
        for codec in codecs:
            prob = codec.pop("prob", 1.0)
            if random_state.random() < prob:
                if codec.get("encoding") == "ULAW":
                    tfm.set_output_format(encoding="u-law")
                else:
                    raise NotImplementedError(f"Codec {codec} not implemented.")
        return tfm.build_array(input_array=audio, sample_rate_in=sample_rate)

    @staticmethod
    def non_linearity(audio, sample_rate, random_state, factor):
        import numpy as np

        if random_state.random() < factor.prob:
            alpha = random_state.random() * (factor.max - factor.min) + factor.min
            audio = np.sign(audio) * np.abs(audio) ** (1 + alpha)
            audio = audio.astype(np.float32)
        return audio


def get_code_for_perturbation():
    classes = ["from typing import List, Dict, Any, Optional"]
    for cls_name, cls in list(globals().items()):
        if isinstance(cls, type):
            classes.append(cls)
    return classes
