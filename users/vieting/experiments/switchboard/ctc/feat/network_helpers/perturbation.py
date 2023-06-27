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
    Helper class to perform perturbation techniques on audio waveforms.
    """

    def __init__(
        self,
        speed: dict = None,
        tempo: dict = None,
        sox_effects: list[dict] = None,
        codecs: list[dict] = None,
        preemphasis: dict = None,
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
    classes = ["import torch", "import numpy as np", "import torchaudio", "import functools", "import random"]
    for cls_name, cls in list(globals().items()):
        if isinstance(cls, type):
            classes.append(cls)
    return classes
