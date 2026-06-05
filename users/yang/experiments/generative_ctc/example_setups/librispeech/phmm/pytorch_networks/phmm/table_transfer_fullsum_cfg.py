from dataclasses import dataclass


@dataclass
class ModelConfig:
    input_vocab_size: int
    output_vocab_size: int
    lm_table_path: str
    lm_vocab_size: int = 41
    lm_context_length: int = 3
    beam_size: int = 200
    softmax_temperature: float = 1.0
    use_lm_silence_score: bool = False
    lm_scale: float = 0.6
    am_scale: float = 1.0
    table_init_scale: float = 0.02

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
