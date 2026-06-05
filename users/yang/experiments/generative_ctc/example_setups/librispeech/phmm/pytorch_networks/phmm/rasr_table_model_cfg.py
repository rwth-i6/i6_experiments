from dataclasses import dataclass


@dataclass
class ModelConfig:
    table_file: str
    table_key: str = "transport"
    phoneme_tokens_key: str = "phoneme_tokens"
    cluster_tokens_key: str = "cluster_tokens"
    min_prob: float = 1e-30
    decode_model: str = "segment"
    merge_consecutive_same_cluster_ids_in_segment_mode: bool = False
    segment_start_downsample_rate: int = 2
    max_segment_start_mismatch_ratio: float = 0.1
    max_num_huge_segment_start_mismatches: int = 10

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
