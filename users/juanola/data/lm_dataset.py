from typing import Any, Dict, Optional

from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat

from i6_core.returnn.config import CodeWrapper
from i6_experiments.common.setups.returnn.datasets import ControlDataset


class LmDataset(ControlDataset):

    def __init__(
            self,
            *,
            corpus_file: tk.Path,
            vocab_settings: Optional[Dict[str, Any]] = None,
            vocab_file: tk.Path = None,
            # super parameters
            partition_epoch: Optional[int] = None,
            segment_file: Optional[tk.Path] = None,
            seq_ordering: Optional[str] = None,
            random_subset: Optional[int] = None,
            additional_options: Optional[Dict] = None,
    ):
        super().__init__(
            partition_epoch=partition_epoch,
            segment_file=segment_file,
            seq_ordering=seq_ordering,
            random_subset=random_subset,
            additional_options=additional_options
        )

        self.corpus_file = corpus_file

        assert vocab_settings is not None or vocab_file is not None, "Needs a vocab param"
        assert not (vocab_settings is not None and vocab_file is not None), "Both vocab params should not be set"
        self.target_options = vocab_settings
        self.vocab_file = vocab_file

    def as_returnn_opts(self) -> Dict[str, Any]:
        d = {
            "class": "LmDataset",
            "corpus_file": self.corpus_file, # CodeWrapper(DelayedFormat('lambda: cf("{}")', self.corpus_file)),
            "orth_replace_map_file": "",
            "word_based": True,
            "seq_end_symbol": "</s>",
            "auto_replace_unknown_symbol": False,
            "unknown_symbol": "<unk>",
            "add_delayed_seq_data": True, #creates the "delayed data!"
            "delayed_seq_data_start_symbol": "<s>",
        }

        if self.target_options is not None:
            d["orth_vocab"] = self.target_options
        elif self.vocab_file is not None:
            d["orth_symbols_map_file"] = self.vocab_file
        else:
            raise Exception("should have some vocab option!")


        sd = super().as_returnn_opts()
        assert all([k not in sd.keys() for k in d.keys()]), (
            "conflicting keys in %s and %s"
            % (str(list(sd.keys())), str(list(d.keys()))),
        )
        d.update(sd)

        return d
