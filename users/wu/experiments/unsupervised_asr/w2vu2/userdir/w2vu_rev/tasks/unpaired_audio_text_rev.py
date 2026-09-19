"""``unpaired_audio_text_rev`` -- the reproduction's task plus the reverse observations (SAE §4a step 4).

Two deltas over fairseq's ``unpaired_audio_text``, and nothing else:

1. ``load_dataset`` builds :class:`w2vu_rev.rev_dataset.RevExtractedFeaturesDataset`, which adds the
   unit stream, its length and the frozen eta to ``net_input`` (the only route to the model).  The
   unpaired-text wrapper, the dictionary, the shuffling, the aux target and the valid split are the
   reference's.
2. ``optimizer_step`` steps the ``reverse`` group together with ``generator``, so phi is updated in
   the generator update (SAE_4A_attrib.md step 4).  Its learning rate, betas and eps come from the
   ``reverse`` group in the config, not from here.  With a frozen phi there is no such group and the
   step is the reference's exactly.
"""

# NO `from __future__ import annotations` here: omegaconf 2.0.6 (fairseq 0.12.2's pin) reads the
# dataclass field types at registration and cannot resolve STRING annotations -- with the future
# import every field type becomes a string and hydra dies in `issubclass(type_, Enum)`.

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from fairseq.data import StripTokenDataset, data_utils
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import register_task
from unsupervised.tasks.unpaired_audio_text import UnpairedAudioText, UnpairedAudioTextConfig

from w2vu_rev.rev_dataset import RevExtractedFeaturesDataset

logger = logging.getLogger(__name__)


@dataclass
class UnpairedAudioTextRevConfig(UnpairedAudioTextConfig):
    rev_units_dir: Optional[str] = field(
        default=None,
        metadata={"help": "dir with {split}.rev500 and {split}.eta.npy (W2vu2RevUnitsJob)"},
    )


@register_task("unpaired_audio_text_rev", dataclass=UnpairedAudioTextRevConfig)
class UnpairedAudioTextRev(UnpairedAudioText):
    cfg: UnpairedAudioTextRevConfig

    def optimizer_step(self, optimizer, model, update_num):
        groups = {model.get_groups_for_update(update_num)}
        if "generator" in groups and not getattr(model, "rev_frozen", True):
            groups.add("reverse")
        optimizer.step(groups=groups)

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg
        assert self.cfg.rev_units_dir, "unpaired_audio_text_rev needs task.rev_units_dir"

        has_unpaired_text = os.path.exists(os.path.join(self.cfg.text_data, f"{split}.idx"))

        self.datasets[split] = RevExtractedFeaturesDataset(
            rev_units_dir=self.cfg.rev_units_dir,
            path=data_path,
            split=split,
            min_length=3,
            max_length=task_cfg.max_length,
            labels=None if has_unpaired_text else task_cfg.labels,
            label_dict=self.target_dictionary,
            shuffle=getattr(task_cfg, "shuffle", True),
            sort_by_length=task_cfg.sort_by_length,
            aux_target_postfix=task_cfg.aux_target_postfix,
        )

        logger.info(f"split {split} has unpaired text? {has_unpaired_text}")
        if has_unpaired_text:
            text_dataset = data_utils.load_indexed_dataset(
                os.path.join(self.cfg.text_data, split), self.target_dictionary
            )
            text_dataset = StripTokenDataset(text_dataset, self.target_dictionary.eos())
            from unsupervised.data import RandomInputDataset

            self.datasets[split] = RandomInputDataset(
                self.datasets[split],
                text_dataset,
                ["random_label"],
                add_to_input=True,
                pad_idx=self.target_dictionary.pad(),
            )
