
from typing import cast

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.embeddings import build_training_datasets
from ...data.common import DatasetSettings

def run_xvector_system_try1():
    prefix_name = "experiments/loquacious/standalone_2025/speaker_embeddings/xvector_try1"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=5,
        train_seq_ordering="random",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_training_datasets(
        prefix=prefix_name,
        loquacious_key="train.medium",
        settings=train_settings,
    )
    speaker_datastream = cast(LabelDatastream, train_data.datastreams["speaker_labels"])
    speaker_labels = speaker_datastream.vocab_size