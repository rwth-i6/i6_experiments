from i6_core.returnn.config import ReturnnConfig
from i6_experiments.users.schmitt.datasets.oggzip import get_dataset_dict
from sisyphus import Path

from typing import Dict


class DumpDatasetConfigBuilder:
  @staticmethod
  def get_dump_dataset_config(dataset_dict: Dict, extern_data_dict: Dict, dataset_dump_key: str):
    return ReturnnConfig(
      config=dict(
        backend="tensorflow",
        eval=dataset_dict,
        extern_data=extern_data_dict,
        network={
          "output": {"class": "copy", "from": "data:%s" % dataset_dump_key, "is_output_layer": True}}))


if __name__ == "__main__":
  dataset_dict = get_dataset_dict(
    oggzip_path_list=[Path(
      "/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.NSdIHfk1iw2M/output/out.ogg.zip",
      cached=True)],
    bpe_file=Path(
      "/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.codes"),
    vocab_file=Path(
      "/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab"),
    segment_file=None,
    fixed_random_subset=None,
    partition_epoch=1,
    pre_process=None,
    seq_ordering="sorted_reverse",
    epoch_wise_filter=None
  )

  extern_data_dict = {
    "data": {"available_for_inference": True, "shape": (None, 1), "dim": 1},
    "targets": {
      "available_for_inference": True,
      "shape": (None,),
      "dim": 10025,
      "sparse": True,
    },
  }

  returnn_config = DumpDatasetConfigBuilder.get_dump_dataset_config(dataset_dict, extern_data_dict, "targets")
  returnn_config.write("config.config")
