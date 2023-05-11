#!/usr/bin/env python3
#!vim: set sw=2:

"""
This script reads some alignment (B, T)
and copies it to some HDF-file, but replaces some values.
Hacky way to swap blank-indices for RNA/RNN-T models. (blank = 0 or last)
"""

import os
import sys
from argparse import ArgumentParser
from returnn.util import better_exchook


def run_replace_hdf(args):
  returnn_root = args.returnn_root
  sys.path.insert(0, returnn_root)
  from returnn.tf.engine import Engine, Runner
  from returnn.datasets import init_dataset
  from returnn.config import get_global_config
  from returnn.util.basic import get_login_username

  config = get_global_config(auto_create=True)
  config.update(dict(
    batching="sorted",
    batch_size=100000,
    max_seqs=1000,
    log_batch_size=True,

    network={
      "constant_replace_with0": {"class": "constant", "value": args.replace_with, "dtype": "int32", "with_batch_dim": True},
      "constant_replace_with": {"class": "reinterpret_data", "from": "constant_replace_with0", "set_sparse": True, "set_sparse_dim": 1031},
      "replace": {"class": "compare", "from": "data", "value": args.replace_value, "kind": "equal"},
      "switch": {"class": "switch", "from": [], "condition": "replace", "true_from": "constant_replace_with", "false_from": "data"},
      "output": {"class": "hdf_dump", "filename": "dump.hdf", "from": ["switch"], "is_output_layer": True},
    },
    allow_random_model_init=True,
    extern_data = {"data": {"dim": 1031, "sparse": True, "dtype": "int32"}},
    model="/tmp/%s/copy-align/model" % get_login_username(),
    cleanup_old_models=True,
    log_verbosity=5
  ))

  engine = Engine(config)
  engine.init_network_from_config()
  config = get_global_config()
  engine.network.recurrent = True  # we need to set this in order to get not-flat data

  dump_layer = engine.network.layers["output"]
  batch_size = config.typed_value('batch_size', 1)

  replaced_hdf_file = args.swap_alignment_path
  hdf_file = args.alignment_path
  if os.path.exists(replaced_hdf_file):
      print("'%s' already exists. skipping." % replaced_hdf_file)
      exit(0)
  print("Reading from '%s'" % hdf_file)
  print("Replacing all occurrences of %d with %d." % (args.replace_value, args.replace_with))
  print("Writing to '%s'" % replaced_hdf_file)
  dump_layer.filename = replaced_hdf_file
  dataset = init_dataset({"class": "HDFDataset", "files": [hdf_file]})
  dataset.init_seq_order(epoch=1)

  dataset_batches = dataset.generate_batches(
    recurrent_net=engine.network.recurrent,
    batch_size=batch_size,
    max_seqs=config.int('max_seqs', -1),
    used_data_keys=engine.network.get_used_data_keys())


  runner = Runner(
    engine=engine, dataset=dataset, batches=dataset_batches,
    train=False, eval=False)
  runner.run(report_prefix=engine.get_epoch_str() + " %r dump copy-align" % args.alignment_path)
  if not runner.finalized:
    print("Runner not finalized, quitting.")
    sys.exit(1)
  assert dump_layer.hdf_writer  # nothing written?
  engine.network.call_graph_reset_callbacks()


def main():
  better_exchook.install()
  arg_parser = ArgumentParser()
  arg_parser.add_argument("--alignment-path")
  arg_parser.add_argument("--swap-alignment-path")
  arg_parser.add_argument("--replace-value", default=None, type=int)
  arg_parser.add_argument("--replace-with", default=None, type=int)
  arg_parser.add_argument("--suffix", default=".swap")
  arg_parser.add_argument("--returnn-root")
  args = arg_parser.parse_args()
  run_replace_hdf(args)


if __name__ == "__main__":
  main()



