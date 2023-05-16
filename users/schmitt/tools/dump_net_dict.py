#!/usr/bin/env python

"""
Dumps the network topology as JSON on stdout.
"""

from __future__ import print_function

import sys
import argparse
import json
import typing

# import returnn.__main__ as rnn
# from returnn.log import log
# from returnn.pretrain import pretrain_from_config
# from returnn.config import network_json_from_config
import returnn.config

config = None  # type: typing.Optional["returnn.config.Config"]


def init(config_filename, command_line_options):
  """
  :param str config_filename:
  :param list[str] command_line_options:
  """
  rnn.init_better_exchook()
  rnn.init_config(config_filename, command_line_options)
  global config
  config = rnn.config
  config.set("log", [])
  rnn.init_log()
  print("RETURNN dump-dataset starting up.", file=rnn.log.v3)
  rnn.init_config_json_network()


def main(argv):
  """
  Main entry.
  """
  arg_parser = argparse.ArgumentParser(description='Dump network as JSON.')
  arg_parser.add_argument('returnn_config_file')
  arg_parser.add_argument('--epoch', default=1, type=int)
  arg_parser.add_argument('--out', default="/dev/stdout")
  arg_parser.add_argument("--returnn-root", help="path to returnn root")
  args = arg_parser.parse_args(argv[1:])

  sys.path.insert(0, args.returnn_root)
  global rnn
  import returnn.__main__ as rnn

  init(config_filename=args.returnn_config_file, command_line_options=[])

  # if rnn.engine.is_pretrain_epoch(args.epoch):
  # network = rnn.engine.get_net_dict_for_epoch(args.epoch, config)
  network = returnn.config.network_json_from_config(config)
  # else:
  #   network = rnn.engine.network.get_net

  # json_data = network.to_json_content()
  f = open(args.out, 'w')
  f.write("Network dictionary:\n\n")
  f.write(
    "\n".join("%s: %s" % (k, v) for k, v in network.items() if k != "output")
  )
  # print(json.dumps(network, indent=2, sort_keys=True), file=f)
  f.write("\n\nNetwork Output:\n\n")
  f.write("\n".join("%s: %s" % (k, v) for k, v in network["output"]["unit"].items()))

  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)