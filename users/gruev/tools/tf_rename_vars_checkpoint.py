#!/usr/bin/env python3

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Script that renames variables stored in a TF checkpoint according to renaming rules.
# This script is generic for any TF checkpoint. It is not specific to RETURNN.

from __future__ import annotations

import _setup_returnn_env  # noqa

import re
import sys
import numpy
import logging
import argparse
import tensorflow as tf

from tensorflow.python.platform import app
import returnn.tf.compat as tf_compat

FLAGS = None

# Check whether the provided path is valid
def checkpoint_exists(path):
    return (
        tf_compat.v1.gfile.Exists(path)
        or tf_compat.v1.gfile.Exists(path + ".meta")
        or tf_compat.v1.gfile.Exists(path + ".index")
    )


# Simple update based on parsed rule pairs
def update_variable_name(var_name, rules):
    for old, new in rules.items():
        var_name = re.compile(old).sub(new, var_name)
    return var_name


# Main logic for creation of renamed variables
def update_variable_names_in_checkpoint_file(checkpoint_path, output_path, rules):
    """Updates and possibly omit variable names in a checkpoint file.

    Args:
      checkpoint_path str: Name of the checkpoint file.
      output_path str: Name of checkpoint file after variable renaming.
      rules Dict[str, str]: Replacement of substrings in variable names.
    """
    checkpoint = checkpoint_path.strip()
    assert checkpoint and checkpoint_exists(checkpoint), "Invalid checkpoint"
    assert type(rules) == dict, "Invalid rules type %s" % type(rules)

    reader = tf_compat.v1.train.NewCheckpointReader(checkpoint)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_to_dtype_map = reader.get_variable_to_dtype_map()

    # tf_compat.v1.logging.info("Reading and renaming checkpoint variables...")
    var_names = sorted(var_to_shape_map.keys())
    for v in var_names:
        print(update_variable_name(v, rules))

    with tf_compat.v1.variable_scope(tf_compat.v1.get_variable_scope(), reuse=tf_compat.v1.AUTO_REUSE):
        tf_vars = [
            tf_compat.v1.get_variable(
                update_variable_name(v, rules), shape=var_to_shape_map[v], dtype=var_to_dtype_map[v]
            )
            for v in var_names
        ]

    placeholders = [tf_compat.v1.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    assign_ops = [tf_compat.v1.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
    # tf.compat.v1.all_variables() is deprecated
    saver = tf_compat.v1.train.Saver(tf_compat.v1.global_variables())

    with tf_compat.v1.Session() as sess:
        sess.run(tf_compat.v1.global_variables_initializer())
        for p, assign_op, v in zip(placeholders, assign_ops, var_names):
            sess.run(assign_op, {p: reader.get_tensor(v)})
        saver.save(sess, output_path)


"""
### Use variable scope, useful for variable sharing when using tf.compat.v1.get_variable().
### tf_compat.v1.get_variable_scope(): returns the current variable scope
### reuse=tf_compat.v1.AUTO_REUSE: essentially enables variable sharing within the same scope 

### tf_compat.v1.get_variable(): fetches the variable if it exists, or creates it otherwise.
### Feeding placeholders: use the feed_dict argument to Session.run() (or Tensor.eval()).
### Here, tf.Variable will hold the value assigned later to the corresponding placeholder.

### tf_compat.v1.all_variables(): list of all variables in the current graph.
### saver has access to all variables in the current graph.

with tf_compat.v1.variable_scope(tf_compat.v1.get_variable_scope(), reuse=tf_compat.v1.AUTO_REUSE):  
    tf_vars = [tf_compat.v1.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v]) for v in var_values]
placeholders = [tf_compat.v1.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
assign_ops = [tf_compat.v1.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
saver = tf_compat.v1.train.Saver(tf_compat.v1.all_variables())
"""


# noinspection PyUnusedLocal
def main(unused_argv):
    """
    Main entry:
    """
    if not FLAGS.checkpoint_path or not FLAGS.output_path:
        print(
            "Usage: rename_vars_checkpoint --checkpoint_path=checkpoint_in --output_path=checkpoint_out --rules=old1:new1,old2:new2 ..."
        )
        sys.exit(1)
    else:
        update_variable_names_in_checkpoint_file(FLAGS.checkpoint_path, FLAGS.output_path, FLAGS.rules)


if __name__ == "__main__":

    def parse_rules(arg):
        rules = dict()
        try:
            rule_pairs = arg.split(",")
            for rule_pair in rule_pairs:
                old_substr, new_substr = rule_pair.split(":")
                rules[old_substr] = new_substr
            return rules

        except (ValueError, IndexError):
            raise argparse.ArgumentTypeError("Invalid rule format.")

    parser = argparse.ArgumentParser()
    # parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Input checkpoint filename. "
        # "Note, if using Checkpoint V2 format, file_name is the "
        # "shared prefix between all files in the checkpoint.",
    )
    parser.add_argument(
        "--output_path", type=str, default="", help="Output checkpoint filename with renamed variables. "
    )
    parser.add_argument(
        "--rules",
        type=parse_rules,
        default="",
        help="Renaming rules in format old1:new1,old2:new2... with ':' as rule delimiter and ',' as instance delimiter. ",
    )
    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
