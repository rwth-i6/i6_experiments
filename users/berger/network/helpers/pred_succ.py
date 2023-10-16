from typing import Dict, List


def add_pred_succ_targets_noblank(
    network: Dict,
    num_classes: int,
    nonword_labels: List[int],
    base_labels: str = "data:classes",
):

    # Example:
    # Classes = AAABBCCCDD
    # Let A be a nonword label, e.g. silence
    # Then predecessors should be [[[[[BBBCC
    # And successors should be [[[CCDDD[[

    # ====== General layers ======

    # [
    network["boundary_symbol"] = {
        "class": "eval",
        "from": base_labels,
        "eval": f"tf.ones_like(source(0)) * {num_classes}",
    }

    # True at positions of non-word classes
    # 1110000000
    eval_str = f"tf.math.not_equal(source(0), {nonword_labels[0]})"
    for label in nonword_labels[1:]:
        eval_str = f"tf.math.logical_and(tf.math.not_equal(source(0), {label}), {eval_str})"
    network["mask_no_nonword"] = {
        "class": "eval",
        "from": base_labels,
        "eval": eval_str,
        "out_type": {"dtype": "bool"},
    }

    # Non-word classes replaced by boundary symbol
    # [[[BBCCCDD
    network["targets_no_nonword"] = {
        "class": "switch",
        "condition": "mask_no_nonword",
        "true_from": base_labels,
        "false_from": "boundary_symbol",
    }
    network["targets_no_nonword_reinterpret"] = {
        "class": "reinterpret_data",
        "from": "targets_no_nonword",
        "increase_sparse_dim": 1,
    }

    # Shift targets forward by one
    #  AAABBCCCD
    network["targets_shifted"] = {
        "class": "shift_axis",
        "from": base_labels,
        "axis": "T",
        "amount": 1,
        "pad": False,
        "adjust_size_info": True,
    }

    # Pad by adding boundary symbol at the start
    # [AAABBCCCD
    network["targets_shifted_padded"] = {
        "class": "pad",
        "from": "targets_shifted",
        "padding": (1, 0),
        "value": num_classes,
        "axes": "T",
        "mode": "constant",
    }
    network["targets_padded_reinterpret"] = {
        "class": "reinterpret_data",
        "from": "targets_shifted_padded",
        "size_base": base_labels,
        "increase_sparse_dim": 1,
    }

    # True at the first position of each label segment
    # 1001010010
    network["mask_first_label"] = {
        "class": "compare",
        "from": [base_labels, "targets_padded_reinterpret"],
        "kind": "not_equal",
    }

    # ====== Predecessor specific layers ======

    # Base labels to be expanded to predecessor sequence
    # [BCD
    network["pred_labels"] = {
        "class": "masked_computation",
        "from": "targets_no_nonword_reinterpret",
        "mask": "mask_first_label",
        "unit": {"class": "copy"},
    }

    # Pad by boundary label, the initial output will be skipped at unmasking and the pad is the predecessor of the first label
    # [[[BCD
    network["pred_labels_padded"] = {
        "class": "pad",
        "from": "pred_labels",
        "padding": (1, 0),
        "axes": "T",
        "value": num_classes,
        "mode": "constant",
        "initial_output": num_classes,
    }

    # Expand to full predecessor sequence
    # [[[[[BBBCC
    network["base_predecessors"] = {
        "class": "unmask",
        "from": "pred_labels_padded",
        "mask": "mask_first_label",
    }

    # All non-word phones are context-less and have no predecessors
    # [[[[[BBBCC
    network["predecessors"] = {
        "class": "switch",
        "condition": "mask_no_nonword",
        "true_from": "base_predecessors",
        "false_from": "boundary_symbol",
    }

    # ====== Successor specific layers ======

    # Add final 1 to mask
    # 10010100101
    network["mask_first_label_padded"] = {
        "class": "pad",
        "from": "mask_first_label",
        "padding": (0, 1),
        "axes": "T",
        "value": True,
        "mode": "constant",
    }

    # Add final boundary symbol to targets
    # [[[BBCCCDD[
    network["targets_no_nonword_padded_right"] = {
        "class": "pad",
        "from": "targets_no_nonword_reinterpret",
        "padding": (0, 1),
        "axes": "T",
        "value": num_classes,
        "mode": "constant",
    }
    network["targets_padded_right_reinterpret"] = {
        "class": "reinterpret_data",
        "from": "targets_no_nonword_padded_right",
        "size_base": "mask_first_label_padded",
    }

    # Base labels to be expanded to successor sequence
    # [[BCD[
    network["succ_labels"] = {
        "class": "masked_computation",
        "from": "targets_padded_right_reinterpret",
        "mask": "mask_first_label_padded",
        "unit": {"class": "copy", "initial_output": num_classes},
    }

    # 11001010010
    network["mask_first_label_init_1"] = {
        "class": "pad",
        "from": "mask_first_label",
        "padding": (1, 0),
        "axes": "T",
        "value": True,
        "mode": "constant",
    }

    # Expand to full successor sequence
    # [BBBCCDDD[[
    network["successors_init"] = {
        "class": "unmask",
        "from": "succ_labels",
        "mask": "mask_first_label_init_1",
    }
    # Remove initial label again
    # BBBCCDDD[[
    network["successors_base"] = {
        "class": "shift_axis",
        "from": "successors_init",
        "axis": "T",
        "amount": -1,
        "pad": False,
        "adjust_size_info": True,
    }
    network["successors_base_reinterpret"] = {
        "class": "reinterpret_data",
        "from": "successors_base",
        "size_base": base_labels,
    }

    # All non-word phones are context-less and have no successors
    # [[[CCDDD[[
    network["successors"] = {
        "class": "switch",
        "condition": "mask_no_nonword",
        "true_from": "successors_base_reinterpret",
        "false_from": "boundary_symbol",
    }

    return "predecessors", "successors"


def add_pred_succ_targets_blank(
    network: Dict,
    nonword_labels: List[int],
    base_labels: str = "data:classes",
    blank_index: int = 0,
):

    # Example:
    # Classes = ..A.B..C..D
    # Let A be a nonword label, e.g. silence
    # Then predecessors should be .......B..C
    # And successors should be ....C..D...

    # ====== General layers ======

    # [
    network["blank_symbol"] = {
        "class": "eval",
        "from": base_labels,
        "eval": f"tf.ones_like(source(0)) * {blank_index}",
    }

    # True at positions of symbols that are neither blank nor nonword
    # 00001001001
    eval_str = f"tf.math.not_equal(source(0), {blank_index})"
    for label in nonword_labels:
        eval_str = f"tf.math.logical_and(tf.math.not_equal(source(0), {label}), {eval_str})"
    network["mask_true_symbol"] = {
        "class": "eval",
        "from": base_labels,
        "eval": eval_str,
        "out_type": {"dtype": "bool"},
    }

    # Reduced sequence by removing all blanks and nonword symbols
    # BCD
    network["true_labels"] = {
        "class": "masked_computation",
        "from": base_labels,
        "mask": "mask_true_symbol",
        "unit": {"class": "copy"},
    }

    # ====== Predecessor specific layers ======

    # Pad left with two blank symbols
    # ..BCD
    network["labels_padded_left"] = {
        "class": "pad",
        "from": "true_labels",
        "padding": (1, 0),
        "axes": "T",
        "value": blank_index,
        "mode": "constant",
        "initial_output": blank_index,
    }

    # Unfold over sequence length
    # .......BBBC
    network["predecessors_base"] = {
        "class": "unmask",
        "from": "labels_padded_left",
        "mask": "mask_true_symbol",
    }

    # Remove predecessors for blank and non-word labels
    # .......B..C
    network["predecessors"] = {
        "class": "switch",
        "condition": "mask_true_symbol",
        "true_from": "predecessors_base",
        "false_from": "blank_symbol",
    }

    # ====== Successor specific layers ======

    # Pad right with a blank symbol
    # .BCD.
    network["labels_padded_right"] = {
        "class": "pad",
        "from": "true_labels",
        "padding": (0, 1),
        "axes": "T",
        "value": blank_index,
        "mode": "constant",
        "initial_output": blank_index,
    }

    # Add initial 1 to mask
    # 100001001001
    network["mask_true_symbol_init_1"] = {
        "class": "pad",
        "from": "mask_true_symbol",
        "padding": (1, 0),
        "axes": "T",
        "value": True,
        "mode": "constant",
    }

    # Unfold over sequence length
    # BBBBBCCCDDD.
    network["successors_init"] = {
        "class": "unmask",
        "from": "labels_padded_right",
        "mask": "mask_true_symbol_init_1",
    }

    # Remove initial label again
    # BBBBCCCDDD.
    network["successors_base"] = {
        "class": "shift_axis",
        "from": "successors_init",
        "axes": "T",
        "amount": -1,
        "pad": False,
        "adjust_size_info": True,
    }
    network["successors_base_reinterpret"] = {
        "class": "reinterpret_data",
        "from": "successors_base",
        "size_base": base_labels,
    }

    # Remove predecessors for blank and non-word labels
    # ....C..D...
    network["successors"] = {
        "class": "switch",
        "condition": "mask_true_symbol",
        "true_from": "successors_base_reinterpret",
        "false_from": "blank_symbol",
    }

    return "predecessors", "successors"
