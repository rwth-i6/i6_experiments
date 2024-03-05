"""
Helpers for the Sisyphus setup
"""

from __future__ import annotations
import os


_my_dir = os.path.dirname(os.path.abspath(__file__))


def get_prefix_for_config(src_filename: str):
    """
    :param src_filename: pass `__file__` here
    :return: some prefix name
    """
    assert src_filename.endswith(".py")
    assert src_filename.startswith(_my_dir + "/"), f"unexpected prefix in {src_filename}"
    src_filename = src_filename[len(_my_dir) + 1 :]
    assert "/" not in src_filename, f"unexpected path separator in {src_filename}"
    exp_name = src_filename[:-3]
    return "exp2023_10_20_rf/" + exp_name
