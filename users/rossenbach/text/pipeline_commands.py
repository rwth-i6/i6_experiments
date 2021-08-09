"""
A collection of pipeline commands and helpers that can be used to process text with
the PipelineJob
"""

from typing import *

pipe_squeeze_repeating_whitespaces = "tr -s ' '"
pipe_normalize_double_hyphen = "sed \"s/--/-/g\""
pipe_split_dots = "sed \"s/\\./ . /g\""


def get_whitelist_filter_pipe(
        allow_list: List[str],
        allow_lowercase_characters=True,
        allow_uppercase_characters=True,
        allow_numbers=True,
    ) -> str:
    """
    This function creates an sed command
    that will remove all invalid characters

    :param allow_list:
    :param allow_lowercase_characters:
    :param allow_uppercase_characters:
    :param allow_numbers:
    :return:
    """

    whitelist_string = " "
    if allow_lowercase_characters:
        whitelist_string += "a-z"
    if allow_uppercase_characters:
        whitelist_string += "A-Z"
    if allow_numbers:
        whitelist_string += "0-9"

    needs_escape_set = ["!"]

    def add_escape(c):
        if c in needs_escape_set:
            return "\\" + c
        else:
            return c

    whitelist_string += "".join([add_escape(c) for c in allow_list])

    return "sed \"s/[^ " + whitelist_string + "]//g\""

