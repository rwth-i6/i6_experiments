__all__ = ["init_segment_order_shuffle"]

import copy

def init_segment_order_shuffle(system):
    system.csp["crnn_train"] = copy.deepcopy(system.csp["crnn_train"])
    system.csp["crnn_train"].corpus_config.segment_order_shuffle = True
    system.csp["crnn_train"].corpus_config.segment_order_sort_by_time_length = True
    system.csp["crnn_train"].corpus_config.segment_order_sort_by_time_length_chunk_size = 1000