def get_extra_config_segment_order(size=128):
    extra_config = sprint.SprintConfig()
    extra_config['*'].segment_order_sort_by_time_length = True
    extra_config['*'].segment_order_sort_by_time_length_chunk_size = size

    return extra_config