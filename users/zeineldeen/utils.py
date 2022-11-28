def parse_cnn_args(inputs):
    """
    Parse CNN args such as filter sizes, pooling sizes
    For example, for 2 CNN layers with filter sizes 3x3, inputs would be a string of the following format:
        "(3,3)_(3,3)"

    :params str inputs: cnn args to parse
    """
    return [(int(x.split(',')[0].replace('(', '')),
             int(x.split(',')[1].replace(')', ''))) for x in inputs.split('_')]
