def pretrain_construction_algo(idx, net_dict):
    orig_lstm_count = 0
    pool_idx = 0
    while "fwd_lstm_%i" % (orig_lstm_count + 1) in net_dict:
        orig_lstm_count += 1
        if "max_pool_%i" % (orig_lstm_count) in net_dict:
            pool_idx = orig_lstm_count

    num_lstm_layers = max(pool_idx + 1, orig_lstm_count // 2) + idx

    if num_lstm_layers > orig_lstm_count:
        return None

    if num_lstm_layers == pool_idx:
        out_from = ["max_pool_%i" % pool_idx]
    else:
        out_from = ["fwd_lstm_%i" % num_lstm_layers, "bwd_lstm_%i" % num_lstm_layers]

    for i in range(num_lstm_layers + 1, orig_lstm_count + 1):
        del net_dict["fwd_lstm_%i" % i]
        del net_dict["bwd_lstm_%i" % i]

    net_dict["output"]["from"] = out_from
    return net_dict
