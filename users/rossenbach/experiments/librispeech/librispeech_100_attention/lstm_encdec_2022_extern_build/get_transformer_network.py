


def construct_network(epoch: int, source_data: nn.Data, target_data: nn.Data, **kwargs):
    start = time.time()
    feature_dim = source_data.dim_tags[-1]
    time_dim = source_data.dim_tags[-2]

    label_dim = target_data.sparse_dim
    label_time_dim = target_data.dim_tags[-1]

    print("context: %f" % (time.time() - start))
    start = time.time()
    net = BLSTMDownsamplingTransformerASR(audio_feature_dim=feature_dim, target_vocab=label_dim)
    print("net building: %f" % (time.time() - start))
    start = time.time()
    out = net(
        audio_features=nn.get_extern_data(source_data),
        labels=nn.get_extern_data(target_data),
        audio_time_dim=time_dim,
        label_time_dim=label_time_dim,
        label_dim=label_dim,
    )
    print("net calling: %f" % (time.time() - start))
    start = time.time()
    out.mark_as_default_output()
    print("mark output: %f" % (time.time() - start))

    start = time.time()
    for param in net.parameters():
        param.weight_decay = 0.1
    print("weight decay: %f" % (time.time() - start))

    return net

