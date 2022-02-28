from i6_core.rasr.flow import FlowNetwork

def label_alignment_flow(feature_net, alignment_cache_path=None):
    """
    Alignment flow specifically for Weis label aligner

    :param feature_net:
    :param alignment_cache_path:
    :return:
    """
    assert 'features' in feature_net.get_output_ports()
    net = FlowNetwork()
    net.add_output('alignments')
    net.add_param(['id', 'orthography', 'TASK'])

    mapping = net.add_net(feature_net)
    net.interconnect_inputs (feature_net, mapping)
    net.interconnect_outputs(feature_net, mapping)

    aggregate = net.add_node('generic-aggregation-vector-f32', 'aggregate')
    net.link(mapping[feature_net.get_output_links('features').pop()], aggregate)

    alignment = net.add_node('speech-label-alignment', 'alignment', {'id': '$(id)', 'orthography': '$(orthography)'})
    net.link(aggregate, alignment)

    if alignment_cache_path is not None:
        cache = net.add_node('generic-cache', 'alignment-cache', {'id': '$(id)', 'path': alignment_cache_path})
        net.link(alignment, cache               )
        net.link(cache    , 'network:alignments')
    else:
        net.link(alignment, 'network:alignments')

    return net