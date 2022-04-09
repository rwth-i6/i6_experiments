import textwrap

PRETRAIN_1 = textwrap.dedent(
    """
next_func = lambda x : 2**x

def get_pretrain(
    idx=-1,
    pretain_idx=None,
    dim_fact=0.5    ):

    # returns pretrain dict...
    cur_blocks = pretain_idx(idx) # amount of pretrain encoder blocks
    start_blocks = 1

    # key sizes and dim have to be optained from encoder:
    end_blocks = encoder_args["num_blocks"]
    att_heads = encoder_args["att_num_heads"]

    transf_dec_layers = decoder_args['dec_layers']
    num_transf_layers = min(cur_blocks, transf_dec_layers)

	# check break condition:
    if cur_blocks > end_blocks:
        return None # TODO: handle better

    # calculate dimention reduction:
    grow_frac_enc = 1.0 - float(end_blocks - cur_blocks) / (end_blocks - start_blocks)
    dim_frac_enc = dim_fact + (1.0 - dim_fact) * grow_frac_enc

    # start adjusting the conformer params
    diff_opts = {}
    diff_opts["num_blocks"] = cur_blocks

    diff_opts_dec = {}
    diff_opts_dec["dec_layers"] = num_transf_layers

    # these need to be modified to reduce the dimensions
    for key in ['ff_dim', 'enc_key_dim', 'conv_kernel_size']:
        diff_opts[key] = int(encoder_args[key] * dim_frac_enc / float(att_heads)) * att_heads

    for key in ['ff_dim']:
        diff_opts_dec[key] = int(encoder_args[key] * dim_frac_enc / float(att_heads)) * att_heads

    # updated also dropout/l2 and co...
    for name in ["l2", "dropout"]:
        if name in encoder_args and encoder_args[name] is not None:
            diff_opts[name] = encoder_args[name] * dim_frac_enc if idx > 1 else 0.0 # 0.0 for fist 2 pretrain eps

        if name in decoder_args and decoder_args[name] is not None:
            diff_opts_dec[name] = decoder_args[name] * dim_frac_enc if idx > 1 else 0.0 # 0.0 for fist 2 pretrain eps
            

    updated_ops_enc = copy.deepcopy(encoder_args) # Than add the diff_opts
    updated_ops_dec = copy.deepcopy(decoder_args) # Than add the diff_opts
    for dp in diff_opts.keys():
        updated_ops_enc[dp] = diff_opts[dp]

    for dp in diff_opts_dec.keys():
        updated_ops_dec[dp] = diff_opts_dec[dp]

    # make the new encoder network
    conformer = ConformerEncoder(**updated_ops_enc)
    conformer.create_network()

    transformer = TransformerDecoder(base_model=conformer, **updated_ops_dec) 
    transformer.create_network()

    net = conformer.network.get_net()
    net.update(transformer.network.get_net())

    return net


all_nets_pt = []

def get_net_construct(idx, net_dict):

    net = get_pretrain(
                                idx=idx,
                                pretain_idx=next_func)

    return net # and wezzz got our new network

i = 0
while True:
    net = get_net_construct(i, None)
    i += 1

    if net is None:
        break
    all_nets_pt.append(net)

def custom_construction_algo(idx, net_dict):
    if len(all_nets_pt) <= idx:
        return None
    return all_nets_pt[idx]

pretrain = {"repetitions": 6, "copy_param_mode": "subset", "construction_algo": custom_construction_algo}
    """
)