
def get_g2p_parameters():
    g2p_args = {}

    g2p_args["train"] = {
        "num_ramp_ups"   :4,
        "min_iter"       :1,
        "max_iter"       :60,
        "devel"          :"5%",
        "size_constrains":"0,1,0,1"
    }

    g2p_args["apply"] = {
        "variants_mass"  :1.0,
        "variants_number":1
    }

    return g2p_args
