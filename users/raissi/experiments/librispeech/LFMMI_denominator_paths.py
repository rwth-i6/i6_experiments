# the structure is first level lm scale and second level tdp scale

general_prepath = "/work/asr4/raissi/denominators_latticefree/lbs/final"
general_prepath_idiap = "/remote/idiap.svm/temp.speech01/traissi/denominators/lbs/"

subpaths = {
    -1: ("/").join([general_prepath, "no_lm"]),
    1: {
        "word": ("/").join([general_prepath, "word_unigram"]),
        "phone": ("/").join([general_prepath, "phon_unigram"]),
    },
    2: {
        "word": ("/").join([general_prepath, "word_bigram"]),
        "phone": ("/").join([general_prepath, "phon_bigram"]),
    },
}

subpaths_idiap = {
    -1: ("/").join([general_prepath_idiap, "no_lm"]),
    1: {
        "word": ("/").join([general_prepath_idiap, "word_unigram"]),
        "phone": ("/").join([general_prepath_idiap, "phon_unigram"]),
    },
    2: {
        "word": ("/").join([general_prepath_idiap, "word_bigram"]),
        "phone": ("/").join([general_prepath_idiap, "phon_bigram"]),
    },
}


def get_denominator_path(
    tdp_scale,
    lm_scale=1.0,
    root_to_sil_weight=3.5, #dummy number indicating that there was a weight pushing for unigram LM
    normalize_all_root_to_sil=False,
    lm_type="word",
    lm_gram=1,
    apply_weight_pushing=False,
    use_idiap_cluster=False,

):

    assert lm_type in ["word", "phone", "no_lm"]
    sp = subpaths_idiap if use_idiap_cluster else subpaths
    hash_key = "weight_push_mint" if apply_weight_pushing else "base_mint"

    if lm_type == "no_lm":
        subname = f"denominators/denom_tdp{tdp_scale}_root{'all' if normalize_all_root_to_sil else 'sil'}norm.pickle"
        return ("/").join([sp[-1], NO_LM[hash_key], subname])

    subname = f"denominators/denom_LM{lm_scale}_tdp{tdp_scale}_rootSil{root_to_sil_weight}.pickle"

    return ("/").join([sp[lm_gram][lm_type], WORD_UNIGRAM_HASHES[hash_key], subname])


# min{t,l} for minimization operation done in tropical or log
WORD_UNIGRAM_HASHES = {"base_mint": "q5fQcOjT", "weight_push_mint": "JRGOMfUw"}
WORD_BIGRAMS_HASHES = {"weight_push_mint": "i13aGq6Z"}
NO_LM = {"weight_push_mint": "JnuxwUM5"}
