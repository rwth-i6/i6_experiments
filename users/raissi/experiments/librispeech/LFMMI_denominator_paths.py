# the structure is first level lm scale and second level tdp scale

general_prepath = "/work/asr4/raissi/denominators_latticefree/lbs/final"

subpaths = {
    1: {
        "word": ("/").join([general_prepath, "word_unigram"]),
        "phone": ("/").join([general_prepath, "phon_unigram"]),
    },
    2: {
        "word": ("/").join([general_prepath, "word_bigram"]),
        "phone": ("/").join([general_prepath, "phon_bigram"]),
    },
}


def get_denominator_path(
    lm_scale,
    tdp_scale,
    lm_type="word",
    lm_gram=1,
    apply_weight_pushing=False,
):
    subname = f"denominators/denom_LM{lm_scale}_tdp{tdp_scale}.fsa"
    hash_key = "weight_push_mint" if apply_weight_pushing else "base_mint"
    return ("/").join([subpaths[lm_gram][lm_type], WORD_UNIGRAM_HASHES[hash_key], subname])


# min{t,l} for minimization operation done in tropical or log
WORD_UNIGRAM_HASHES = {"base_mint": "q5fQcOjT", "weight_push_mint": "JRGOMfUw"}
