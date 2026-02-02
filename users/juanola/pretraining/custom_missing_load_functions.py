


def qwen_load_lora_adapted_weights(name, shape, preload_model_state, **kwargs):
    """
    FROM: SLLM robin repo
    Tie embedding matrices by loading the output embedding matrix from the input embedding matrix.

    :param name: name of the parameter to load (missing_keys_preload)
    :preload_model_state: dict of pretrained weights

    # LoRA       - decoder.base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight
    # pretrained - decoderXXXXXXXXXXXXXXXXX.model.layers.0.self_attn.q_projXXXXXXXXXXX.weight
    """
    # Ignore LoRA layers
    if ".lora_" in name:
        return None

    if name == "decoder.base_model.model.model.embed_tokens.weight":
        print(f"Loading embedding layer from decoder_embed_func instead from decoder.model.embed_tokens.weight!!!")
        return preload_model_state["decoder_embed_func.weight"]

    # Remove extra tags added by lora
    if ".base_model.model" in name:
        s1, s_aux = name.split(".base_model.model")
        if ".base_layer" in name:
            s2, s3 = s_aux.split(".base_layer")
            return preload_model_state[f"{s1}{s2}{s3}"]
        else:
            return preload_model_state[f"{s1}{s_aux}"]

    return None