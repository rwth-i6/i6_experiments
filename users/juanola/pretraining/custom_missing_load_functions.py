


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
    new_name = name
    if ".base_model.model" in new_name:
        new_name = new_name.replace(".base_model.model", "")

    if ".base_layer" in new_name:
        new_name = new_name.replace(".base_layer", "")

    # Check if the transformed name exists in the checkpoint
    if new_name in preload_model_state:
        return preload_model_state[new_name]

    return None