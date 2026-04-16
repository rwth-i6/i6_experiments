

audio_text_v1 = {
    "aux_loss_layers": (),
    "num_enc_layers": 6,
    "num_text_dec_layers": 3,
    "num_audio_dec_layers": 3,
    "num_heads": 8,
    "model_dim": 512,
    # "text_out_dim": 41,  # this is set later
}

audio_text_v2 = {
    "aux_loss_layers": (),
    "num_enc_layers": 6,
    "num_text_dec_layers": 3,
    "num_audio_dec_layers": 3,
    "num_heads": 8,
    "model_dim": 512,
}

audio_v1 = {
    "aux_loss_layers": (),
    "num_enc_layers": 6,
    "num_text_dec_layers": 0,
    "num_audio_dec_layers": 3,
    "num_heads": 8,
    "model_dim": 512,
    # "text_out_dim": 41,  # this is set later
}

audio_v2 = {
    "aux_loss_layers": (),
    "num_enc_layers": 4,
    "num_text_dec_layers": 0,
    "num_audio_dec_layers": 2,
    "num_heads": 8,
    "model_dim": 512,
}

audio_v3 = {
    "aux_loss_layers": (),
    "num_enc_layers": 4,
    "num_text_dec_layers": 0,
    "num_audio_dec_layers": 2,
    "num_heads": 8,
    "model_dim": 512,
    "audio_decoder_prenet": "linear",
    "audio_decoder_postnet": "conv_res",
}

discrete_audio_v1 = {
    "aux_loss_layers": (),
    "num_enc_layers": 4,
    "num_dec_layers": 2,
    "num_heads": 8,
    "model_dim": 512,
}

text_v1 = {
    "aux_loss_layers": (),
    "num_enc_layers": 6,
    "num_text_dec_layers": 3,
    "num_audio_dec_layers": 0,
    "num_heads": 8,
    "model_dim": 512,
    # "text_out_dim": 41,  # this is set later
}
