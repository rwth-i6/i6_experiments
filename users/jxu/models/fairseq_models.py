xlsr_base = {
    '_name': 'wav2vec2',
    'quantize_targets': True,
    'extractor_mode': 'layer_norm',
    'layer_norm_first': True,
    'encoder_layers': 12,
    'encoder_embed_dim': 768,
    'encoder_ffn_embed_dim': 3072,
    'encoder_attention_heads': 12,
    'final_dim': 256,
    'conv_feature_layers': '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]',
    'conv_bias': True,
    'feature_grad_mult': 0.1,
    'encoder_layerdrop': 0.05,
    'dropout_features': 0.1,
    'dropout_input': 0.1
}

xlsr_large = {
    '_name': 'wav2vec2',
    'quantize_targets': True,
    'extractor_mode': 'layer_norm',
    'layer_norm_first': True,
    'encoder_layers': 24,
    'encoder_embed_dim': 1024,
    'encoder_ffn_embed_dim': 4096,
    'encoder_attention_heads': 16,
    'final_dim': 768,
    'conv_feature_layers': '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512, 2, 2]',
    'conv_bias': True,
    'latent_temp': [2.0, 0.1, 0.999995],
    'feature_grad_mult': 1.0,
}
