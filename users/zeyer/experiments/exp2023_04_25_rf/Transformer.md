Decoder:

- factor on input embedding?
  - Our RETURNN TF recipe always used `embed_weight = self.EncValueTotalDim**0.5`.
  - ESPnet: `PositionalEncoding` scales the input via `xscale = sqrt(self.d_model)`.
  - NanoGPT: No sinusodial encodings but trainable, no scale, but trainable are initialized with same scale as emb.

- embedding layer-norm? (BLOOM-176B)

- ESPnet decoder: https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/decoder.py
- ESPnet example: https://github.com/espnet/espnet/blob/master/egs2/librispeech/asr1/conf/tuning/train_asr_conformer10_hop_length160.yaml
- ESPnet common settings:
  - (model dim) attention_dim=256
  - attention_heads: 8
  - linear_units: 2048
  - (num layers) num_blocks: 6
  - dropout_rate: 0.1
  - positional_dropout_rate: 0.1
  - self_attention_dropout_rate: 0.1
  - src_attention_dropout_rate: 0.1
  - pos_enc_class=PositionalEncoding
  - normalize_before=True

- nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
  - no bias for layernorm and linears
- 'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
- GPT: use GELU activation in FF
- GPT: block_size 1024 (relevant because learnable pos enc for all pos)
- GPT: learnable abs pos enc
- GPT: no bias in softmax
- GPT: shared params softmax + embedding
- GPT param init for FF out: init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
- GPT param init all other matrices: init.normal_(module.weight, mean=0.0, std=0.02)
- GPT param init bias: zeros

- use Flash attention

- Llama model: https://github.com/facebookresearch/llama/blob/main/llama/model.py
  - RMSNorm
  - Rotary pos embeddings
  - grouped KV, repeat KV
  - SILU activation
  - norm first
  - no bias in softmax

- sometimes sliding window attention (supported also by Flash attention)

- T5 style layer norm: No bias and no subtraction of mean

- Conformer uses Swish (Silu) activation in FF

Via https://huggingface.co/transformers/v3.5.1/_modules/transformers/modeling_t5.html:

    # Mesh TensorFlow attention initialization to avoid scaling before softmax
    # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
    d_model = self.config.d_model
    d_kv = self.config.d_kv  # (per head)
    n_heads = self.config.num_heads
    module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * d_kv) ** -0.5))
    module.k.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
    module.v.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
    module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * d_kv) ** -0.5))

- multi-query attention (Shazeer, 2019)
  - also used in PaLM 

- RoPE embeddings, or other positional embeddings
  - RoPE is used in PaLM

- no biases:
  - PaLM and many others...

- training recipe for large scale (huge amount of data) language modeling
  (via Lucas@Google, https://twitter.com/giffmana/status/1739754033194086733)
  AdamW with beta=(0.9,0.95), weight decay 0.1, no dropout, grad clip 1
  LR schedule: linear warmup, cos decay to 1e-5
  RMSNorm instead of LayerNorm
  no bias terms in Linear
  huge batch size? growing batch size?
  (example: Mamba paper on Transformer++: https://arxiv.org/pdf/2312.00752.pdf)

- μParam
- see recent Apple paper? (TODO which?)

- Query-Key Normalization
  - Query-Key Normalization for Transformers, https://aclanthology.org/2020.findings-emnlp.379.pdf
  - https://github.com/google/flax/commit/344529669ce29561f21c518ea63b9e032ce99397
  - https://twitter.com/PiotrPadlewski/status/1690686243296739328
  - Small-scale proxies for large-scale Transformer training instabilities, https://arxiv.org/abs/2309.14322
  - Scaling Vision Transformers to 22 Billion Parameters, https://arxiv.org/abs/2302.05442
  - https://twitter.com/_basilM/status/1625185484837208082
  - https://twitter.com/m__dehghani/status/1625192439999827971
  - https://discuss.huggingface.co/t/how-to-implement-key-query-layer-normalized-transformers-llms-in-huggingface/31997
  - Transformers without Tears: Improving the Normalization of Self-Attention, https://arxiv.org/abs/1910.05895

- From PaLM:
  > We use a dynamic weight decay of lr^2.0 during training, where lr is the current learning rate.
- From PaLM:
  > Because the input and output embedding layers are shared, we scale the pre-softmax output logits by 1/√n,
  > where n is the embedding size.

- Optimizer, Adam variants:
  - PaLM: Adafactor, scales the learning rate by the root-mean-square of the parameter matrix,
    > parameter scaling has the benefit that parameter matrices which operate at different scales
    > (the embeddings and layer norm scales) do not have their learning rate scaled down at the same rate.
  - Check ZipFormer

- Check ZipFormer for many things...

- Adaptive layer norm (adaLN) (e.g. as in https://arxiv.org/pdf/2212.09748.pdf)
