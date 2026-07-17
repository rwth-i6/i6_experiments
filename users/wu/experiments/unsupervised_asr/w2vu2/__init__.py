"""SAE §1c — wav2vec-U 2.0 GAN (arXiv:2204.02492) over frozen BEST-RQ features.

The GAN itself is **not reimplemented**: we run fairseq's reference implementation
(`fairseq/examples/wav2vec/unsupervised`, shipped inside the fairseq wheel) from the dedicated
`w2vu` conda env. This package only produces the data fairseq expects and wraps its trainer as
sisyphus jobs, so every published quirk of the reference (gradient-penalty norm over the time axis,
perplexity-space diversity, flipped BCE labels, ...) is inherited rather than re-derived.

Deviations forced by our encoder, all logged in SAE_1c.md:
  BEST-RQ is 512-d @ 25 Hz; wav2vec2-Large is 1024-d @ 50 Hz. The paper's load-bearing constraint is
  the *generator output* rate (~10 Hz ground-truth phone rate; 25-28 Hz provably diverges), so
  stride 3@50Hz -> 16.7 Hz becomes stride 2@25Hz -> 12.5 Hz, and kernel 9 (180 ms) becomes kernel 5
  (200 ms). The MFCC aux target subsamples by 4 (100 Hz -> 25 Hz) instead of 2.
"""
