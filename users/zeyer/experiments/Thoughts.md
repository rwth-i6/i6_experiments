# Model interface

Define simple model interface
input -> output,
which can be used across tasks
and architectures.

Types of architectures:
- Attention-based encoder-decoder
- Transducer
- Hybrid HMM

Tasks:
- ASR
- MT
- LM

Model interface needs:
- Support for both sparse or dense input, shape [B,T,D].
- in_dim (sparse or dense feature) and time dim could be given explicitly.
- dense output; or directly label logits? any label topo?

We partly have this interface already in returnn_common.nn.{encoder,decoder}.
This should be finished and then just used here.
