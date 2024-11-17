For some given setup (e.g. Conformer CTC on Librispeech),
when changing the hardware/environment or features of GPU,
some things naturally change,
and some hyperparameters need to be retuned.

Related:

- https://github.com/rwth-i6/returnn/wiki/Analysing-neural-networks
- https://github.com/google-research/tuning_playbook

Relevant aspects of the setup:

- Seq2seq architecture (CTC, AED, transducer, HMM, ...)
- Neural network architecture (Conformer, Transformer, LSTM, CNN, ...)
- Model size (number of layers, number of neurons)
- Loss function
- Optimizer
- Dataset

Relevant hardware/environment aspects:

- Num GPUs (and type of distributed training)
- Precision (float32/bfloat16) or mixed precision (float32+bfloat16)
- GPU memory size

Relevant hyperparameters:

- Batch size
- Num gradient accumulation steps
- Learning rate
- Num total steps / num epochs
- (Learning rate schedule maybe, but usually OCLR quite robust to any of these changes)
- (Gradient clipping?)
- Regularization: Weight decay (others maybe less important, unclear...)

Some observations:

- You want to use bfloat16, either via AMP or purely, when the hardware allows it.
- We use param averaging distributed training usually, because NVlink is not always available, or we do multi-node training, etc.
- Set batch size reasonably high to utilize the GPU. But for any changes, the other things might need to be adjusted, maybe even num epochs, so it's not directly clear what the optimal batch size is (w.r.t. best overall training time to reach good performance). When you don't see any speed improvements anymore, stop increasing it further. 
- Lower gradient accumulation is usually always better, as long as it still converges fine.
- Weight decay needs adjustment when changing the batch size (maybe also grad accum).
