# Joiner sparsity: live-model validation

Models are instantiated from their generated training configurations and checkpoints load strictly.
Counts are taken on effective weights captured at the actual joiner F.linear call in eval mode.

| Model | Layer | Raw exact zeros % | Saved-grid zeros % | Live eval zeros % | Different zero assignments |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP_baseline | 1 | 0.00000 | N/A | 0.00000 | 0 |
| FP_baseline | 2 | 0.00000 | N/A | 0.00000 | 0 |
| W4_from_scratch | 1 | 0.00000 | 86.67798 | 86.67798 | 0 |
| W4_from_scratch | 2 | 0.00000 | 74.27576 | 74.27576 | 0 |
| W8_from_scratch | 1 | 0.00000 | 6.31909 | 6.31909 | 0 |
| W8_from_scratch | 2 | 0.00000 | 2.66206 | 2.66206 | 0 |
| W4_finetuned_from_FP_25ep | 1 | 0.00000 | 76.97898 | 76.97898 | 0 |
| W4_finetuned_from_FP_25ep | 2 | 0.00000 | 47.33055 | 47.33055 | 0 |
| W4_finetuned_from_W8_25ep | 1 | 0.00000 | 72.99830 | 72.99830 | 0 |
| W4_finetuned_from_W8_25ep | 2 | 0.00000 | 57.03336 | 57.03336 | 0 |
| Full_context_W8A8_from_scratch | 1 | 0.00000 | 5.54606 | 5.54606 | 0 |
| Full_context_W8A8_from_scratch | 2 | 0.00000 | 2.73068 | 2.73068 | 0 |
| W8_finetuned_from_FP_25ep | 1 | 0.00000 | 4.11004 | 4.11004 | 0 |
| W8_finetuned_from_FP_25ep | 2 | 0.00000 | 2.67367 | 2.67367 | 0 |
