# Zyang's GMM-Oracle ("Cheating") Segmentation & Clustering

This document summarizes how the **oracle ("cheating") segmentation** and its corresponding cluster IDs were built by **zyang**, and how they are used within this setup.

---

## 1. Motivation & Overview

In unsupervised ASR, segments are usually hypothesized without supervision (e.g., via learned convolutional boundary models). To establish a theoretical **upper-bound oracle baseline**, this setup uses ground-truth phonetic boundaries derived from an acoustic model forced alignment.

Because the segmentation is taken directly from the ground-truth forced alignment, each segment corresponds to **exactly one non-silence phoneme**, completely eliminating boundary and oversegmentation errors. This provides the performance ceiling for downstream segment-level modeling and cross-modal embedding evaluations.

- **Author / Source Setup**: `zyang` (`/u/zyang/setups/mini`)
- **Original Sisyphus Recipe**: `/u/zyang/setups/mini/recipe/i6_experiments/users/yang/experiments/generative_ctc/example_setups/librispeech/phmm/experiments/phmm_phon/gmm_oracle_segment_clustering.py`
- **Reference Documentation**: `/u/zyang/setups/mini/HANDOFF_GMM_ORACLE_SEGMENT_CLUSTERING.md`

---

## 2. Segmentation Construction (`nosil-gmm`)

1. **Alignment Source**:
   - Built on LibriSpeech 960h using GMM forced alignment references (`phoneme_ref_from_gmm_vad`, 20 ms frame rate).
2. **Segment Boundaries**:
   - Consecutive identical phoneme labels form a single segment (run-length encoding).
   - Silence frames/segments (label `0`) are explicitly removed (`nosil-gmm`).
3. **Scale**:
   - **32,666,275 phoneme segments** across **278,400 utterances** (split into 20 shards, `000` to `019`).
   - Every segment carries exactly **one** phoneme identity.

---

## 3. Acoustic Representations

1. **Underlying Frame Features**:
   - Fairseq wav2vec2-U representations, reduced via PCA to **512 dimensions** (`pca512_frame`, 20 ms frame shift, silence-removed).
   - Located at: `output/example_setups/librispeech/phmm_standalone_2024/ls960_fairseq_wav2vecu_frame_and_segment_reclustering/frame/pca512_frame/train_hdf/train.{001..020}.hdf`.
2. **Segment Mean Pooling**:
   - For each non-silence phoneme segment, all 512-dim frame vectors spanning the segment are averaged (mean-pooled) into a single 512-dim segment vector.
   - Job: `ComputeGmmAlignmentSegmentRepresentationsNanSafeJob`
   - Handles numerical stability: zeroes rare NaNs and clips corrupted values ($|x| > 10^6$) that could destabilize clustering.
   - Result: 20 shards of dense `[n_segments, 512]` float32 representations (`gmm_segment_reps.{000..019}.hdf`).

---

## 4. Faiss K-Means Clustering Pipeline

Clustering was performed using `create_faiss_segment_clustering_pipeline_parallel_sampling`:

1. **Sampling**:
   - A uniform random sample of **4,000,000 segment representations** was drawn across all 20 shards.
2. **K-Means Training**:
   - Trained on a 24 GB GPU using Faiss for $K \in \{128, 256, 512\}$ (and later extended up to $K=8192$).
   - Hyperparameters: 25 iterations (`niter=25`), 3 restarts (`nredo=3`).
   - Parameter `max_points_per_centroid = 4_000_000 // K + 1` ensured that all 4M sampled vectors were fully utilized.
   - Output centroids: `centers_k{K}.npy` ($[K, 512]$ float32).
3. **Corpus-Wide Assignment**:
   - All 32,666,275 segments were assigned to the nearest centroid on CPU.
4. **Post-Processing & Sanitization**:
   - **`SanitizeFaissClusterLabelsJob`**: Repaired rare CPU Faiss out-of-range cluster IDs into valid range $[0, K-1]$.
   - **`MakeSparseClusterHDFDatasetCompatibleJob`**: Formatted assignments into RETURNN-compatible sparse HDF files with 2-column `seqLengths`.

---

## 5. Deliverables & Directory Layout

Base directory on Curium:
`/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/ls960_gmm_oracle_segment_clustering`

- **Cluster Labels (RETURNN-compatible)**:
  `clustering/k{K}/compat/cluster_labels_k{K}.{000..019}.returnn_compat.hdf`
- **Sanitized Cluster Labels**:
  `clustering/k{K}/cluster_labels_k{K}.{000..019}.hdf`
- **Centroids**:
  `clustering/k{K}/centers_k{K}.npy`
- **Segment Phonemes (1:1 Ground Truth Target)**:
  `alias/.../segment_phonemes/shard_{000..019}/output/gmm_segment_phonemes.{000..019}.hdf`

### Quality Benchmark (Segment-Level vs. GMM Phoneme Reference)

| $K$ | Cluster Purity | Phone Purity | PNMI | Oracle Majority-Map PER |
|:---:|:---:|:---:|:---:|:---:|
| **128** | 0.703 | 0.285 | 0.711 | 0.297 |
| **256** | 0.780 | 0.190 | 0.771 | 0.220 |
| **512** | 0.822 | 0.115 | 0.816 | 0.178 |
| *1024* | *0.852* | - | *0.845* | *0.148* |
| *2048* | *0.876* | - | *0.867* | *0.124* |
| *4096* | *0.893* | - | *0.884* | *0.107* |
| *8192* | *0.903* | - | *0.896* | *0.097* |

---

## 6. Usage in `oyarzun` Recipes

1. **Path Resolution via `cluster_adapter.py`**:
   - `get_cheating_segmentation_path(k_clusters)` automatically points to:
     - On Curium: `/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/ls960_gmm_oracle_segment_clustering/clustering/k{k_clusters}/compat`
     - On RWTH-HPC: `/rwthfs/rz/cluster/hpcwork/bpd03090/cheating_segmentation/ls960_gmm_oracle_segment_clustering/...`
2. **Evaluation in `config_librispeech_960_wo_sil_cross_modal_cheat_seg.py`**:
   - Used by `CheatingSegmentationNearestNeighborEvalJob`:
     - Loads either the segment boundaries or pre-segmented cluster labels.
     - Averages representations within each oracle segment.
     - Performs zero-parameter nearest-neighbor classification against text phoneme embeddings using cosine similarity.
