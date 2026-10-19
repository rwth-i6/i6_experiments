"""
Evaluation and Analysis Pipeline for Self-Supervised SpeechT5 Base (Zero-Shot / Un-Finetuned ASR).

Evaluates the self-supervised SpeechT5 Base model (fairseq speecht5_base.pt converted to HF)
on LibriSpeech without any fine-tuning for ASR:
1. Direct ASR decoding on LibriSpeech evaluation splits (WER, CER, detailed error analysis).
2. Decoder Cross-Attention analysis & heatmap visualizations across layers and heads.
3. Shared-Encoder state PCA visualization (Audio frames vs. Text LM frames) with pairwise
   cosine similarity and L2 distance metrics.
"""

from typing import List, Dict, Optional, Tuple, Iterator, Union, Any
import os
import sys
import glob
import json
import logging
import math
import numpy as np

import soundfile as sf
import torch

from sisyphus import tk, Task

logger = logging.getLogger(__name__)


def compute_levenshtein(ref_tokens: List[str], hyp_tokens: List[str]) -> Tuple[int, int, int, int]:
    """
    Computes Levenshtein edit distance between reference and hypothesis tokens.
    Returns: (substitutions, deletions, insertions, total_reference_tokens)
    """
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,        # Deletion
                dp[i, j - 1] + 1,        # Insertion
                dp[i - 1, j - 1] + cost   # Match / Substitution
            )

    # Backtrack to count operations
    i, j = m, n
    subs = dels = inss = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            if dp[i, j] == dp[i - 1, j - 1] + cost:
                if cost == 1:
                    subs += 1
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            dels += 1
            i -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            inss += 1
            j -= 1
        else:
            break

    return subs, dels, inss, m


def load_phoneme_lexicon(lexicon_path: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Loads word-to-phonemes dictionary for PER evaluation.
    Tries CMUDict (via container_libs / nltk) or custom lexicon file.
    """
    lex = {}
    if lexicon_path and os.path.exists(lexicon_path):
        try:
            with open(lexicon_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        word = parts[0].lower()
                        phones = [p.rstrip("012") for p in parts[1:]]
                        if word not in lex:
                            lex[word] = phones
            logger.info(f"Loaded {len(lex)} lexicon entries from {lexicon_path}")
            return lex
        except Exception as e:
            logger.warning(f"Failed to load custom lexicon from {lexicon_path}: {e}")

    # Fallback to CMUDict from nltk / container_libs
    container_libs = "/rwthfs/rz/cluster/home/p0023999/experiments/2026_05_07_first_experiments/container_libs"
    if container_libs not in sys.path and os.path.exists(container_libs):
        sys.path.insert(0, container_libs)
    try:
        import nltk
        from nltk.corpus import cmudict
        cmu = cmudict.dict()
        for w, prons in cmu.items():
            lex[w.lower()] = [p.rstrip("012") for p in prons[0]]
        logger.info(f"Loaded {len(lex)} CMUDict lexicon entries")
    except Exception as e:
        logger.warning(f"Could not load CMUDict: {e}")
    return lex


def words_to_phonemes(text: str, lexicon: Dict[str, List[str]]) -> List[str]:
    """Convert text words into phoneme tokens using lexicon, with character fallback."""
    words = text.lower().split()
    phones = []
    for w in words:
        clean_w = "".join(c for c in w if c.isalnum() or c == "'")
        if clean_w in lexicon:
            phones.extend(lexicon[clean_w])
        elif w in lexicon:
            phones.extend(lexicon[w])
        else:
            # character fallback for out-of-vocabulary words
            phones.extend(list(clean_w))
    return phones


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def _avg_pairwise_cosine_similarity(x: np.ndarray, y: Optional[np.ndarray] = None) -> float:
    """Average pairwise cosine similarity in the original feature space."""
    x_norm = _l2_normalize(x)
    if y is None:
        n = x_norm.shape[0]
        if n < 2:
            return float("nan")
        sims = x_norm @ x_norm.T
        off_diag_sum = float(sims.sum() - np.trace(sims))
        return off_diag_sum / (n * (n - 1))
    y_norm = _l2_normalize(y)
    if x_norm.shape[0] == 0 or y_norm.shape[0] == 0:
        return float("nan")
    return float((x_norm @ y_norm.T).mean())


def _avg_pairwise_l2_distance(x: np.ndarray, y: Optional[np.ndarray] = None) -> float:
    """Average pairwise L2 distance in the original feature space."""
    x = x.astype(np.float64)
    if y is None:
        n = x.shape[0]
        if n < 2:
            return float("nan")
        x2 = np.sum(x**2, axis=1)
        dists = np.sqrt(np.clip(x2[:, None] + x2[None, :] - 2 * (x @ x.T), 0, None))
        off_diag_sum = float(dists.sum())
        return off_diag_sum / (n * (n - 1))
    y = y.astype(np.float64)
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan")
    x2 = np.sum(x**2, axis=1)
    y2 = np.sum(y**2, axis=1)
    dists = np.sqrt(np.clip(x2[:, None] + y2[None, :] - 2 * (x @ y.T), 0, None))
    return float(dists.mean())


def _fit_pca_2d(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit 2D PCA returning (mean [F], components [F, 2])."""
    samples = samples.astype(np.float32)
    mean = samples.mean(axis=0, keepdims=True)
    centered = samples - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    return mean[0], components


def _project(samples: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (samples.astype(np.float32) - mean[None]) @ components


def load_librispeech_dataset(dataset_dir: str, max_seqs: Optional[int] = None) -> List[Dict[str, Any]]:
    """Scan LibriSpeech directory for audio and trans.txt transcripts."""
    entries = []
    trans_files = sorted(glob.glob(os.path.join(dataset_dir, "**", "*.trans.txt"), recursive=True))
    for tf in trans_files:
        chapter_dir = os.path.dirname(tf)
        with open(tf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                seq_id = parts[0]
                ref_text = parts[1] if len(parts) > 1 else ""
                flac_path = os.path.join(chapter_dir, f"{seq_id}.flac")
                if os.path.exists(flac_path):
                    entries.append({
                        "seq_id": seq_id,
                        "flac_path": flac_path,
                        "reference": ref_text,
                    })
                    if max_seqs is not None and len(entries) >= max_seqs:
                        return entries
    return entries


def evaluate_speecht5_base(
    model_dir: str,
    dataset_dir: str,
    text_lm_path: Optional[str] = None,
    phoneme_lexicon_path: Optional[str] = None,
    output_dir: str = "./speecht5_eval_output",
    max_seqs: Optional[int] = None,
    max_cross_att_seqs: int = 10,
    max_pca_seqs: int = 50,
    max_pca_points: int = 10_000,
    device: Optional[str] = None,
    model_title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run evaluation and analyses for SpeechT5 (both Base self-supervised and fine-tuned ASR).
    Computes WER, CER, and PER (via pronunciation lexicon), cross-attention heatmaps,
    and PCA representation analyses with pretraining codebook visualizations.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import SpeechT5Config, SpeechT5Processor, SpeechT5ForSpeechToText
    import transformers.models.speecht5.modeling_speecht5 as st5

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_title is None:
        if "asr" in model_dir.lower() or "finetun" in model_dir.lower():
            model_title = "SpeechT5 Fine-Tuned ASR"
        else:
            model_title = "SpeechT5 Base (Self-Supervised)"

    os.makedirs(output_dir, exist_ok=True)
    cross_att_dir = os.path.join(output_dir, "cross_attention")
    pca_dir = os.path.join(output_dir, "pca_analysis")
    os.makedirs(cross_att_dir, exist_ok=True)
    os.makedirs(pca_dir, exist_ok=True)

    print(f"Loading {model_title} from: {model_dir}")
    processor = SpeechT5Processor.from_pretrained(model_dir)
    model = SpeechT5ForSpeechToText.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Load phoneme lexicon for PER calculation
    phoneme_lexicon = load_phoneme_lexicon(phoneme_lexicon_path)
    if phoneme_lexicon:
        print(f"Phoneme lexicon loaded with {len(phoneme_lexicon)} words for PER evaluation.")
    else:
        print("Warning: No phoneme lexicon available; PER will not be computed.")

    # Load Text Encoder with pretrained/fine-tuned weights for PCA analysis
    text_encoder = None
    text_prenet_path = os.path.join(model_dir, "text_encoder_prenet.pt")
    if os.path.exists(text_prenet_path):
        print(f"Loading Text Encoder weights from: {text_prenet_path}")
        config = SpeechT5Config.from_pretrained(model_dir)
        text_encoder = st5.SpeechT5EncoderWithTextPrenet(config)
        text_encoder.wrapped_encoder.load_state_dict(model.speecht5.encoder.wrapped_encoder.state_dict())
        prenet_weights = torch.load(text_prenet_path, map_location="cpu", weights_only=False)
        text_encoder.prenet.embed_tokens.weight.data = prenet_weights["embed_tokens.weight"]
        if "alpha" in prenet_weights:
            text_encoder.prenet.encode_positions.alpha.data = prenet_weights["alpha"]
        text_encoder.to(device)
        text_encoder.eval()
    elif hasattr(model.speecht5, "decoder") and hasattr(model.speecht5.decoder, "prenet") and hasattr(model.speecht5.decoder.prenet, "embed_tokens"):
        print("Using model decoder token embeddings for Text Encoder prenet...")
        config = SpeechT5Config.from_pretrained(model_dir)
        text_encoder = st5.SpeechT5EncoderWithTextPrenet(config)
        text_encoder.wrapped_encoder.load_state_dict(model.speecht5.encoder.wrapped_encoder.state_dict())
        text_encoder.prenet.embed_tokens.weight.data = model.speecht5.decoder.prenet.embed_tokens.weight.data.clone()
        text_encoder.to(device)
        text_encoder.eval()

    # Load dataset
    print(f"Scanning LibriSpeech dataset in: {dataset_dir}")
    dataset_entries = load_librispeech_dataset(dataset_dir, max_seqs=max_seqs)
    total_seqs = len(dataset_entries)
    print(f"Found {total_seqs} audio utterances to evaluate.")

    hypotheses = {}
    references = {}

    tot_sub_w = tot_del_w = tot_ins_w = tot_ref_w = 0
    tot_sub_c = tot_del_c = tot_ins_c = tot_ref_c = 0
    tot_sub_p = tot_del_p = tot_ins_p = tot_ref_p = 0

    audio_encoder_states_list = []
    cross_att_count = 0

    hyp_file = open(os.path.join(output_dir, "hypotheses.txt"), "w", encoding="utf-8")
    ref_file = open(os.path.join(output_dir, "reference.txt"), "w", encoding="utf-8")

    for idx, entry in enumerate(dataset_entries):
        seq_id = entry["seq_id"]
        flac_path = entry["flac_path"]
        ref_text = entry["reference"].upper()

        audio, sr = sf.read(flac_path)
        inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt").to(device)

        with torch.no_grad():
            # ASR generation
            if cross_att_count < max_cross_att_seqs:
                gen_out = model.generate(
                    inputs.input_values,
                    max_length=450,
                    return_dict_in_generate=True,
                    output_attentions=True,
                )
                pred_tokens = gen_out.sequences
            else:
                pred_tokens = model.generate(inputs.input_values, max_length=450)

            hyp_text = processor.batch_decode(pred_tokens, skip_special_tokens=True)[0].strip().upper()

            # Shared-encoder forward pass for PCA collection
            if len(audio_encoder_states_list) < max_pca_seqs:
                enc_out = model.speecht5.encoder(inputs.input_values)
                audio_enc = enc_out.last_hidden_state.squeeze(0).cpu().numpy()  # [T_frames, D]
                audio_encoder_states_list.append(audio_enc)

            # Cross-attention analysis for the first max_cross_att_seqs utterances
            if cross_att_count < max_cross_att_seqs:
                cross_att_count += 1
                # Forward with generated decoder input ids to capture full cross-attention map
                fwd = model(
                    input_values=inputs.input_values,
                    decoder_input_ids=pred_tokens,
                    output_attentions=True,
                )
                # fwd.cross_attentions is tuple of length num_layers, each [1, num_heads, tgt_len, src_len]
                if fwd.cross_attentions is not None and len(fwd.cross_attentions) > 0:
                    att_layers = [ca.squeeze(0).cpu().numpy() for ca in fwd.cross_attentions]  # each [H, L, T]
                    att_np = np.stack(att_layers, axis=0)  # [num_layers, num_heads, tgt_len, src_len]
                    
                    # Save npz
                    npz_path = os.path.join(cross_att_dir, f"{seq_id}.npz")
                    np.savez_compressed(
                        npz_path,
                        seq_id=seq_id,
                        att_weights=att_np,
                        hyp_text=hyp_text,
                        ref_text=ref_text,
                    )

                    # Plot head-averaged attention across all layers
                    num_layers = att_np.shape[0]
                    fig, axes = plt.subplots(1, num_layers, figsize=(4 * num_layers, 4), sharey=True)
                    if num_layers == 1:
                        axes = [axes]
                    for l_idx in range(num_layers):
                        head_avg = att_np[l_idx].mean(axis=0)  # [tgt_len, src_len]
                        im = axes[l_idx].imshow(head_avg, aspect="auto", origin="lower", cmap="viridis")
                        axes[l_idx].set_title(f"Layer {l_idx}")
                        axes[l_idx].set_xlabel("Encoder Audio Frames")
                        if l_idx == 0:
                            axes[l_idx].set_ylabel("Generated Tokens")
                    fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
                    fig.suptitle(f"{seq_id} Cross-Attention (Head Avg)\nHyp: {hyp_text[:40]}...", fontsize=10)
                    plt.savefig(os.path.join(cross_att_dir, f"{seq_id}_head-avg.png"), dpi=120, bbox_inches="tight")
                    plt.close(fig)

        # WER calculation
        ref_words = ref_text.split()
        hyp_words = hyp_text.split()
        sub_w, del_w, ins_w, n_ref_w = compute_levenshtein(ref_words, hyp_words)
        tot_sub_w += sub_w
        tot_del_w += del_w
        tot_ins_w += ins_w
        tot_ref_w += n_ref_w

        # CER calculation
        ref_chars = list(ref_text)
        hyp_chars = list(hyp_text)
        sub_c, del_c, ins_c, n_ref_c = compute_levenshtein(ref_chars, hyp_chars)
        tot_sub_c += sub_c
        tot_del_c += del_c
        tot_ins_c += ins_c
        tot_ref_c += n_ref_c

        # PER calculation
        if phoneme_lexicon:
            ref_phones = words_to_phonemes(ref_text, phoneme_lexicon)
            hyp_phones = words_to_phonemes(hyp_text, phoneme_lexicon)
            sub_p, del_p, ins_p, n_ref_p = compute_levenshtein(ref_phones, hyp_phones)
            tot_sub_p += sub_p
            tot_del_p += del_p
            tot_ins_p += ins_p
            tot_ref_p += n_ref_p

        hypotheses[seq_id] = hyp_text
        references[seq_id] = ref_text

        hyp_file.write(f"{seq_id} {hyp_text}\n")
        ref_file.write(f"{seq_id} {ref_text}\n")
        hyp_file.flush()
        ref_file.flush()

        if (idx + 1) % 10 == 0 or idx == total_seqs - 1:
            current_wer = (tot_sub_w + tot_del_w + tot_ins_w) / max(1, tot_ref_w) * 100.0
            per_info = f" | PER: {(tot_sub_p + tot_del_p + tot_ins_p)/max(1, tot_ref_p)*100.0:.2f}%" if phoneme_lexicon else ""
            print(f"[{idx+1}/{total_seqs}] Current WER: {current_wer:.2f}%{per_info} | Sample Hyp: '{hyp_text[:50]}'")

    hyp_file.close()
    ref_file.close()

    final_wer = (tot_sub_w + tot_del_w + tot_ins_w) / max(1, tot_ref_w) * 100.0
    final_cer = (tot_sub_c + tot_del_c + tot_ins_c) / max(1, tot_ref_c) * 100.0
    final_per = (tot_sub_p + tot_del_p + tot_ins_p) / max(1, tot_ref_p) * 100.0 if phoneme_lexicon else None

    asr_summary = {
        "num_sequences": total_seqs,
        "wer": final_wer,
        "cer": final_cer,
        "per": final_per,
        "word_errors": {
            "substitutions": int(tot_sub_w),
            "deletions": int(tot_del_w),
            "insertions": int(tot_ins_w),
            "total_ref_words": int(tot_ref_w),
        },
        "char_errors": {
            "substitutions": int(tot_sub_c),
            "deletions": int(tot_del_c),
            "insertions": int(tot_ins_c),
            "total_ref_chars": int(tot_ref_c),
        },
        "phoneme_errors": {
            "substitutions": int(tot_sub_p),
            "deletions": int(tot_del_p),
            "insertions": int(tot_ins_p),
            "total_ref_phones": int(tot_ref_p),
        } if phoneme_lexicon else None,
    }

    with open(os.path.join(output_dir, "asr_summary.json"), "w", encoding="utf-8") as f:
        json.dump(asr_summary, f, indent=2)

    with open(os.path.join(output_dir, "asr_report.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"{model_title.upper()} ASR REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Evaluated Sequences : {total_seqs}\n")
        f.write(f"Word Error Rate     : {final_wer:.2f}%\n")
        f.write(f"  - Substitutions   : {tot_sub_w} ({tot_sub_w/max(1,tot_ref_w)*100:.2f}%)\n")
        f.write(f"  - Deletions       : {tot_del_w} ({tot_del_w/max(1,tot_ref_w)*100:.2f}%)\n")
        f.write(f"  - Insertions      : {tot_ins_w} ({tot_ins_w/max(1,tot_ref_w)*100:.2f}%)\n")
        f.write(f"  - Reference Words : {tot_ref_w}\n\n")
        if final_per is not None:
            f.write(f"Phoneme Error Rate  : {final_per:.2f}%\n")
            f.write(f"  - Substitutions   : {tot_sub_p} ({tot_sub_p/max(1,tot_ref_p)*100:.2f}%)\n")
            f.write(f"  - Deletions       : {tot_del_p} ({tot_del_p/max(1,tot_ref_p)*100:.2f}%)\n")
            f.write(f"  - Insertions      : {tot_ins_p} ({tot_ins_p/max(1,tot_ref_p)*100:.2f}%)\n")
            f.write(f"  - Reference Phones: {tot_ref_p}\n\n")
        f.write(f"Character Error Rate: {final_cer:.2f}%\n")
        f.write(f"  - Substitutions   : {tot_sub_c} ({tot_sub_c/max(1,tot_ref_c)*100:.2f}%)\n")
        f.write(f"  - Deletions       : {tot_del_c} ({tot_del_c/max(1,tot_ref_c)*100:.2f}%)\n")
        f.write(f"  - Insertions      : {tot_ins_c} ({tot_ins_c/max(1,tot_ref_c)*100:.2f}%)\n")
        f.write(f"  - Reference Chars : {tot_ref_c}\n")
        f.write("=" * 60 + "\n\n")
        f.write("SAMPLE HYPOTHESES:\n")
        for k in list(hypotheses.keys())[:10]:
            f.write(f"[{k}]\n")
            f.write(f"  REF: {references[k]}\n")
            f.write(f"  HYP: {hypotheses[k]}\n")

    # =========================================================================
    # PCA VISUALIZATION & CODEBOOK ANALYSIS (Audio vs. Text LM vs. Codebook)
    # =========================================================================
    pca_summary = {}
    if text_encoder is not None and len(audio_encoder_states_list) > 0:
        print(f"Running PCA Analysis of Shared-Encoder states for {model_title} (Audio vs. Text LM)...")
        # Load sample LM text lines
        lm_texts = []
        if text_lm_path and os.path.exists(text_lm_path):
            with open(text_lm_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 10:
                        lm_texts.append(line)
                        if len(lm_texts) >= max_pca_seqs:
                            break
        if not lm_texts:
            # Fallback to reference texts
            lm_texts = [entry["reference"] for entry in dataset_entries[:max_pca_seqs]]

        text_encoder_states_list = []
        with torch.no_grad():
            for text_line in lm_texts:
                toks = processor.tokenizer(text_line, truncation=True, max_length=400, return_tensors="pt").to(device)
                text_out = text_encoder(input_values=toks.input_ids)
                text_states_list = text_out.last_hidden_state.squeeze(0).cpu().numpy()  # [T_text, D]
                text_encoder_states_list.append(text_states_list)

        audio_pooled = np.concatenate(audio_encoder_states_list, axis=0)
        text_pooled = np.concatenate(text_encoder_states_list, axis=0)

        if audio_pooled.shape[0] > max_pca_points:
            subsample_idx = np.random.choice(audio_pooled.shape[0], max_pca_points, replace=False)
            audio_pooled_sub = audio_pooled[subsample_idx]
        else:
            audio_pooled_sub = audio_pooled

        if text_pooled.shape[0] > max_pca_points:
            subsample_idx = np.random.choice(text_pooled.shape[0], max_pca_points, replace=False)
            text_pooled_sub = text_pooled[subsample_idx]
        else:
            text_pooled_sub = text_pooled

        # Pairwise cosine similarity in original continuous feature space
        cos_audio_audio = _avg_pairwise_cosine_similarity(audio_pooled_sub)
        cos_text_text = _avg_pairwise_cosine_similarity(text_pooled_sub)
        cos_cross = _avg_pairwise_cosine_similarity(audio_pooled_sub, text_pooled_sub)

        # Pairwise L2 distances
        l2_audio_audio = _avg_pairwise_l2_distance(audio_pooled_sub)
        l2_text_text = _avg_pairwise_l2_distance(text_pooled_sub)
        l2_cross = _avg_pairwise_l2_distance(audio_pooled_sub, text_pooled_sub)

        # Fit 2D PCA on pooled representations
        pooled_all = np.concatenate([audio_pooled_sub, text_pooled_sub], axis=0)
        mean, components = _fit_pca_2d(pooled_all)

        audio_proj = _project(audio_pooled_sub, mean, components)
        text_proj = _project(text_pooled_sub, mean, components)

        # ---------------------------------------------------------------------
        # Vector Quantization & Codebook Analysis
        # ---------------------------------------------------------------------
        quant_path = os.path.join(model_dir, "quantizer.pt")
        if not os.path.exists(quant_path):
            quant_path = "/rwthfs/rz/cluster/hpcwork/bpd03090/speecht5_base_converted/quantizer.pt"

        quant_loaded = False
        codebook_proj = audio_quant_proj = text_quant_proj = None
        quant_metrics = {}

        if os.path.exists(quant_path):
            try:
                quant_data = torch.load(quant_path, map_location="cpu", weights_only=False)
                # quant_data contains: 'vars' [1, 200, 384], 'weight_proj.weight' [200, 768], 'weight_proj.bias' [200]
                q_vars = quant_data["vars"].squeeze(0)  # [200, 384]
                q_codebook = q_vars.view(2, 100, 384)
                q_w = quant_data["weight_proj.weight"]
                q_b = quant_data["weight_proj.bias"]
                quant_loaded = True
                print(f"Loaded SpeechT5 Pretraining Vector Quantizer from: {quant_path}")

                def quantize_states(states_np):
                    h = torch.from_numpy(states_np).float()
                    logits = torch.nn.functional.linear(h, q_w, q_b).view(-1, 2, 100)
                    idx = logits.argmax(dim=-1)  # [N, 2]
                    q0 = q_codebook[0, idx[:, 0]]
                    q1 = q_codebook[1, idx[:, 1]]
                    q = torch.cat([q0, q1], dim=-1).numpy()
                    return q, idx.numpy()

                audio_quant_sub, audio_code_idx = quantize_states(audio_pooled_sub)
                text_quant_sub, text_code_idx = quantize_states(text_pooled_sub)

                # Codebook Centroid Vectors in 768-d:
                # Group 0 entries concatenated with mean of Group 1
                # Group 1 entries concatenated with mean of Group 0
                mean_g1 = q_codebook[1].mean(dim=0, keepdim=True)
                mean_g0 = q_codebook[0].mean(dim=0, keepdim=True)
                g0_centroids = torch.cat([q_codebook[0], mean_g1.expand(100, -1)], dim=-1).numpy()
                g1_centroids = torch.cat([mean_g0.expand(100, -1), q_codebook[1]], dim=-1).numpy()
                centroid_codes = np.concatenate([g0_centroids, g1_centroids], axis=0)  # [200, 768]

                # Active discrete codes instantiated by audio & text
                active_codes = np.unique(np.concatenate([audio_quant_sub, text_quant_sub], axis=0), axis=0)
                codebook_all_ref = np.unique(np.concatenate([active_codes, centroid_codes], axis=0), axis=0)

                # Project to 2D PCA space
                codebook_proj = _project(codebook_all_ref, mean, components)
                audio_quant_proj = _project(audio_quant_sub, mean, components)
                text_quant_proj = _project(text_quant_sub, mean, components)

                # Compute quantizer & codebook metrics
                unique_audio_codes = int(len(np.unique(audio_code_idx, axis=0)))
                unique_text_codes = int(len(np.unique(text_code_idx, axis=0)))
                audio_code_set = set(map(tuple, audio_code_idx))
                text_code_set = set(map(tuple, text_code_idx))
                shared_codes = int(len(audio_code_set.intersection(text_code_set)))

                cos_audio_quant = _avg_pairwise_cosine_similarity(audio_quant_sub)
                cos_text_quant = _avg_pairwise_cosine_similarity(text_quant_sub)
                cos_cross_quant = _avg_pairwise_cosine_similarity(audio_quant_sub, text_quant_sub)

                # Reconstruction fidelity between continuous and quantized representations
                dot_a = np.sum(audio_pooled_sub * audio_quant_sub, axis=-1)
                norm_a = np.linalg.norm(audio_pooled_sub, axis=-1) * np.linalg.norm(audio_quant_sub, axis=-1) + 1e-9
                fidelity_audio = float(np.mean(dot_a / norm_a))

                dot_t = np.sum(text_pooled_sub * text_quant_sub, axis=-1)
                norm_t = np.linalg.norm(text_pooled_sub, axis=-1) * np.linalg.norm(text_quant_sub, axis=-1) + 1e-9
                fidelity_text = float(np.mean(dot_t / norm_t))

                quant_metrics = {
                    "unique_audio_codebook_entries": unique_audio_codes,
                    "unique_text_codebook_entries": unique_text_codes,
                    "shared_codebook_entries": shared_codes,
                    "within_quantized_audio_cosine": cos_audio_quant,
                    "within_quantized_text_cosine": cos_text_quant,
                    "cross_modal_quantized_cosine": cos_cross_quant,
                    "audio_quantization_fidelity": fidelity_audio,
                    "text_quantization_fidelity": fidelity_text,
                }
            except Exception as e:
                print(f"Warning: Failed to perform vector quantization analysis: {e}")

        # ---------------------------------------------------------------------
        # GRAPH 1: Continuous Audio vs. Text LM States (+ Codebook Centroids)
        # ---------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(audio_proj[:, 0], audio_proj[:, 1], alpha=0.25, s=10, label="Audio Encoder States", color="#1f77b4")
        ax.scatter(text_proj[:, 0], text_proj[:, 1], alpha=0.25, s=10, label="Text LM Encoder States", color="#ff7f0e")
        if quant_loaded and codebook_proj is not None:
            ax.scatter(codebook_proj[:, 0], codebook_proj[:, 1], alpha=0.85, s=36, marker="D", edgecolor="#222222", linewidths=0.5, label="Pretraining Codebook Codes", color="#9467bd")
        ax.set_title(f"{model_title} Shared-Encoder State PCA\nAudio (LibriSpeech) vs. Text (LM) with Pretraining Codebook")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend(frameon=True, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(pca_dir, "audio_vs_text_pca.png"), dpi=200)
        plt.close(fig)

        # ---------------------------------------------------------------------
        # GRAPH 2: Quantized Audio & Text Representations vs. Codebook Codes
        # ---------------------------------------------------------------------
        if quant_loaded and audio_quant_proj is not None:
            fig, ax = plt.subplots(figsize=(9, 7))
            ax.scatter(audio_quant_proj[:, 0], audio_quant_proj[:, 1], alpha=0.35, s=14, label="Quantized Audio Representations", color="#00aaff")
            ax.scatter(text_quant_proj[:, 0], text_quant_proj[:, 1], alpha=0.35, s=14, label="Quantized Text LM Representations", color="#ff5555")
            ax.scatter(codebook_proj[:, 0], codebook_proj[:, 1], alpha=0.9, s=40, marker="D", edgecolor="#222222", linewidths=0.5, label="Codebook Centroids (Pretraining)", color="#9467bd")
            ax.set_title(f"{model_title} Quantized Representations & Codebook\nAudio vs. Text Quantized States in Shared Space")
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.legend(frameon=True, loc="upper right")
            ax.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(pca_dir, "quantized_representations_pca.png"), dpi=200)
            plt.close(fig)

        pca_summary = {
            "model_title": model_title,
            "num_audio_points": int(audio_pooled_sub.shape[0]),
            "num_text_points": int(text_pooled_sub.shape[0]),
            "cosine_similarity": {
                "within_audio": cos_audio_audio,
                "within_text": cos_text_text,
                "cross_modal_audio_text": cos_cross,
            },
            "l2_distance": {
                "within_audio": l2_audio_audio,
                "within_text": l2_text_text,
                "cross_modal_audio_text": l2_cross,
            },
            "codebook_quantization": quant_metrics if quant_loaded else None,
        }

        with open(os.path.join(pca_dir, "pca_summary.json"), "w", encoding="utf-8") as f:
            json.dump(pca_summary, f, indent=2)

        with open(os.path.join(pca_dir, "cosine_similarities.txt"), "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"{model_title.upper()} SHARED-ENCODER REPRESENTATION SIMILARITIES\n")
            f.write("=" * 60 + "\n")
            f.write(f"Within Audio Cosine Similarity     : {cos_audio_audio:.4f}\n")
            f.write(f"Within Text LM Cosine Similarity   : {cos_text_text:.4f}\n")
            f.write(f"Cross-Modal (Audio-Text) Similarity: {cos_cross:.4f}\n\n")
            f.write(f"Within Audio L2 Distance           : {l2_audio_audio:.4f}\n")
            f.write(f"Within Text LM L2 Distance         : {l2_text_text:.4f}\n")
            f.write(f"Cross-Modal (Audio-Text) L2 Dist   : {l2_cross:.4f}\n")
            if quant_loaded:
                f.write("-" * 60 + "\n")
                f.write("VECTOR QUANTIZATION & CODEBOOK METRICS\n")
                f.write("-" * 60 + "\n")
                f.write(f"Unique Audio Codebook Entries Used : {quant_metrics.get('unique_audio_codebook_entries', 0)}\n")
                f.write(f"Unique Text Codebook Entries Used  : {quant_metrics.get('unique_text_codebook_entries', 0)}\n")
                f.write(f"Shared Codebook Entries (A ∩ T)    : {quant_metrics.get('shared_codebook_entries', 0)}\n")
                f.write(f"Within Quantized Audio Cosine      : {quant_metrics.get('within_quantized_audio_cosine', 0):.4f}\n")
                f.write(f"Within Quantized Text Cosine       : {quant_metrics.get('within_quantized_text_cosine', 0):.4f}\n")
                f.write(f"Cross-Modal Quantized Cosine       : {quant_metrics.get('cross_modal_quantized_cosine', 0):.4f}\n")
                f.write(f"Audio Quantization Fidelity        : {quant_metrics.get('audio_quantization_fidelity', 0):.4f}\n")
                f.write(f"Text Quantization Fidelity         : {quant_metrics.get('text_quantization_fidelity', 0):.4f}\n")
            f.write("=" * 60 + "\n")

    print(f"{model_title} evaluation and analysis completed successfully!")
    return {
        "asr": asr_summary,
        "pca": pca_summary,
    }


class SpeechT5BaseEvalJob(tk.Job):
    """
    Sisyphus Job to evaluate SpeechT5 (Base self-supervised or Fine-tuned ASR) on LibriSpeech.
    """

    def __init__(
        self,
        model_dir: Union[tk.Path, str],
        dataset_dir: Union[tk.Path, str],
        text_lm_path: Optional[Union[tk.Path, str]] = None,
        phoneme_lexicon_path: Optional[Union[tk.Path, str]] = None,
        max_seqs: Optional[int] = None,
        max_cross_att_seqs: int = 10,
        max_pca_seqs: int = 50,
        name_prefix: str = "speecht5_eval",
        model_title: Optional[str] = None,
        dummy_version: Optional[Union[int, str]] = None,
    ):
        super().__init__()
        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        self.text_lm_path = text_lm_path
        self.phoneme_lexicon_path = phoneme_lexicon_path
        self.max_seqs = max_seqs
        self.max_cross_att_seqs = max_cross_att_seqs
        self.max_pca_seqs = max_pca_seqs
        self.name_prefix = name_prefix
        self.model_title = model_title
        self.dummy_version = dummy_version

        # Outputs
        self.out_hypotheses_txt = self.output_path("hypotheses.txt")
        self.out_reference_txt = self.output_path("reference.txt")
        self.out_asr_summary_json = self.output_path("asr_summary.json")
        self.out_asr_report_txt = self.output_path("asr_report.txt")
        self.out_cross_att_dir = self.output_path("cross_attention", directory=True)
        self.out_pca_dir = self.output_path("pca_analysis", directory=True)

    def tasks(self) -> Iterator[Task]:
        # Evaluates SpeechT5 with GPU (or fallback to CPU)
        yield Task("run", rqmt={"cpu": 4, "mem": 16, "time": 4, "gpu": 1})

    def _resolve_path(self, p: Optional[Union[tk.Path, str]]) -> Optional[str]:
        if p is None:
            return None
        if isinstance(p, tk.Path):
            path_str = p.get_path()
        else:
            path_str = os.path.expandvars(str(p))
        if path_str.startswith("/hpcwork/"):
            path_str = "/rwthfs/rz/cluster/hpcwork/" + path_str[len("/hpcwork/"):]
        return path_str

    def run(self):
        m_dir = self._resolve_path(self.model_dir)
        d_dir = self._resolve_path(self.dataset_dir)
        lm_path = self._resolve_path(self.text_lm_path)
        lex_path = self._resolve_path(self.phoneme_lexicon_path)
        out_dir = self.output_path("").get_path()

        evaluate_speecht5_base(
            model_dir=m_dir,
            dataset_dir=d_dir,
            text_lm_path=lm_path,
            phoneme_lexicon_path=lex_path,
            output_dir=out_dir,
            max_seqs=self.max_seqs,
            max_cross_att_seqs=self.max_cross_att_seqs,
            max_pca_seqs=self.max_pca_seqs,
            model_title=self.model_title,
        )


SpeechT5EvalJob = SpeechT5BaseEvalJob
