import argparse
import ast
import json
import sys
import numpy as np
import os
import xml, gzip
import xml.etree.ElementTree as ET
from xml import etree
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

# dataset = None


def hdf_dataset_init(out_dim, file_name):
  """
  :param str file_name: filename of hdf dataset file in the filesystem
  :rtype: hdf_dataset_mod.HDFDatasetWriter
  """
  import returnn.datasets.hdf as hdf_dataset_mod
  return hdf_dataset_mod.SimpleHDFWriter(
    filename=file_name, dim=out_dim, ndim=1)


def parse_lexicon(lexicon):
  lemmas = lexicon.findall("lemma")
  # print(lemmas[0].attrib)
  lemmas = [lemma for lemma in lemmas if len(lemma.findall("orth")) > 0 and len(lemma.findall("phon")) > 0]
  result = {
    orth.text: [pron.text for pron in lemma.findall("phon")] for lemma in lemmas for orth in lemma.findall("orth")}

  return result


def get_word_phon_map(greedy, words, special_tokens, phonemes_non_blank, seq_idx, bpe_non_blank, tag):
  skip_seq = False
  pron_not_in_seq = False
  if greedy:
    # get a unique mapping from words to phonemes
    # if a unique mapping is not possible, store and skip the sequence
    word_phon_map = []
    # start = time.time()
    rem_phons = " ".join(phonemes_non_blank[phonemes_non_blank != "[SILENCE]"])
    for word in words:
      if word in special_tokens:
        word_phon_map.append(word)
        rem_phons = rem_phons[len(word + " "):]
      else:
        if word not in lexicon:
          skip_seq = True
          break
        else:
          phon_cands = lexicon[word]
        # print("PHON CANDS: ", phon_cands)
        matching_cands = []
        for cand in phon_cands:
          if rem_phons.startswith(cand):
            matching_cands.append(cand)

        if len(matching_cands) > 1:
          # only keep the longest matching candidate
          matching_cands = [sorted(matching_cands, key=len)[-1]]
        if len(matching_cands) != 1:
          assert len(matching_cands) == 0
          skip_seq = True
          break
        word_phon_map.append(matching_cands[0])
        rem_phons = rem_phons[len(matching_cands[0] + " "):]

  else:
    non_sil_phons = " ".join(phonemes_non_blank[phonemes_non_blank != "[SILENCE]"])
    hyps = [[[], non_sil_phons]]
    # print("current hyps: ", hyps)
    # print("-----------------------")
    for word in words:
      new_hyps = []
      for hyp in hyps:
        # in this case, there is no pronunciation
        # we just add a new hypotheses if the remaining phon seq starts with the special token
        if word in special_tokens:
          if hyp[1].startswith(word):
            ext_seq = list(hyp[0])
            ext_seq.append(word)
            rem_phons = hyp[1][len(word + " "):]
            new_hyps.append([ext_seq, rem_phons])
        else:
          if word not in lexicon:
            print("WORD NOT IN LEXICON: ", word)
            skip_seq = True
            break
          else:
            phon_cands = lexicon[word]
          # in this case, we possibly have multiple pronunciations
          # for each one, we check whether it matches our remaining phon seq and, if yes, we add another hypothesis
          # print("PHON CANDS: ", phon_cands)
          i = 0
          for cand in phon_cands:
            # the remaining seq needs to start with the candidate and the candidate needs to "consume" whole phonemes
            # of the alignment
            if hyp[1].startswith(cand) and (hyp[1][len(cand):].startswith(" ") or len(hyp[1][len(cand):]) == 0):
              # print("HYP: ", hyp)
              # print("CAND: ", cand)
              i += 1
              ext_seq = list(hyp[0])
              ext_seq.append(cand)
              rem_phons = hyp[1][len(cand + " "):]
              new_hyps.append([ext_seq, rem_phons])
          # if i > 1:
            # print("NOOOOOOOOOOOOOW: ", seq_idx)
          # print(len(new_hyps))
      if len(new_hyps) == 0:
        print(bpe_non_blank)
        print(hyps)
        print(seq_idx)
        print(tag)


      hyps = list(new_hyps)
      if len(hyps) == 0:
        print("0 HYPS!!!!!")
        break
      # print("current hyps: ", hyps)
      # print("-----------------------")

      # skip_seq = True
      # word_phon_map = []
      # non_sil_phons = " ".join(phonemes_non_blank[phonemes_non_blank != "[SILENCE]"])
      # for word in words:
      #   if word not in special_tokens:
      #     phon_cands = lexicon[word]
      #     # print("CANDS: ", phon_cands)
      #     # print("phonemes: ", non_sil_phons)
      #     cand_in_seq = [cand in non_sil_phons for cand in phon_cands]
      #     if not any(cand_in_seq):
      #       pron_not_in_seq = True
      #       # print("!!!!!!!!!!!!!!!!!!")
      #       # print("CANNOT WORK")
      #       # print("!!!!!!!!!!!!!!!!!!")
      #       break
    if len(hyps) == 0:
      print(bpe_non_blank)
      print(phonemes_non_blank)
      print(seq_idx)
      print(tag)
      skip_seq = True
      word_phon_map = []
      # assert False
    elif len(hyps) >= 1:
      # print("MORE THAN 1 HYP!!!!")
      # print(bpe_non_blank)
      # print(hyps)
      # print(seq_idx)
      # print(tag)
      hyps = [hyp for hyp in hyps if hyp[1] == ""]
      # print(hyps)
      # assert len(hyps) == 1
      if len(hyps) != 1:
        print("MORE THAN 1 HYP!!!!")
        print(bpe_non_blank)
        print(hyps)
        print(seq_idx)
        print(tag)

      word_phon_map = hyps[0][0]
    if len(words) != len(word_phon_map) and len(word_phon_map) > 0:
      print(words)
      print(word_phon_map)
      raise ValueError

  # assert False

  return word_phon_map, skip_seq, pron_not_in_seq



def create_augmented_alignment(bpe_upsampling_factor, hdf_dataset, skipped_seqs_file):
  dataset.init_seq_order()
  seq_idx = 0
  sil_idx = 0
  special_tokens = ("[NOISE]", "[VOCALIZEDNOISE]", "[LAUGHTER]")
  skipped_seqs = []

  # bpe_merge_time = 0
  # words_to_phon_time = 0
  # words_to_bpe_time = 0
  # word_bound_time = 0
  # bpe_sil_align_time = 0

  not_in_lex_count = 0
  matching_cands_0_count = 0
  matching_cands_g1_count = 0
  pron_not_in_phon_seq_count = 0
  match_bpe_to_phons_err_count = 0
  num_bpe_single_char_at_start = 0
  num_segments = 0

  while dataset.is_less_than_num_seqs(seq_idx) and seq_idx < float("inf"):
    skip_seq = False
    # print(seq_idx)
    if seq_idx % 1000 == 0:
      complete_frac = dataset.get_complete_frac(seq_idx)
      print("Progress: %.02f" % (complete_frac * 100))
    dataset.load_seqs(seq_idx, seq_idx + 1)

    # load alignments (idx sequences)
    bpe_align = dataset.get_data(seq_idx, "bpe_align")
    if np.all(bpe_align == bpe_blank_idx):
      skipped_seqs.append(dataset.get_tag(seq_idx))
      seq_idx += 1
      continue
    phoneme_align = dataset.get_data(seq_idx, "data")
    # bpe and phoneme string sequence
    bpes = np.array([bpe_vocab[idx] for idx in bpe_align])
    phonemes = np.array([phoneme_vocab[idx] for idx in phoneme_align])
    # string seqs without blanks
    bpes_non_blank = bpes[bpe_align != bpe_blank_idx]
    phonemes_non_blank = phonemes[phoneme_align != phoneme_blank_idx]
    # upscale bpe sequence to match phoneme align length
    rem_num = len(phoneme_align) % bpe_upsampling_factor
    upscaled_bpe_align = [i for j in bpe_align[:-1] for i in ([bpe_blank_idx] * (bpe_upsampling_factor-1)) + [j]]
    if rem_num == 0:
      upscaled_bpe_align += [i for j in bpe_align[-1:] for i in ([bpe_blank_idx] * (bpe_upsampling_factor-1)) + [j]]
    else:
      upscaled_bpe_align += [i for j in bpe_align[-1:] for i in ([bpe_blank_idx] * (rem_num - 1)) + [j]]
    # get word sequence by merging non-blank bpe strings
    words = []
    cur = ""
    # start = time.time()
    for subword in bpes_non_blank:
      cur += subword
      if subword.endswith("@@"):
        cur = cur[:-2]
      else:
        words += [cur]
        cur = ""
    # bpe_merge_time += time.time() - start

    # print("BPES: ", bpes_non_blank)
    word_phon_map, skip_seq, pron_not_in_seq = get_word_phon_map(
      greedy=False, words=words, special_tokens=special_tokens, phonemes_non_blank=phonemes_non_blank, seq_idx=seq_idx,
      bpe_non_blank=bpes_non_blank, tag=dataset.get_tag(seq_idx))
    if pron_not_in_seq:
      pron_not_in_phon_seq_count += 1
    if skip_seq:
      matching_cands_0_count += 1
      skipped_seqs.append(dataset.get_tag(seq_idx))
      seq_idx += 1
      continue
    # print(word_phon_map)

    # words_to_phon_time += time.time() - start
    # start = time.time()

    # get mapping from word sequence to bpe tokens
    # additionally, store the fraction of a subword of the merged word
    # e.g.: for [cat@@, s], store the fraction that "cat@@" and "s" inhabit in the total alignment of "cats"
    word_bpe_map = []
    mapping = []
    prev_bound = 0
    for i, bpe_idx in enumerate(upscaled_bpe_align):
      if bpe_idx != bpe_blank_idx:
        seg_size = i - prev_bound
        prev_bound += seg_size
        if prev_bound == seg_size:
          seg_size += 1
        if bpe_vocab[bpe_idx].endswith("@@"):
          mapping.append([bpe_idx, seg_size])
          if len(mapping) == 1:
            num_segments += 1
            if len(bpe_vocab[bpe_idx].split("@")[0]) in [1]:
              num_bpe_single_char_at_start += 1
        else:
          mapping.append([bpe_idx, seg_size])
          total_size = sum(size for _, size in mapping)
          mapping = [[label, size / total_size] for label, size in mapping]
          word_bpe_map.append(mapping)
          mapping = []

    # words_to_bpe_time += time.time() - start
    # start = time.time()

    # determine the word boundaries in the phoneme alignment
    word_ends = []
    word_starts = []
    sil_bounds = []
    new_bpe_align = []
    align_idx = 0
    # go through each word to phoneme mapping
    for mapping in word_phon_map:
      # print(mapping)
      word_phons = mapping.split(" ")
      num_phons = len(word_phons)
      # word_phons = [phon_to_idx[phon] for phon in word_phons]
      last_phon_in_word = word_phons[-1]
      if last_phon_in_word not in special_tokens:
        last_phon_in_word += "{#+#}@f.0"
      else:
        last_phon_in_word += "{#+#}.0"
      last_phon_in_word = phon_to_idx[last_phon_in_word]
      # go through the phoneme align, starting from the word boundary of the previous word
      phon_counter = 0
      for i, phon_idx in enumerate(phoneme_align[align_idx:]):
        if phon_idx != sil_idx and phon_idx != phoneme_blank_idx:
          phon_counter += 1
          if phon_counter == 1:
            word_starts.append(align_idx + i)
        if phon_idx == sil_idx:
          word_ends.append(align_idx + i)
          word_starts.append(align_idx + i)
          sil_bounds.append(align_idx + i)
        # store word end positions, update the align_idx and go to the next word mapping
        elif phon_idx == last_phon_in_word:
          word_ends.append(align_idx + i)
          align_idx = align_idx + i + 1
          break
    if len(phoneme_align[align_idx:]) > 0:
      if phoneme_align[-1] != sil_idx:
        print("SEQ IDX: ", seq_idx)
        print("WORD PHON MAP: ", word_phon_map)
        print("PHON ALIGN: ", phonemes_non_blank)
        print("BPE ALIGN: ", bpe_align)
        print("WORDS: ", words)
      assert phoneme_align[-1] == sil_idx
      sil_bounds.append(len(phoneme_align)-1)
      word_ends.append(len(phoneme_align) - 1)
      word_starts.append(len(phoneme_align) - 1)

    # print("SEQ IDX: ", seq_idx)
    # print("WORD PHON MAP: ", word_phon_map)
    # print("PHON ALIGN: ", phoneme_align)
    # print("BPE ALIGN: ", bpe_align)
    # print("WORDS: ", words)
    # print("WORD ENDS: ", word_ends)
    # print("WORD STARTS: ", word_starts)


    # word_bound_time += time.time() - start
    # start = time.time()

    new_bpe_blank_idx = bpe_blank_idx + 1
    bpe_sil_align = [new_bpe_blank_idx] * len(phoneme_align)
    prev_end = 0
    bpe_idx = 0
    # print(seq_idx)
    for start, end in zip(word_starts, word_ends):
      if end in sil_bounds:
        # just set the same silence bound in the new alignment
        bpe_sil_align[end] = sil_idx
        prev_end = end
      else:
        size = end - prev_end
        if start == 0:
          size += 1
        bpe_map = word_bpe_map[bpe_idx]
        if len(bpe_map) != 1:
          # bpe_sil_align[start] = bpe_map[0][0]
          for i, (bpe, _) in enumerate(bpe_map):
            if i != len(bpe_map) - 1:
              frac = 1 / len(bpe_map)
              offset = max(int(size * frac), 1)
              if prev_end + offset > len(bpe_sil_align) - 1:
                print("\n\n")
                print("CANNOT MATCH BPE ALIGN TO PHON ALIGN")
                print("BPE: ", bpes)
                print("PHONEMES: ", phonemes)
                print("TAG: ", tag)
                print("PHON ALIGN: ", phoneme_align)
                print("BPE SIL ALIGN: ", bpe_sil_align)
                print("BPE ALIGN: ", bpe_align)
                print("PREV BOUND: ", prev_end)
                print("BOUND: ", end)
                print("BOUNDS: ", word_ends)
                print("OFFSET: ", offset)
                print("SIZE: ", size)
                print("FRAC: ", frac)
                print("BPE MAP: ", bpe_map)
                print("\n\n")
                # in this case, it cannot easily be guaranteed that each bpe label gets at least one frame
                # therefore, we skip
                skip_seq = True
                match_bpe_to_phons_err_count += 1
                break
              bpe_sil_align[prev_end + offset] = bpe
              prev_end += offset
            else:
              bpe_sil_align[end] = bpe
        else:
          bpe_sil_align[end] = bpe_map[0][0]
        # check, whether there was a problem in the bpe mapping loop
        if skip_seq:
          break
        bpe_idx += 1
        prev_end = end
    # like above, if there was an error, we skip the sequence add the tag to the blacklist
    if skip_seq:
      skipped_seqs.append(dataset.get_tag(seq_idx))
      seq_idx += 1
      continue

    # plot some random examples
    if np.random.rand(1) < 0.001:
      plot_aligns(upscaled_bpe_align, phoneme_align, bpe_sil_align, seq_idx)

    # dump new alignment into hdf file
    seq_len = len(bpe_sil_align)
    tag = dataset.get_tag(seq_idx)
    new_data = tf.constant(np.expand_dims(bpe_sil_align, axis=0), dtype="int32")
    extra = {}
    seq_lens = {0: tf.constant([seq_len]).numpy()}
    ndim_without_features = 1  # - (0 if data_obj.sparse or data_obj.feature_dim_axis is None else 1)
    for dim in range(ndim_without_features):
      if dim not in seq_lens:
        seq_lens[dim] = np.array([new_data.shape[dim + 1]] * 1, dtype="int32")
    batch_seq_sizes = np.zeros((1, len(seq_lens)), dtype="int32")
    for i, (axis, size) in enumerate(sorted(seq_lens.items())):
      batch_seq_sizes[:, i] = size
    extra["seq_sizes"] = batch_seq_sizes

    hdf_dataset.insert_batch(new_data, seq_len=seq_lens, seq_tag=[tag], extra=extra)

    seq_idx += 1

  # print(bpe_merge_time)
  # print(words_to_phon_time)
  # print(words_to_bpe_time)
  # print(word_bound_time)
  # print(bpe_sil_align_time)

  print("NOT IN LEX COUNT: ", not_in_lex_count)
  print("MATCHING CANDS 0 COUNT: ", matching_cands_0_count)
  print("MATCHING CANDS GREATER 1 COUNT: ", matching_cands_g1_count)
  print("PRON NOT IN SEQ COUNT: ", pron_not_in_phon_seq_count)
  print("MATCH BPE TO PHONS ERR COUNT: ", match_bpe_to_phons_err_count)
  print("NUM SEGMENTS: ", num_segments)
  print("NUM FIRST BPE OF SEGMENT IS SINGLE CHAR: ", num_bpe_single_char_at_start)

  with open(skipped_seqs_file, "w+") as f:
    f.write(str(skipped_seqs))
  # print("Skipped Sequence Pairs:")
  # print("\n".join([str(pair) for pair in skipped_pairs]))


def plot_aligns(bpe_align, phoneme_align, bpe_silence_align, seq_idx):
  # rem_num = len(phoneme_align) % 6
  # bpe_align = [
  #   i for j in red_bpe_align[:-1] for i in ([bpe_blank_idx] * 5) + [j]]
  # print(bpe_align)
  # if rem_num == 0:
  #   bpe_align += [i for j in red_bpe_align[-1:] for i in ([bpe_blank_idx] * 5) + [j]]
  # else:
  #   bpe_align += [i for j in red_bpe_align[-1:] for i in ([bpe_blank_idx] * (rem_num - 1)) + [j]]
  matrix = np.concatenate(
    [np.array([[0 if i == bpe_blank_idx else 1 for i in bpe_align]]),
     np.array([[0 if i == bpe_blank_idx + 1 else 1 for i in bpe_silence_align]]),
     np.array([[0 if i == phoneme_blank_idx else 1 for i in phoneme_align]])],
    axis=0
  )

  matrix_masked = []
  for i, m in enumerate(matrix):
    mask = np.ones(matrix.shape)
    mask[i::3] = np.zeros(m.shape)
    matrix_masked.append(np.ma.masked_array(matrix, mask))

  bpe_align = np.array(bpe_align)
  bpe_silence_align = np.array(bpe_silence_align)
  phoneme_align = np.array(phoneme_align)

  bpe_ticks = np.where(bpe_align != bpe_blank_idx)[0]
  bpe_labels = bpe_align[bpe_align != bpe_blank_idx]
  bpe_labels = [bpe_vocab[i] for i in bpe_labels]

  bpe_silence_ticks = np.where(bpe_silence_align != bpe_blank_idx + 1)[0]
  bpe_silence_labels = bpe_silence_align[bpe_silence_align != bpe_blank_idx + 1]
  bpe_silence_labels = [bpe_silence_vocab[i] for i in bpe_silence_labels]

  phoneme_ticks = np.where(phoneme_align != phoneme_blank_idx)[0]
  phoneme_labels = phoneme_align[phoneme_align != phoneme_blank_idx]
  phoneme_labels = [phoneme_vocab[i] for i in phoneme_labels]

  plt.figure(figsize=(10, 5), constrained_layout=True)
  ax = plt.gca()
  # fig = plt.gcf()
  # fig.set_size_inches(10, 2)
  for mat, cmap in zip(matrix_masked, [plt.cm.get_cmap("Blues"), plt.cm.get_cmap("Reds"), plt.cm.get_cmap("Greens")]):
    ax.matshow(mat, aspect="auto", cmap=cmap)
  # # create second x axis for hmm alignment labels and plot same matrix
  hmm_ax = ax.twiny()
  bpe_silence_ax = ax.twiny()

  #
  ax.set_xticks(list(bpe_ticks))
  # ax.xaxis.set_major_formatter(ticker.NullFormatter())
  # ax.set_xticks(bpe_xticks_minor, minor=True)
  ax.set_xticklabels(list(bpe_labels), rotation="vertical")
  # ax.tick_params(axis="x", which="minor", length=0, labelsize=17)
  # ax.tick_params(axis="x", which="major", length=10)
  ax.set_xlabel("BPE Alignment", color="darkblue")
  # ax.set_ylabel("Output RNA BPE Labels", fontsize=18)
  ax.xaxis.tick_top()
  ax.xaxis.set_label_position('top')
  # ax.spines['top'].set_position(('outward', 50))
  # # for tick in ax.xaxis.get_minor_ticks():
  # #   tick.label1.set_horizontalalignment("center")
  ax.set_yticklabels(["BPE", "BPE", "BPE + Silence", "Phonemes"])

  bpe_silence_ax.set_xlim(ax.get_xlim())
  bpe_silence_ax.set_xticks(list(bpe_silence_ticks))
  # bpe_silence_ax.xaxis.tick_top()
  bpe_silence_ax.set_xlabel("BPE + Silence Alignment", color="darkred")
  bpe_silence_ax.xaxis.set_label_position('top')
  bpe_silence_ax.spines['top'].set_position(('outward', 70))
  bpe_silence_ax.set_xticklabels(list(bpe_silence_labels), rotation="vertical")

  # # set x ticks and labels and positions for hmm axis
  hmm_ax.set_xticks(phoneme_ticks)
  # hmm_ax.xaxis.set_major_formatter(ticker.NullFormatter())
  # hmm_ax.set_xticks(hmm_xticks_minor, minor=True)
  hmm_ax.set_xticklabels(phoneme_labels, rotation="vertical")
  # hmm_ax.tick_params(axis="x", which="minor", length=0)
  # hmm_ax.tick_params(axis="x", which="major", length=0)
  hmm_ax.xaxis.set_ticks_position('bottom')
  hmm_ax.xaxis.set_label_position('bottom')
  hmm_ax.set_xlabel("HMM Phoneme Alignment", color="darkgreen")
  #
  # time_xticks = [x - .5 for x in range(0, len(phoneme_align), 2)]
  # time_xticks_labels = [x for x in range(0, len(phoneme_align), 2)]
  # time_ax.set_xlabel("Input Time Frames", fontsize=18)
  # time_ax.xaxis.tick_bottom()
  # time_ax.xaxis.set_label_position('bottom')
  # time_ax.set_xlim(ax.get_xlim())
  # time_ax.set_xticks(time_xticks)
  # time_ax.set_xticklabels(time_xticks_labels, fontsize=17)

  plt.savefig("plot%s.pdf" % seq_idx)
  plt.savefig("plot%s.png" % seq_idx)
  plt.close()


def init(
  bpe_hdf, phoneme_hdf, bpe_vocab_file, phoneme_vocab_file, phoneme_lexicon_file, bpe_blank, phoneme_blank,
  segment_file, out_vocab):
  global config
  global dataset
  global word_vocab
  global bpe_vocab
  global bpe_silence_vocab
  global phoneme_vocab
  global lexicon
  global bpe_blank_idx
  global phoneme_blank_idx
  global phon_to_idx

  bpe_blank_idx = bpe_blank
  phoneme_blank_idx = phoneme_blank

  with open(bpe_vocab_file, "r") as f:
    bpe_vocab = ast.literal_eval(f.read())
    bpe_silence_vocab = bpe_vocab.copy()
    voc_noise_idx = bpe_vocab.pop("[VOCALIZED-NOISE]")
    bpe_vocab["[VOCALIZEDNOISE]"] = voc_noise_idx
    bpe_vocab = {int(v): k for k, v in bpe_vocab.items()}
    bpe_vocab[bpe_blank] = "<b>"

  bpe_silence_vocab["<s>"] = bpe_blank
  bpe_silence_vocab["</s>"] = bpe_blank
  bpe_silence_vocab["[SILENCE]"] = 0
  with open(out_vocab, "w+") as f:
    json.dump(bpe_silence_vocab, f)
  bpe_silence_vocab["<b>"] = bpe_blank + 1
  voc_noise_idx = bpe_silence_vocab.pop("[VOCALIZED-NOISE]")
  bpe_silence_vocab["[VOCALIZEDNOISE]"] = voc_noise_idx
  bpe_silence_vocab = {int(v): k for k, v in bpe_silence_vocab.items()}

  with open(phoneme_vocab_file, "r") as f:
    phon_to_idx = ast.literal_eval(f.read())
    phon_to_idx = {k: int(v) for k, v in phon_to_idx.items()}
    phoneme_vocab = {int(v): k.split("{")[0] for k, v in phon_to_idx.items()}
    phoneme_vocab[phoneme_blank] = "<b>"
    phon_to_idx["<b>"] = phoneme_blank
  word_vocab_file = "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/words/vocab_json"
  with open(word_vocab_file, "r") as f:
    word_vocab = ast.literal_eval(f.read())
    word_vocab = {int(v): k for k, v in word_vocab.items()}
  with gzip.open(phoneme_lexicon_file, "r") as f:
    # xml_parser = ET.XMLParser(encoding="iso-8859-5")
    lexicon = ET.fromstring(f.read())
    lexicon = parse_lexicon(lexicon)

  rnn.init_better_exchook()
  rnn.init_thread_join_hack()
  rnn.init_config(config_filename=None, default_config={"cache_size": 0})
  config = rnn.config
  config.set("log", None)
  rnn.init_log()
  print("Returnn augment bpe align starting up", file=rnn.log.v2)

  # init bpe and phoneme align datasets
  bpe_dataset_dict = {
    "class": "HDFDataset", "files": [bpe_hdf], "use_cache_manager": True, 'estimated_num_seqs': 3000,
    'partition_epoch': 1,
    'seq_list_filter_file': segment_file
  }
  phoneme_dataset_dict = {
    "class": "HDFDataset", "files": [phoneme_hdf], "use_cache_manager": True, 'estimated_num_seqs': 3000,
    'partition_epoch': 1,
    'seq_list_filter_file': segment_file
  }

  dataset_dict = {
    'class': 'MetaDataset',
    # "seq_list_filter_file": segment_file,
    'data_map':
      {'bpe_align': ('bpe_align', 'data'), 'data': ('data', 'data')},
    'datasets': {
      'bpe_align': bpe_dataset_dict, "data": phoneme_dataset_dict},
    'seq_order_control_dataset': 'data'}

  dataset = rnn.init_dataset(dataset_dict)

  rnn.returnn_greeting()
  rnn.init_faulthandler()
  rnn.init_config_json_network()


def main():
  arg_parser = argparse.ArgumentParser(description="Calculate segment statistics.")
  arg_parser.add_argument("bpe_align_hdf", help="hdf file which contains the extracted bpe alignments")
  arg_parser.add_argument("phoneme_align_hdf", help="hdf file which contains the extracted phoneme alignments")
  arg_parser.add_argument("--bpe_blank_idx", help="the blank index in the bpe alignment", type=int)
  arg_parser.add_argument("--phoneme_blank_idx", help="the blank index in the phoneme alignment", type=int)
  arg_parser.add_argument("--bpe_vocab", help="mapping from bpe idx to label", type=str)
  arg_parser.add_argument("--phoneme_vocab", help="mapping from phoneme idx to label", type=str)
  arg_parser.add_argument("--phoneme_lexicon", help="mapping from words to phonemes", type=str)
  arg_parser.add_argument("--segment_file", help="segment whitelist", type=str)
  arg_parser.add_argument(
    "--bpe_upsampling_factor", help="factor to get bpe alignment to same length as phoneme alignment", type=int)
  arg_parser.add_argument("--out_align", help="output path for augmented alignment", type=str)
  arg_parser.add_argument("--out_vocab", help="output path for augmented vocab", type=str)
  arg_parser.add_argument("--out_skipped_seqs", help="output path for skipped seqs", type=str)
  arg_parser.add_argument("--returnn_root", type=str)
  args = arg_parser.parse_args()
  sys.path.insert(0, args.returnn_root)
  global rnn
  import returnn.__main__ as rnn
  import returnn.tf.compat as tf_compat
  tf_compat.v1.enable_eager_execution()

  init(
    args.bpe_align_hdf, args.phoneme_align_hdf, args.bpe_vocab, args.phoneme_vocab, args.phoneme_lexicon,
    args.bpe_blank_idx, args.phoneme_blank_idx, args.segment_file, args.out_vocab)

  hdf_dataset = hdf_dataset_init(
    out_dim=dataset.get_data_dim("bpe_align") + 1, file_name=args.out_align)

  try:
    create_augmented_alignment(args.bpe_upsampling_factor, hdf_dataset, skipped_seqs_file=args.out_skipped_seqs)
    hdf_dataset.close()
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn.finalize()


if __name__ == "__main__":
  main()
