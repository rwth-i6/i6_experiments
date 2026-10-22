"""
RETURNN PyTorch implementation of UNdreaMT (Artetxe et al., ICLR 2018):
"Unsupervised Neural Machine Translation" (https://github.com/artetxem/undreamt/tree/master/undreamt)

Architecture:
  - Special token convention (matching undreamt/data.py):
      PAD = 0, OOV (<unk>) = 1, EOS (</s>) = 2, SOS (<s>) = 3, regular vocabulary words = 4 .. V+3.
  - CrossLingualEmbedding:
      Sums a frozen pre-trained cross-lingual word embedding table (V+1, 300) with a trainable
      special-symbol embedding table (4+1, 300) so pre-trained cross-lingual embeddings remain
      strictly fixed while special tokens (OOV, EOS, SOS) are learned.
  - RNNEncoder:
      Shared 2-layer bidirectional GRU (hidden_size=600, i.e. 300 per direction, dropout=0.3).
  - RNNAttentionDecoder:
      Language-specific 2-layer StackedGRU (hidden_size=600, dropout=0.3) with input feeding
      (concatenating current token embedding [300] and previous attentional hidden state [600] -> [900])
      and GlobalAttention (alignment_function='general', score(h_t, h_s) = h_t^T W_a h_s).
  - Generator:
      Language-specific LinearGenerator (or EmbeddingGenerator) projecting 600 -> V+4 log-probabilities.
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf


PAD_ID = 0
OOV_ID = 1
EOS_ID = 2
SOS_ID = 3
SPECIAL_SYMBOLS = 4


class CrossLingualWordEmbedding(nn.Module):
    """
    Combines pre-trained cross-lingual word embeddings (frozen by default) for regular words (ids >= 4)
    and trainable embeddings for special symbols (PAD=0, OOV=1, EOS=2, SOS=3), matching undreamt/encoder.py
    and undreamt/data.py.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 300,
        freeze_word_embeddings: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.freeze_word_embeddings = freeze_word_embeddings

        # Regular word embeddings: index 0 is padding (for special symbols), 1..vocab_size are words (ids 4..vocab_size+3)
        self.word_embeddings = nn.Embedding(vocab_size + 1, embedding_size, padding_idx=0)
        self.word_embeddings.weight.requires_grad = not freeze_word_embeddings

        # Special symbol embeddings: index 0 is PAD (zero), 1=OOV, 2=EOS, 3=SOS
        self.special_embeddings = nn.Embedding(SPECIAL_SYMBOLS, embedding_size, padding_idx=PAD_ID)

    def load_pretrained_npy(self, npy_path: str):
        """
        Loads a (V, D) numpy array of aligned cross-lingual embeddings into word_embeddings.weight[1:V+1].
        """
        resolved_path = os.path.expandvars(str(npy_path))
        if resolved_path.startswith("/hpcwork/"):
            alt = "/rwthfs/rz/cluster/hpcwork/" + resolved_path[len("/hpcwork/") :]
            if os.path.exists(alt):
                resolved_path = alt

        weights = np.load(resolved_path).astype(np.float32)
        V, D = weights.shape
        assert D == self.embedding_size, f"Embedding dim mismatch: expected {self.embedding_size}, got {D}"
        num_to_copy = min(V, self.vocab_size)
        with torch.no_grad():
            self.word_embeddings.weight.data[0].zero_()
            self.word_embeddings.weight.data[1 : num_to_copy + 1] = torch.from_numpy(weights[:num_to_copy])

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        :param ids: LongTensor of token IDs with shape (...), values in [0, vocab_size + 3]
        :return: FloatTensor of embeddings with shape (..., embedding_size)
        """
        is_special = ids < SPECIAL_SYMBOLS
        special_ids = torch.where(is_special, ids, torch.zeros_like(ids))
        word_ids = torch.where(is_special, torch.zeros_like(ids), ids - (SPECIAL_SYMBOLS - 1))
        word_ids = word_ids.clamp(min=0, max=self.vocab_size)
        return self.word_embeddings(word_ids) + self.special_embeddings(special_ids)


class GlobalAttention(nn.Module):
    """
    Global attention mechanism from undreamt/attention.py:
      alignment_function = 'general' (score(h_t, h_s) = h_t^T W_a h_s) or 'dot'
      attentional hidden state = tanh(W_c * c_t + W_q * h_t)
    """

    def __init__(self, dim: int, alignment_function: str = "general"):
        super().__init__()
        self.dim = dim
        self.alignment_function = alignment_function
        if self.alignment_function == "general":
            self.align = nn.Linear(dim, dim, bias=False)
        elif self.alignment_function != "dot":
            raise ValueError(f"Invalid alignment function: {alignment_function}")
        self.context_projection = nn.Linear(dim, dim, bias=False)
        self.query_projection = nn.Linear(dim, dim, bias=True)
        self.tanh = nn.Tanh()

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param query: (batch, dim)
        :param context: (src_len, batch, dim)
        :param mask: BoolTensor of shape (batch, src_len), True where padded
        :return: attentional output (batch, dim)
        """
        # context_t: (batch, src_len, dim)
        context_t = context.transpose(0, 1)

        # Align scores: (batch, 1, dim) x (batch, dim, src_len) -> (batch, src_len)
        q = query if self.alignment_function == "dot" else self.align(query)
        scores = q.unsqueeze(1).bmm(context_t.transpose(1, 2)).squeeze(1)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        weights = F.softmax(scores, dim=-1)  # (batch, src_len)
        weighted_context = weights.unsqueeze(1).bmm(context_t).squeeze(1)  # (batch, dim)
        return self.tanh(self.context_projection(weighted_context) + self.query_projection(query))


class StackedGRU(nn.Module):
    """
    Multi-layer step-by-step GRU with inter-layer dropout (matching undreamt/decoder.py).
    """

    def __init__(self, layers: int, input_size: int, hidden_size: int, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = layers
        self.layers = nn.ModuleList()
        cur_in = input_size
        for _ in range(layers):
            self.layers.append(nn.GRUCell(cur_in, hidden_size))
            cur_in = hidden_size

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: (batch, input_size)
        :param hidden: (num_layers, batch, hidden_size)
        :return: (top_output: (batch, hidden_size), new_hidden: (num_layers, batch, hidden_size))
        """
        next_hidden = []
        cur = x
        for i, layer in enumerate(self.layers):
            h_i = layer(cur, hidden[i])
            next_hidden.append(h_i)
            cur = h_i
            if i + 1 != self.num_layers:
                cur = self.dropout(cur)
        return cur, torch.stack(next_hidden, dim=0)


class UNdreaMTRNNEncoder(nn.Module):
    """
    2-layer Bidirectional GRU Encoder from undreamt/encoder.py.
    """

    def __init__(
        self,
        embedding_size: int = 300,
        hidden_size: int = 600,
        layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        if bidirectional and hidden_size % 2 != 0:
            raise ValueError("hidden_size must be even for bidirectional encoder")
        self.directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.layers = layers
        self.hidden_size = hidden_size // self.directions
        self.total_hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            embedding_size,
            self.hidden_size,
            num_layers=layers,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0,
        )

    def forward(
        self,
        ids: torch.Tensor,
        lengths: torch.Tensor,
        word_embeddings: CrossLingualWordEmbedding,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param ids: (src_len, batch) LongTensor of token IDs
        :param lengths: (batch,) LongTensor of valid sequence lengths
        :param word_embeddings: language-specific CrossLingualWordEmbedding module
        :return:
          - output: (src_len, batch, hidden_size)
          - hidden: (layers, batch, hidden_size)
          - mask: (batch, src_len) BoolTensor (True on padded positions)
        """
        max_len, batch_size = ids.shape
        device = ids.device

        # Sort by descending length for pack_padded_sequence
        clamped_lengths = lengths.clamp(min=1, max=max_len).cpu()
        sorted_lengths, sort_idx = torch.sort(clamped_lengths, descending=True)
        _, unsort_idx = torch.sort(sort_idx.to(device))

        sorted_ids = ids[:, sort_idx.to(device)]
        embs = self.dropout(word_embeddings(sorted_ids))  # (src_len, batch, emb_dim)

        packed = nn.utils.rnn.pack_padded_sequence(embs, sorted_lengths.tolist(), batch_first=False)
        h0 = torch.zeros(self.layers * self.directions, batch_size, self.hidden_size, device=device, dtype=embs.dtype)
        output, hidden = self.rnn(packed, h0)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, total_length=max_len, batch_first=False)

        if self.bidirectional:
            # Concatenate forward and backward final states per layer: (layers, batch, 2 * hidden_size)
            hidden = torch.stack(
                [torch.cat((hidden[2 * i], hidden[2 * i + 1]), dim=1) for i in range(self.layers)],
                dim=0,
            )

        # Unsort back to original batch order
        output = output[:, unsort_idx, :]
        hidden = hidden[:, unsort_idx, :]

        # Padding mask: True where step >= length
        steps = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        mask = steps >= lengths.unsqueeze(1)  # (batch, max_len)

        return output, hidden, mask


class UNdreaMTRNNAttentionDecoder(nn.Module):
    """
    2-layer StackedGRU Decoder with GlobalAttention and input feeding from undreamt/decoder.py.
    """

    def __init__(
        self,
        embedding_size: int = 300,
        hidden_size: int = 600,
        layers: int = 2,
        dropout: float = 0.3,
        input_feeding: bool = True,
        alignment_function: str = "general",
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout = nn.Dropout(dropout)
        self.input_feeding = input_feeding
        rnn_input_size = embedding_size + hidden_size if input_feeding else embedding_size
        self.stacked_rnn = StackedGRU(layers, rnn_input_size, hidden_size, dropout=dropout)
        self.attention = GlobalAttention(hidden_size, alignment_function=alignment_function)

    def forward_step(
        self,
        input_ids: torch.Tensor,
        prev_hidden: torch.Tensor,
        prev_output: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        word_embeddings: CrossLingualWordEmbedding,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes a single decoder step.
        :param input_ids: (batch,) LongTensor
        :param prev_hidden: (layers, batch, hidden_size)
        :param prev_output: (batch, hidden_size)
        :param context: (src_len, batch, hidden_size)
        :param context_mask: (batch, src_len) BoolTensor
        :param word_embeddings: CrossLingualWordEmbedding
        :return: (output: (batch, hidden_size), hidden: (layers, batch, hidden_size))
        """
        emb = self.dropout(word_embeddings(input_ids))  # (batch, emb_dim)
        if self.input_feeding:
            emb = torch.cat([emb, prev_output], dim=1)
        rnn_out, hidden = self.stacked_rnn(emb, prev_hidden)
        att_out = self.attention(rnn_out, context, context_mask)
        att_out = self.dropout(att_out)
        return att_out, hidden

    def forward(
        self,
        ids: torch.Tensor,
        init_hidden: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        word_embeddings: CrossLingualWordEmbedding,
        generator: nn.Module,
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass across all target steps.
        :param ids: (trg_len, batch) LongTensor (starts with SOS_ID)
        :return: logprobs of shape (trg_len, batch, vocab_size + 4)
        """
        batch_size = ids.shape[1]
        hidden = init_hidden
        prev_out = torch.zeros(batch_size, self.hidden_size, device=ids.device, dtype=context.dtype)
        outputs = []
        for step_ids in ids.split(1, dim=0):
            prev_out, hidden = self.forward_step(
                step_ids.squeeze(0),
                hidden,
                prev_out,
                context,
                context_mask,
                word_embeddings,
            )
            outputs.append(prev_out)
        stacked_outputs = torch.stack(outputs, dim=0)  # (trg_len, batch, hidden_size)
        return generator(stacked_outputs)


class LinearGenerator(nn.Module):
    """
    Linear output projection + LogSoftmax from undreamt/generator.py.
    Output dimension is vocab_size + SPECIAL_SYMBOLS (where PAD_ID=0 is masked out).
    """

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.total_vocab_size = vocab_size + SPECIAL_SYMBOLS
        self.out = nn.Linear(hidden_size, self.total_vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.out(x), dim=-1)


class EmbeddingGenerator(nn.Module):
    """
    Pretrained-embedding-tied output generator from undreamt/generator.py.
    """

    def __init__(self, hidden_size: int, embedding_size: int, word_embeddings: CrossLingualWordEmbedding):
        super().__init__()
        self.hidden2embedding = nn.Linear(hidden_size, embedding_size)
        self.word_embeddings = word_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb_proj = self.hidden2embedding(x)
        # Concatenate special embeddings (1..3) and word embeddings (1..V)
        # Index 0 is PAD, 1..3 are OOV, EOS, SOS, 4..V+3 are regular words
        weight = torch.cat(
            [
                self.word_embeddings.special_embeddings.weight,  # (4, D)
                self.word_embeddings.word_embeddings.weight[1:],  # (V, D)
            ],
            dim=0,
        )
        scores = F.linear(emb_proj, weight)
        return F.log_softmax(scores, dim=-1)


class UNdreaMTModel(nn.Module):
    """
    Complete UNdreaMT Unsupervised Neural Machine Translation model in PyTorch/RETURNN.
    Shares a single 2-layer BiGRU encoder across source (e.g. German) and target (e.g. English),
    with frozen pre-trained cross-lingual encoder embeddings and language-specific decoders & generators.
    """

    def __init__(
        self,
        src_vocab_size: int = 50000,
        trg_vocab_size: int = 50000,
        embedding_size: int = 300,
        hidden_size: int = 600,
        layers: int = 2,
        dropout: float = 0.3,
        param_init: float = 0.1,
        src_pretrained_emb_path: Optional[str] = None,
        trg_pretrained_emb_path: Optional[str] = None,
        freeze_encoder_embeddings: bool = True,
        freeze_decoder_embeddings: bool = False,
        share_encoder_decoder_embeddings: bool = False,
        use_embedding_generator: bool = False,
        alignment_function: str = "general",
        **kwargs,
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout = dropout
        self.param_init = param_init

        # 1. Encoder cross-lingual embeddings (fixed by default)
        self.src_encoder_embeddings = CrossLingualWordEmbedding(
            vocab_size=src_vocab_size,
            embedding_size=embedding_size,
            freeze_word_embeddings=freeze_encoder_embeddings,
        )
        self.trg_encoder_embeddings = CrossLingualWordEmbedding(
            vocab_size=trg_vocab_size,
            embedding_size=embedding_size,
            freeze_word_embeddings=freeze_encoder_embeddings,
        )

        # 2. Decoder embeddings (learned from scratch by default, or shared/fixed if configured)
        if share_encoder_decoder_embeddings:
            self.src_decoder_embeddings = self.src_encoder_embeddings
            self.trg_decoder_embeddings = self.trg_encoder_embeddings
        else:
            self.src_decoder_embeddings = CrossLingualWordEmbedding(
                vocab_size=src_vocab_size,
                embedding_size=embedding_size,
                freeze_word_embeddings=freeze_decoder_embeddings,
            )
            self.trg_decoder_embeddings = CrossLingualWordEmbedding(
                vocab_size=trg_vocab_size,
                embedding_size=embedding_size,
                freeze_word_embeddings=freeze_decoder_embeddings,
            )

        # 3. Shared 2-layer BiGRU encoder
        self.encoder = UNdreaMTRNNEncoder(
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            layers=layers,
            bidirectional=True,
            dropout=dropout,
        )

        # 4. Language-specific 2-layer StackedGRU attention decoders
        self.src_decoder = UNdreaMTRNNAttentionDecoder(
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            layers=layers,
            dropout=dropout,
            input_feeding=True,
            alignment_function=alignment_function,
        )
        self.trg_decoder = UNdreaMTRNNAttentionDecoder(
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            layers=layers,
            dropout=dropout,
            input_feeding=True,
            alignment_function=alignment_function,
        )

        # 5. Language-specific generators
        if use_embedding_generator:
            self.src_generator = EmbeddingGenerator(hidden_size, embedding_size, self.src_decoder_embeddings)
            self.trg_generator = EmbeddingGenerator(hidden_size, embedding_size, self.trg_decoder_embeddings)
        else:
            self.src_generator = LinearGenerator(hidden_size, src_vocab_size)
            self.trg_generator = LinearGenerator(hidden_size, trg_vocab_size)

        # 6. Uniform [-0.1, 0.1] parameter initialization (matching undreamt/train.py)
        for p in self.parameters():
            if p.requires_grad:
                p.data.uniform_(-param_init, param_init)

        with torch.no_grad():
            self.src_encoder_embeddings.special_embeddings.weight.data[PAD_ID].zero_()
            self.trg_encoder_embeddings.special_embeddings.weight.data[PAD_ID].zero_()
            self.src_decoder_embeddings.special_embeddings.weight.data[PAD_ID].zero_()
            self.trg_decoder_embeddings.special_embeddings.weight.data[PAD_ID].zero_()

        # 7. Load pre-trained cross-lingual embeddings
        if src_pretrained_emb_path:
            self.src_encoder_embeddings.load_pretrained_npy(src_pretrained_emb_path)
            if freeze_decoder_embeddings and not share_encoder_decoder_embeddings:
                self.src_decoder_embeddings.load_pretrained_npy(src_pretrained_emb_path)
        if trg_pretrained_emb_path:
            self.trg_encoder_embeddings.load_pretrained_npy(trg_pretrained_emb_path)
            if freeze_decoder_embeddings and not share_encoder_decoder_embeddings:
                self.trg_decoder_embeddings.load_pretrained_npy(trg_pretrained_emb_path)

    def get_lang_modules(self, lang: str):
        """
        Returns (encoder_embeddings, decoder_embeddings, decoder, generator, vocab_size) for 'src' or 'trg'.
        """
        if lang in ("src", "de", "audio"):
            return (
                self.src_encoder_embeddings,
                self.src_decoder_embeddings,
                self.src_decoder,
                self.src_generator,
                self.src_vocab_size + SPECIAL_SYMBOLS,
            )
        elif lang in ("trg", "en", "text"):
            return (
                self.trg_encoder_embeddings,
                self.trg_decoder_embeddings,
                self.trg_decoder,
                self.trg_generator,
                self.trg_vocab_size + SPECIAL_SYMBOLS,
            )
        else:
            raise ValueError(f"Unknown language key: {lang}")

    def forward_lang_encoder(
        self,
        indices_bt: torch.Tensor,
        lengths_b: torch.Tensor,
        lang: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs the shared 2-layer BiGRU encoder on batch-first token indices [B, T] for the given language
        ('src'/'de' or 'trg'/'en') and returns:
          - enc_btf: [B, T, hidden_size] per-token encoder representations
          - enc_hidden: [layers, B, hidden_size]
          - enc_mask: [B, T] padding mask
          - lengths_b: [B] sequence lengths
        """
        enc_emb, _, _, _, _ = self.get_lang_modules(lang)
        # Transpose [B, T] -> [T, B] for UNdreaMTRNNEncoder
        ids_tb = indices_bt.long().transpose(0, 1).contiguous()
        out_tbf, hidden, mask = self.encoder(ids_tb, lengths_b.long(), enc_emb)
        out_btf = out_tbf.transpose(0, 1).contiguous()  # [B, T, F]
        return out_btf, hidden, mask, lengths_b

    def forward_src_encoder(
        self, indices_bt: torch.Tensor, lengths_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_lang_encoder(indices_bt, lengths_b, lang="src")

    def forward_trg_encoder(
        self, indices_bt: torch.Tensor, lengths_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_lang_encoder(indices_bt, lengths_b, lang="trg")

    # Aliases for compatibility with shared-encoder analysis modules
    def forward_audio(
        self, indices_bt: torch.Tensor, lengths_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_src_encoder(indices_bt, lengths_b)

    def forward_text(
        self, indices_bt: torch.Tensor, lengths_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_trg_encoder(indices_bt, lengths_b)

    def forward_seq2seq(
        self,
        src_ids: torch.Tensor,
        src_lengths: torch.Tensor,
        trg_input_ids: torch.Tensor,
        src_lang: str,
        trg_lang: str,
    ) -> torch.Tensor:
        """
        Teacher-forced sequence-to-sequence forward pass from src_lang to trg_lang.
        :param src_ids: (src_len, batch) LongTensor (including EOS at end of each sequence)
        :param src_lengths: (batch,) LongTensor
        :param trg_input_ids: (trg_len, batch) LongTensor (starting with SOS_ID)
        :return: logprobs of shape (trg_len, batch, trg_vocab_size + 4)
        """
        src_enc_emb, _, _, _, _ = self.get_lang_modules(src_lang)
        _, trg_dec_emb, trg_decoder, trg_generator, _ = self.get_lang_modules(trg_lang)

        enc_out, enc_hidden, enc_mask = self.encoder(src_ids, src_lengths, src_enc_emb)
        logprobs = trg_decoder(trg_input_ids, enc_hidden, enc_out, enc_mask, trg_dec_emb, trg_generator)
        return logprobs

    @torch.no_grad()
    def greedy_translate(
        self,
        src_ids: torch.Tensor,
        src_lengths: torch.Tensor,
        src_lang: str,
        trg_lang: str,
        max_ratio: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        On-the-fly greedy decoding from src_lang to trg_lang (matching Translator.greedy in undreamt/translator.py).
        Runs in eval() mode (no dropout) and returns (pred_ids_with_eos: (pred_len, batch), pred_lengths: (batch,)).
        """
        was_training = self.training
        self.eval()
        try:
            src_enc_emb, _, _, _, _ = self.get_lang_modules(src_lang)
            _, trg_dec_emb, trg_decoder, trg_generator, _ = self.get_lang_modules(trg_lang)

            max_src_len, batch_size = src_ids.shape
            device = src_ids.device

            enc_out, hidden, enc_mask = self.encoder(src_ids, src_lengths, src_enc_emb)
            prev_out = torch.zeros(batch_size, self.hidden_size, device=device, dtype=enc_out.dtype)
            cur_input = torch.full((batch_size,), SOS_ID, dtype=torch.long, device=device)

            max_trg_len = max(2, int(max_src_len * max_ratio))
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            pred_Lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
            generated_steps = []

            for step in range(max_trg_len):
                prev_out, hidden = trg_decoder.forward_step(
                    cur_input,
                    hidden,
                    prev_out,
                    enc_out,
                    enc_mask,
                    trg_dec_emb,
                )
                logprobs = trg_generator(prev_out)  # (batch, V+4)
                logprobs[:, PAD_ID] = -1e9
                logprobs[:, SOS_ID] = -1e9
                next_ids = torch.argmax(logprobs, dim=-1)  # (batch,)

                # Mask already-finished sequences with PAD_ID
                next_ids = torch.where(finished, torch.full_like(next_ids, PAD_ID), next_ids)
                generated_steps.append(next_ids)

                just_finished = (next_ids == EOS_ID) & (~finished)
                pred_Lengths = torch.where(just_finished, torch.full_like(pred_Lengths, step + 1), pred_Lengths)
                finished = finished | just_finished

                # Also enforce per-sentence max length (2 * src_len)
                over_len = ((step + 1) >= (src_lengths * 2).clamp(min=2)) & (~finished)
                if over_len.any():
                    generated_steps[-1] = torch.where(over_len, torch.full_like(next_ids, EOS_ID), generated_steps[-1])
                    pred_Lengths = torch.where(over_len, torch.full_like(pred_Lengths, step + 1), pred_Lengths)
                    finished = finished | over_len

                cur_input = generated_steps[-1]
                if finished.all():
                    break

            # Ensure all sequences end with EOS_ID
            if not finished.all():
                eos_col = torch.where(finished, torch.full_like(cur_input, PAD_ID), torch.full_like(cur_input, EOS_ID))
                generated_steps.append(eos_col)
                pred_Lengths = torch.where(finished, pred_Lengths, torch.full_like(pred_Lengths, len(generated_steps)))

            pred_ids = torch.stack(generated_steps, dim=0)  # (pred_len, batch)
            return pred_ids, pred_Lengths
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def beam_search_translate(
        self,
        src_ids: torch.Tensor,
        src_lengths: torch.Tensor,
        src_lang: str,
        trg_lang: str,
        beam_size: int = 12,
        max_ratio: float = 2.0,
        lp_alpha: float = 0.6,
    ) -> List[Tuple[List[int], float]]:
        """
        Beam search translation with Google NMT length penalty (matching Translator.beam_search in undreamt/translator.py):
          length_penalty(length, alpha) = (5 + length) ** alpha / (6 ** alpha)
        Returns list of (best_token_ids, best_score) for each sentence in batch.
        """
        self.eval()
        if beam_size <= 1:
            pred_ids, pred_lens = self.greedy_translate(src_ids, src_lengths, src_lang, trg_lang, max_ratio=max_ratio)
            results = []
            for b in range(src_ids.shape[1]):
                seq = pred_ids[: pred_lens[b], b].tolist()
                if seq and seq[-1] == EOS_ID:
                    seq = seq[:-1]
                results.append((seq, 0.0))
            return results

        src_enc_emb, _, _, _, _ = self.get_lang_modules(src_lang)
        _, trg_dec_emb, trg_decoder, trg_generator, vocab_total = self.get_lang_modules(trg_lang)

        enc_out, enc_hidden, enc_mask = self.encoder(src_ids, src_lengths, src_enc_emb)
        batch_size = src_ids.shape[1]
        device = src_ids.device

        def _lp(length: int) -> float:
            if lp_alpha <= 0.0:
                return 1.0
            return ((5.0 + length) ** lp_alpha) / (6.0 ** lp_alpha)

        batch_results: List[Tuple[List[int], float]] = []
        for b in range(batch_size):
            s_len = int(src_lengths[b].item())
            max_len = max(4, int(s_len * max_ratio))
            ctx_b = enc_out[:s_len, b : b + 1, :].repeat(1, beam_size, 1)  # (s_len, beam, H)
            mask_b = enc_mask[b : b + 1, :s_len].repeat(beam_size, 1)  # (beam, s_len)
            hid_b = enc_hidden[:, b : b + 1, :].repeat(1, beam_size, 1)  # (layers, beam, H)
            prev_out_b = torch.zeros(beam_size, self.hidden_size, device=device, dtype=enc_out.dtype)

            # Active hypotheses: list of (tokens, raw_logprob, beam_row_idx)
            beams = [([SOS_ID], 0.0, 0)]
            completed: List[Tuple[List[int], float]] = []

            for step in range(max_len):
                num_active = len(beams)
                in_ids = torch.tensor([h[0][-1] for h in beams], dtype=torch.long, device=device)
                row_indices = torch.tensor([h[2] for h in beams], dtype=torch.long, device=device)

                cur_hid = hid_b[:, row_indices, :]
                cur_prev_out = prev_out_b[row_indices, :]
                cur_ctx = ctx_b[:, :num_active, :]
                cur_mask = mask_b[:num_active, :]

                next_out, next_hid = trg_decoder.forward_step(
                    in_ids, cur_hid, cur_prev_out, cur_ctx, cur_mask, trg_dec_emb
                )
                logprobs = trg_generator(next_out)  # (num_active, V+4)
                logprobs[:, PAD_ID] = -1e9
                logprobs[:, SOS_ID] = -1e9

                topk_vals, topk_ids = torch.topk(logprobs, k=min(beam_size * 2, vocab_total), dim=-1)
                candidates = []
                for i, (toks, score, _) in enumerate(beams):
                    for k_idx in range(topk_ids.shape[1]):
                        tok_id = int(topk_ids[i, k_idx].item())
                        new_score = score + float(topk_vals[i, k_idx].item())
                        candidates.append((toks + [tok_id], new_score, i))

                # Sort candidates by normalized score
                candidates.sort(key=lambda c: c[1] / _lp(len(c[0]) - 1), reverse=True)
                new_beams = []
                for cand_toks, cand_score, parent_i in candidates:
                    if cand_toks[-1] == EOS_ID:
                        norm_score = cand_score / _lp(len(cand_toks) - 1)
                        completed.append((cand_toks[1:-1], norm_score))
                    else:
                        new_beams.append((cand_toks, cand_score, parent_i))
                        if len(new_beams) >= beam_size:
                            break

                if not new_beams or len(completed) >= beam_size:
                    break

                # Update hidden and prev_out buffers for new_beams
                sel = torch.tensor([nb[2] for nb in new_beams], dtype=torch.long, device=device)
                hid_b[:, : len(new_beams), :] = next_hid[:, sel, :]
                prev_out_b[: len(new_beams), :] = next_out[sel, :]
                beams = [(nb[0], nb[1], idx) for idx, nb in enumerate(new_beams)]

            if not completed:
                for toks, score, _ in beams:
                    completed.append((toks[1:], score / _lp(max(1, len(toks) - 1))))

            completed.sort(key=lambda x: x[1], reverse=True)
            batch_results.append(completed[0])

        return batch_results


Model = UNdreaMTModel


def get_model(
    num_inputs: Optional[int] = None,
    num_outputs: Optional[int] = None,
    **net_args,
) -> UNdreaMTModel:
    """
    RETURNN model factory entrypoint.
    """
    default_src = max(1, num_inputs - SPECIAL_SYMBOLS) if num_inputs is not None else 50000
    default_trg = max(1, num_outputs - SPECIAL_SYMBOLS) if num_outputs is not None else 50000
    src_vocab_size = net_args.pop("src_vocab_size", default_src)
    trg_vocab_size = net_args.pop("trg_vocab_size", default_trg)
    return UNdreaMTModel(
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        **net_args,
    )

