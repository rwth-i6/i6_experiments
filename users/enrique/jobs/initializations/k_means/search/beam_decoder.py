import torch
import numpy as np

class Beam:
    """
    A class to store beam information: the cumulative log probability
    and the sequence of tokens (history).
    """
    def __init__(self, logprob, history):
        self.logprob = logprob
        self.history = history

    def __repr__(self):
        # A helper for debugging
        return f"Beam(logprob={self.logprob:.4f}, history={self.history})"

class BeamDecoder:
    """
    A Viterbi-like decoder for ASR that uses an n-gram language model and
    includes a top-k beam search to prune the search space.

    Using infinite beam size and a 2 gram LM is equivalent to a Viterbi search.
    """

    def __init__(self, vocab_tokens, language_model, acoustic_model, beam_size=10,  lm_scale=1.0, am_scale=1.0):
        """
        Initializes the decoder.
        """
        self.vocab_tokens = vocab_tokens
        self.vocab_size = len(vocab_tokens)
        self.vocab_mapping = {idx: token for idx, token in enumerate(vocab_tokens)}
        self.lm = language_model
        self.am = acoustic_model
        self.beam_size = beam_size

        self.lm_scale = lm_scale
        self.am_scale = am_scale

        # Use -1 as a convention for a padding/start token if needed,
        # but the history will be managed explicitly.
        # Make sure your token IDs are non-negative.
        self.START_TOKEN_ID = 0 # A symbolic start, not in vocab

        # Use PyTorch for computations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")


    def decode(self, observations):
        """
        Decodes a sequence of observations using the Viterbi algorithm with
        beam search.
        """
        T = len(observations)
        if T == 0:
            return [], 0.0

        # --- Initialization (t=0) ---
        # Start with a single beam with logprob=0 and an empty history.
        # The history will be built up. For the LM, we can pad with start tokens.
        lm_context_size = self.lm.get_context_size()
        initial_history = [self.START_TOKEN_ID] * lm_context_size
        beams = [Beam(0.0, initial_history)]

        # --- Main Loop over Observations ---
        for t in range(T):
            # At each step, we expand the current beams and then prune them.
            
            # --- 1. Get Acoustic Model Scores ---
            # This can be pre-computed for the timestep for all tokens.
            am_log_probs = torch.tensor(
                [self.am.get_log_prob(observations[t], token_id) for token_id in range(self.vocab_size)],
                dtype=torch.float32,
                device=self.device
            ) # Shape: (vocab_size,)

            # --- 2. Build Candidate Score Matrix ---
            # We want to create a matrix of shape (current_beam_size, vocab_size)
            # where entry (i, j) is the total score of extending beam i with token j.
            
            # Collect scores and histories from the current set of beams
            current_beam_logprobs = torch.tensor([b.logprob for b in beams], device=self.device) # Shape: (current_beam_size,)
            
            # Get LM scores for all current beams and all possible next tokens
            lm_log_probs = torch.zeros(len(beams), self.vocab_size, device=self.device)
            for i, beam in enumerate(beams):
                # This loop over beams is often necessary if the LM isn't batchable.
                # If your LM could take a batch of histories, this could be vectorized further.
                lm_log_probs[i] = torch.tensor(
                    [self.lm.get_log_prob(token_id, beam.history[lm_context_size+1:]) for token_id in range(self.vocab_size)],
                    device=self.device
                )

            # Now, calculate the total scores for all new candidate paths
            # Broadcasting adds the scores correctly:
            # (current_beam_size, 1) + (current_beam_size, vocab_size) + (vocab_size,)
            # -> (current_beam_size, vocab_size)
            candidate_scores = current_beam_logprobs.unsqueeze(1) + self.lm_scale * lm_log_probs + self.am_scale * am_log_probs
            
            # On the first step, len(beams) is 1, so candidate_scores is (1, vocab_size).
            # We need to ensure we select 'beam_size' candidates.
            candidate_scores_n_elements = len(beams) * self.vocab_size
            effective_beam_size = self.beam_size if self.beam_size < candidate_scores_n_elements else candidate_scores_n_elements

            # --- 3. Find Global Top-K Candidates ---
            # Flatten the scores and find the top 'k' scores and their original indices.
            topk_scores, topk_flat_indices = torch.topk(candidate_scores.view(-1), k=effective_beam_size)
            
            # Convert the flat indices back into (beam_idx, token_idx) pairs
            beam_indices = topk_flat_indices // self.vocab_size
            token_indices = topk_flat_indices % self.vocab_size

            # --- 4. Construct the New Beams ---
            new_beams = []
            for i in range(effective_beam_size):
                score = topk_scores[i].item()
                beam_idx = beam_indices[i].item()
                token_idx = token_indices[i].item()

                parent_beam = beams[beam_idx]
                
                new_history = parent_beam.history[:] + [token_idx] if t > 0 else [token_idx]

                new_beams.append(Beam(logprob=score, history=new_history))
            
            del beams
            del candidate_scores
            del am_log_probs
            del lm_log_probs

            beams = new_beams


        best_beam = max(beams, key=lambda b: b.logprob)
        
        return best_beam.history, best_beam.logprob

