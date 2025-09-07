import torch
import numpy as np

class ViterbiASR:
    """
    A Viterbi decoder for ASR that uses an n-gram language model and
    includes a top-k beam search to prune the search space.

    The implementation works with log probabilities for numerical stability.
    """

    def __init__(self, vocab, language_model, acoustic_model, beam_size=10):
        """
        Initializes the decoder.

        Args:
            vocab (list): A list of all words in the vocabulary.
            language_model: A user-provided object that has a method
                            `get_log_prob(word, history)` which returns the
                            n-gram log probability.
            acoustic_model: A user-provided object that has a method
                            `get_log_prob(sound_observation, word)` which
                            returns the log probability of an observation
                            given a word (log q(x|y)).
            beam_size (int): The number of best paths to keep at each step (k).
        """
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.word_to_idx = {word: i for i, word in enumerate(vocab)}
        self.idx_to_word = {i: word for i, word in enumerate(vocab)}

        self.lm = language_model
        self.am = acoustic_model
        self.k = beam_size
        self.START_TOKEN = "<s>"
        self.END_TOKEN = "</s>"


    def decode(self, observations):
        """
        Finds the most likely sequence of words for a given sequence of observations.

        Args:
            observations (list): A list of sound observations (e.g., feature vectors).

        Returns:
            list: The most likely sequence of words.
            float: The log probability of the best path.
        """
        T = len(observations)
        if T == 0:
            return [], 0.0

        # --- Data Structures ---
        # viterbi_matrix[t, j] stores the max log-prob of a path ending in word j at time t
        viterbi_matrix = torch.full((T, self.vocab_size), -float('inf'))

        # backpointers[t, j] stores the index of the best previous word on the path to j at t
        backpointers = torch.zeros((T, self.vocab_size), dtype=torch.long)
        
        # active_paths[t] will store the indices of the k-best states from time t-1
        # This is the core of our beam search.
        active_paths = torch.zeros(self.k, dtype=torch.long)


        # --- 1. Initialization Step (t = 0) ---
        # For the first observation, the history is just the start token.
        history = (self.START_TOKEN,)
        for j in range(self.vocab_size):
            word = self.idx_to_word[j]
            lm_log_prob = self.lm.get_log_prob(word, history)
            am_log_prob = self.am.get_log_prob(observations[0], word)
            viterbi_matrix[0, j] = lm_log_prob + am_log_prob

        # Prune the first step to the beam size
        top_k_probs, top_k_indices = torch.topk(viterbi_matrix[0], self.k)
        
        # Inactivate all other paths
        inactive_indices = torch.ones(self.vocab_size, dtype=torch.bool)
        inactive_indices[top_k_indices] = False
        viterbi_matrix[0, inactive_indices] = -float('inf')
        
        active_paths = top_k_indices


        # --- 2. Recursion Step (t = 1 to T-1) ---
        for t in range(1, T):
            # We create a temporary matrix to hold candidate scores
            candidate_scores = torch.full((self.k, self.vocab_size), -float('inf'))
            
            # This will store the backpointer for each candidate
            candidate_backpointers = torch.zeros((self.k, self.vocab_size), dtype=torch.long)

            # Iterate only over the active paths (words) from the previous step
            for i, prev_word_idx in enumerate(active_paths):
                # To get the n-gram probability, we need the history.
                # We reconstruct the history for the current active path.
                # Note: For a more efficient implementation with large n-grams,
                # you might store histories along with scores.
                # Here, we reconstruct it by backtracking.
                temp_path = [prev_word_idx.item()]
                for time in range(t - 1, 0, -1):
                    prev_idx = backpointers[time, temp_path[-1]].item()
                    temp_path.append(prev_idx)
                
                history_indices = reversed(temp_path)
                history = (self.START_TOKEN,) + tuple(self.idx_to_word[idx] for idx in history_indices)

                # Now calculate the score for transitioning to each next word
                for j in range(self.vocab_size):
                    word = self.idx_to_word[j]
                    
                    # The transition score depends on the LM and the previous Viterbi score
                    lm_log_prob = self.lm.get_log_prob(word, history)
                    transition_score = viterbi_matrix[t-1, prev_word_idx] + lm_log_prob
                    
                    # The full score includes the acoustic model's observation probability
                    am_log_prob = self.am.get_log_prob(observations[t], word)
                    score = transition_score + am_log_prob
                    
                    candidate_scores[i, j] = score
                    candidate_backpointers[i, j] = prev_word_idx

            # Now, for each current word j, find the best path leading to it
            # from the k active previous paths.
            viterbi_matrix[t], best_prev_indices_for_j = torch.max(candidate_scores, dim=0)
            
            # Use the indices to get the correct backpointers
            for j in range(self.vocab_size):
                backpointers[t, j] = candidate_backpointers[best_prev_indices_for_j[j], j]

            # Prune the current step to the beam size
            top_k_probs, top_k_indices = torch.topk(viterbi_matrix[t], self.k)
            inactive_indices = torch.ones(self.vocab_size, dtype=torch.bool)
            inactive_indices[top_k_indices] = False
            viterbi_matrix[t, inactive_indices] = -float('inf')
            active_paths = top_k_indices


        # --- 3. Termination ---
        # Find the best path ending at the last time step
        best_path_prob, last_word_idx = torch.max(viterbi_matrix[T-1, :], dim=0)
        last_word_idx = last_word_idx.item()


        # --- 4. Backtracking ---
        best_path = [self.idx_to_word[last_word_idx]]
        current_word_idx = last_word_idx

        for t in range(T - 1, 0, -1):
            prev_word_idx = backpointers[t, current_word_idx].item()
            best_path.insert(0, self.idx_to_word[prev_word_idx])
            current_word_idx = prev_word_idx

        return best_path, best_path_prob.item()


# =============================================================================
# EXAMPLE USAGE: You need to provide your own models for this to run.
# =============================================================================

class MockLanguageModel:
    """
    A dummy Language Model for demonstration.
    This model should be replaced with your actual n-gram LM.
    It gives a higher probability to sequences of repeating words.
    """
    def __init__(self, vocab):
        self.vocab = vocab
        # Assign a uniform log probability as a base
        self.unigram_log_prob = np.log(1.0 / len(vocab))

    def get_log_prob(self, word, history):
        # A simple logic: prefer repeating the last word in the history
        if len(history) > 1 and word == history[-1]:
            return self.unigram_log_prob + np.log(2.0) # Boost probability
        return self.unigram_log_prob


class MockAcousticModel:
    """
    A dummy Acoustic Model for demonstration.
    This model should be replaced with your actual q(x|y) provider.
    It assumes a simple mapping where observation 's_i' corresponds to 'word_i'.
    """
    def __init__(self, vocab):
        self.word_to_idx = {word: i for i, word in enumerate(vocab)}

    def get_log_prob(self, sound_observation, word):
        # Simple logic: if sound 's_i' matches 'word_i', prob is high, else low.
        # e.g., sound_observation could be "sound_for_hello"
        expected_word = sound_observation.split('_')[-1]
        if word == expected_word:
            return np.log(0.8)  # High probability
        else:
            return np.log(0.1)  # Low probability


if __name__ == '__main__':
    # --- Setup ---
    vocabulary = ["hello", "world", "speech", "recognition"]
    
    # These are your observations (sounds)
    # In a real scenario, these would be acoustic feature vectors.
    observations = [
        "sound_for_hello",
        "sound_for_hello", # A "mistake" where "hello" is repeated
        "sound_for_world",
        "sound_for_recognition"
    ]

    # Initialize your models
    mock_lm = MockLanguageModel(vocabulary)
    mock_am = MockAcousticModel(vocabulary)

    # --- Decoding ---
    # Initialize the Viterbi decoder with a beam size of 2
    decoder = ViterbiASR(
        vocab=vocabulary,
        language_model=mock_lm,
        acoustic_model=mock_am,
        beam_size=2
    )

    # Run the decoding process
    best_sequence, best_log_prob = decoder.decode(observations)

    print(f"Observations: {observations}")
    print("-" * 30)
    print(f"Vocabulary: {vocabulary}")
    print(f"Beam Size (k): {decoder.k}")
    print("-" * 30)
    print(f"Most Likely Word Sequence: {best_sequence}")
    print(f"Best Path Log Probability: {best_log_prob:.4f}")

    # Example of how the mock models influence the result:
    # The acoustic model strongly prefers the sequence:
    # ['hello', 'hello', 'world', 'recognition']
    # The language model prefers repeating words, so it also likes the two 'hello's.
    # The combination should find this path.

