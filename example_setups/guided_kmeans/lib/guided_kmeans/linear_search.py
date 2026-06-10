import math
from typing import Dict, List, Tuple
import timeit

import torch
import kenlm  # KenLM python bindings

def kenlm_base_score(
    model: "kenlm.Model",
    state: "kenlm.State",
    word: str
) -> Tuple[float, "kenlm.State"]:
    """
    Query KenLM for the incremental score of 'word' given 'state'.
    Returns (ln_prob, new_state), where ln_prob is a natural-log score.
    KenLM stores log10 probabilities internally → convert to natural log.
    """
    new_state = kenlm.State()
    # BaseScore returns log10 probability
    log10p = model.BaseScore(state, word, new_state)
    return (log10p * math.log(10.0), new_state)


def linear_search_decode_kenlm(
    am_logprobs: torch.Tensor,
    labels: List[str],
    lm_model: "kenlm.Model",
    lm_scale: float = 1.0,
    bos_token: str = "<s>",
) -> Dict:
    """
    Full linear DP search with KenLM for LM scoring.
    am_logprobs: [T, M] tensor of acoustic log-probs (natural log).
    labels: M tokens matching LM vocabulary (OOV words must be in KenLM vocab).
    lm_model: kenlm.Model loaded on disk.
    lm_scale: scaling factor for LM log-probabilities.
    """

    T, M = am_logprobs.shape
    assert M == len(labels), "Mismatch between labels and AM outputs"

    # For histories, we keep (kenlm.State, history_tokens) tuples
    # history_tokens are stored for backtrace only (not needed by KenLM itself).
    hist_len = lm_model.order - 1

    # Initialize state: start at BOS context
    init_state = kenlm.State()
    lm_model.BeginSentenceWrite(init_state)

    # DP tables: current states → best score
    curr: Dict[Tuple[int, ...], Tuple[float, "kenlm.State"]] = {
        tuple(): (0.0, init_state)
    }

    backptr: List[Dict[Tuple[int, ...], Tuple[Tuple[int, ...], int, float, float, float]]] = []

    for t in range(T):
        next_states: Dict[Tuple[int, ...], Tuple[float, "kenlm.State"]] = {}
        next_bp = {}

        frame = am_logprobs[t]

        for (hist_idx, (cum_score, ken_state)) in curr.items():
            for y in range(M):
                am = frame[y].item()
                token = labels[y]

                # LM score given current state
                lm_lp, new_state = kenlm_base_score(lm_model, ken_state, token)
                lm_lp_scaled = lm_scale * lm_lp

                score = cum_score + am + lm_lp_scaled

                # Build new history indices (keep ints for backtrace)
                new_hist_idx = (*hist_idx, y)[-hist_len:] if hist_len > 0 else tuple()

                # Keep best
                existing = next_states.get(new_hist_idx)
                if existing is None or score > existing[0]:
                    next_states[new_hist_idx] = (score, new_state)
                    next_bp[new_hist_idx] = (hist_idx, y, am, lm_lp_scaled, score)

        curr = next_states
        backptr.append(next_bp)

    # Find best ending
    best_hist = max(curr.items(), key=lambda kv: kv[1][0])[0]
    best_score = curr[best_hist][0]

    # Traceback
    seq_idxs = [0] * T
    per_step = [None] * T
    h = best_hist
    for t in reversed(range(T)):
        prev_h, y, am, lm_lp_scaled, cum = backptr[t][h]
        seq_idxs[t] = y
        per_step[t] = {
            "t": t,
            "label": labels[y],
            "am": am,
            "lm": lm_lp_scaled,
            "cum": cum,
        }
        h = prev_h

    seq_labels = [labels[i] for i in seq_idxs]
    return {
        "labels": seq_labels,
        "indices": seq_idxs,
        "per_step": per_step,
        "total_score": best_score,
    }

def linear_search_decode_hmm_label_dependent(
    am_logprobs: torch.Tensor,               # [T, M]
    labels: List[str],
    lm_model: "kenlm.Model",
    lm_scale: float = 1.0,
    stay_logprob: Dict[str, float] | None = None,
    step_logprob: Dict[str, float] | None = None,
    bos_token: str = "<s>"
) -> Dict:
    """
    Decoder with label-dependent HMM transitions.
    - stay_logprob[label]: log-probability of staying in that phoneme
    - step_logprob[label]: log-probability of stepping from that phoneme
    LM score is applied only on a step.
    """

    T, M = am_logprobs.shape
    assert M == len(labels)

    # Default: uniform loops/steps if not provided
    if stay_logprob is None:
        stay_logprob = {lab: math.log(0.5) for lab in labels}
    if step_logprob is None:
        step_logprob = {lab: math.log(0.5) for lab in labels}

    init_state = kenlm.State()
    lm_model.BeginSentenceWrite(init_state)

    # DP keyed by (LM history tuple, current label index)
    curr: Dict[Tuple[Tuple[str, ...], int | None],
              Tuple[float, "kenlm.State"]] = {
        ((bos_token,), None): (0.0, init_state)
    }
    backptr: List[Dict] = []

    for t in range(T):
        frame = am_logprobs[t]
        next_dp: Dict[Tuple[Tuple[str, ...], int],
                     Tuple[float, "kenlm.State"]] = {}
        next_bp: Dict = {}

        for (hist, curr_idx), (cum_score, ken_state) in curr.items():
            for y in range(M):
                am = frame[y].item()
                token = labels[y]

                # Step (new phoneme): apply label-dependent step logprob
                (lm_lp, new_state) = kenlm_base_score(lm_model, ken_state, token)
                lm_lp_scaled = lm_scale * lm_lp
                log_step = step_logprob.get(token, math.log(1e-9))

                step_score = cum_score + am + log_step + lm_lp_scaled
                new_hist = (*hist, token)[- max(lm_model.order - 1, 0):]
                key_step = (new_hist, y)

                existing = next_dp.get(key_step)
                if existing is None or step_score > existing[0]:
                    next_dp[key_step] = (step_score, new_state)
                    next_bp[key_step] = ((hist, curr_idx), y, am, lm_lp_scaled, log_step, step_score)

            # Stay (loop): only if we have a current phoneme
            if curr_idx is not None:
                curr_label = labels[curr_idx]
                log_stay = stay_logprob.get(curr_label, math.log(1e-9))

                am = frame[curr_idx].item()
                stay_score = cum_score + am + log_stay
                key_stay = (hist, curr_idx)

                existing = next_dp.get(key_stay)
                if existing is None or stay_score > existing[0]:
                    next_dp[key_stay] = (stay_score, ken_state)
                    next_bp[key_stay] = ((hist, curr_idx), curr_idx, am, 0.0, log_stay, stay_score)

        backptr.append(next_bp)
        curr = next_dp

    best_state = max(curr.items(), key=lambda kv: kv[1][0])[0]
    best_score = curr[best_state][0]

    seq_idx: List[int] = [0]*T
    per_step: List[Dict] = [None]*T
    st = best_state

    for t in reversed(range(T)):
        prev, label_idx, am, lm_lp_scaled, trans_lp, cum = backptr[t][st]
        seq_idx[t] = label_idx
        per_step[t] = {
            't': t,
            'label': labels[label_idx] if label_idx is not None else None,
            'am': am,
            'lm': lm_lp_scaled,
            'transition': trans_lp,
            'cum': cum
        }
        st = prev

    seq_labels = [labels[i] for i in seq_idx]
    return {
        "labels": seq_labels,
        "indices": seq_idx,
        "per_step": per_step,
        "total_score": best_score,
    }

# ————————————————————————————————————————————
# Vectorized Batched Decoder with HMM + KenLM LM
# ————————————————————————————————————————————

def batch_kenlm_adv_scores(
    lm: "kenlm.Model",
    hist_states: List[Tuple[Tuple[str, ...], "kenlm.State"]],
    next_labels: List[str],
    lm_scale: float,
):
    """
    Score all next_labels for each sequence in the batch using KenLM in a batched loop.
    hist_states: [(history_token_tuple, kenlm.State), ...] length batch.
    next_labels: full label list of length M.
    Returns:
        lm_scores: torch tensor of shape [batch, M] in natural-log
        new_states: nested list length [batch][M] of new kenlm.States
    """
    B = len(hist_states)
    M = len(next_labels)
    lm_scores = torch.zeros(B, M)
    new_states = [[None]*M for _ in range(B)]

    for b in range(B):
        hist_tokens, base_state = hist_states[b]
        # Rebuild KenLM state from history
        state = kenlm.State()
        lm.BeginSentenceWrite(state)
        for tok in hist_tokens:
            # _lp, new_s = lm.BaseScore(state, tok, state)
            new_state = kenlm.State()
            log10p = lm.BaseScore(state, tok, new_state)
            # lm_lp = log10p * math.log(10.0)  # convert to natural log
            # new_s = ken
        # Score every next label
        for j, lbl in enumerate(next_labels):
            st2 = kenlm.State()
            log10p = lm.BaseScore(state, lbl, kenlm.State())
            lm_scores[b, j] = log10p * math.log(10.0) * lm_scale
            new_states[b][j] = st2
    return lm_scores, new_states


def vectorized_hmm_decode_batched(
    am_logp: torch.Tensor,
    labels: List[str],
    lm_model: "kenlm.Model",
    stay_logprob: Dict[str, float] | None = None,
    step_logprob: Dict[str, float] | None = None,
    lm_scale: float = 1.0,
    bos_token: str = "<s>"
):
    """
    Full batched vectorized decoder with label-dependent stay/step transitions.
    Inputs:
      am_logp: tensor [batch, T, M] (natural log)
      labels: list of M phoneme strings matching label order
      lm_model: a KenLM n-gram model loaded in Python
      stay_logprob: dict mapping label->loop logprob
      step_logprob: dict mapping label->step logprob
      lm_scale: scale applied to LM log probabilities
    """
    B, T, M = am_logp.shape
    DEVICE = am_logp.device

    # Initialize KenLM states and histories for each batch
    init_states: List[Tuple[Tuple[str, ...], kenlm.State]] = []
    for _ in range(B):
        s0 = kenlm.State()
        lm_model.BeginSentenceWrite(s0)
        init_states.append(((bos_token,), s0))
    
    # Default: uniform loops/steps if not provided
    if stay_logprob is None:
        stay_logprob = {lab: math.log(0.5) for lab in labels}
    if step_logprob is None:
        step_logprob = {lab: math.log(0.5) for lab in labels}

    # DP tensors for scores
    # best_score[b, j] = best path score at time t for batch b and label j
    best_scores = torch.full((B, M), -1e9, device=DEVICE)
    prev_best: Dict = {}  # for backtrace storage at each time
    backpointers: List[Dict] = []

    # First timestep: only transitions (no stay yet) from dummy start
    lm_scores0, states0 = batch_kenlm_adv_scores(
        lm_model, init_states, labels, lm_scale
    )
    # am_logp[:, 0, :] + step_logprob (vector form)
    step_vec = torch.tensor([step_logprob[l] for l in labels], device=DEVICE)
    best_scores = am_logp[:, 0, :] + step_vec + lm_scores0

    # Store initial backpointer placeholders
    backpointers.append({(b, j): (None, j) for b in range(B) for j in range(M)})
    curr_states = states0  # list of states for each b, j

    # DP over frames
    for t in range(1, T):
        frame = am_logp[:, t, :]  # [B, M]

        # ==== Step candidates ====
        # Build histories for LM from best prev labels
        # curr_hist[b][j] = history tokens for batch b and label j
        batch_histories = []
        for b in range(B):
            # we pull previous best label for starting history
            hist_tokens, _ = init_states[b]
            batch_histories.append((tuple(hist_tokens), None))

        lm_scores_step, new_states_step = batch_kenlm_adv_scores(
            lm_model, batch_histories, labels, lm_scale
        )

        step_vec = torch.tensor([step_logprob[l] for l in labels], device=DEVICE)
        cand_step = best_scores.unsqueeze(2) + frame.unsqueeze(1) \
                    + step_vec.unsqueeze(0).unsqueeze(1) + lm_scores_step.unsqueeze(1)
        # cand_step[b, i, j] = score stepping from i->j

        # Max over i dimension
        step_max, step_arg = torch.max(cand_step, dim=1)

        # ==== Stay candidates ====
        stay_vec = torch.tensor([stay_logprob[l] for l in labels], device=DEVICE)
        cand_stay = best_scores + frame + stay_vec.unsqueeze(0)

        # ==== Combine stay vs step ====
        combined = torch.stack([cand_stay, step_max], dim=-1)
        best_scores, which = torch.max(combined, dim=-1)

        # Record backpointers
        bp_t = {}
        for b in range(B):
            for j in range(M):
                if which[b, j] == 0:
                    # stay: prev label index is same j
                    bp_t[(b, j)] = (j, j)
                else:
                    prev_lab = step_arg[b, j].item()
                    bp_t[(b, j)] = (prev_lab, j)
        backpointers.append(bp_t)

    # ==== Traceback ====
    best_final = torch.argmax(best_scores, dim=1).tolist()
    sequences = [[None]*T for _ in range(B)]

    for b in range(B):
        cur_label = best_final[b]
        for t in reversed(range(T)):
            _, lab_out = backpointers[t][(b, cur_label)]
            sequences[b][t] = labels[lab_out]
            cur_label = backpointers[t][(b, cur_label)][0]

    return sequences, best_scores.cpu().tolist()

def main():
    lm_path = "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/data/2gram/output/phones/lm.phones.filtered.02.arpa"
    model = kenlm.Model(lm_path)

    phones_path = "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/i6_experiments/users/enrique/jobs/fairseq/wav2vec/wav2vec_data_utils/PrepareWav2VecTextDataJob.RZfllsI3R2Pd/output/text/phones/dict.txt"

    labels = []
    with open(phones_path, "rt") as fp:
        for line in fp:
            phone = line.split(" ")[0]
            labels.append(phone)

    # Fake acoustic log-probs
    # labels = ["AA","AE","AH","B","K","<unk>"]
    T, M = 1000, len(labels)
    torch.manual_seed(0)
    am_logprobs = torch.log_softmax(torch.randn(T, M), dim=-1)

    start_time = timeit.default_timer()
    result = linear_search_decode_hmm_label_dependent(
        am_logprobs, labels, model, lm_scale=1.0, bos_token="<s>",
    )
    end_time = timeit.default_timer()

    print("Best sequence:", result["labels"])
    print("Score:", result["total_score"])
    print(f"Time: {end_time - start_time}s")

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Load a binary or ARPA LM with KenLM
    # lm_path = "phoneme_lm.arpa"  # or .bin
    main()
    # for s in result["per_step"]:
    #     print(s)

    # am_logprobs_batch = am_logprobs.unsqueeze(0)
    # seq, scores = vectorized_hmm_decode_batched(
    #     am_logprobs_batch, labels, model, lm_scale=1.0, bos_token="<s>",
    # )
    # print("Best sequence:", seq)
    # print("Score:", scores)
    # for s in result["per_step"]:
    #     print(s)
