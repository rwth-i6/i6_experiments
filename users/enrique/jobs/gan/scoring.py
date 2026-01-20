from sisyphus import Job, Task, tk
import kenlm
from typing import Optional, Tuple, List

def score_line(line: str, model: kenlm.Model):

    # Score using KenLM (log10 probability) 
    return model.score(line)


def score_text(raw_text: str, model_path: str, symbols_to_remove: Optional[List[str]] = []):
    model = kenlm.Model(model_path)
    lines = raw_text.strip().split("\n")

    total_score = 0.0
    total_words = 0
    line_scores = []

    for line in lines:
        tokens = []
        for token in line.split():
            if token not in symbols_to_remove:
                tokens.append(token)
        cleaned_line = " ".join(tokens)
        if len(cleaned_line) == 0:
            #raise ValueError("Empty line encountered.")
            continue
        score = score_line(cleaned_line, model)
        if score is None:
            raise ValueError(f"Scoring failed for line: {cleaned_line}")

        line_scores.append((line, score))
        total_score += score
        total_words += len(tokens) + 1

    normalized_score = total_score / total_words if total_words > 0 else 0.0
    # Perplexity is 10^(-normalized_score) because KenLM uses log10
    perplexity = 10.0**(-normalized_score) if total_words > 0 else 0.0

    return total_score, line_scores, normalized_score, perplexity, line_scores

def vocab_used(raw_text: str, symbols_to_remove: Optional[List[str]] = None) -> float:
    used = set()
    for line in raw_text.strip().split("\n"):
        tokens = line.strip().split()
        for t in tokens:
            if symbols_to_remove and t in symbols_to_remove:
                continue
            
            used.add(t)
    return len(used), used


def unsupervised_metric_with_lm_and_vocab_usage(
    scores_and_vocab_usage: List[Tuple[str,Tuple[float, float]]],
) -> List[str]:

    results = []
    for identifier, (lm_score, vocab_usage) in scores_and_vocab_usage:
        result = f"ID: {identifier}, LM Score: {lm_score:.4f}, Vocab Used: {vocab_usage}"
        results.append(result)
    return results






class VocabUsageJob(Job):
    def __init__(
        self,
        text: tk.Path,
        symbols_to_remove: Optional[List[str]] = [],
    ):
        self.text = text
        self.symbols_to_remove = symbols_to_remove

        self.vocab_used_count = self.output_var("vocab_used_count")
        self.vocab_used_dict = self.output_path("vocab_used.txt")


    def tasks(self):
        yield Task("run",rqmt={"time": 1000, "cpu": 1, "mem": 4})

    def run(self):
        with open(self.text.get_path(), "r") as f:
            raw_text = f.read()

        vocab_used_count, used_vocab = vocab_used(
            raw_text,
            self.symbols_to_remove,
        )

        self.vocab_used_count.set(vocab_used_count)
        
        with open(self.vocab_used_dict.get_path(), "w") as f:
            for token in sorted(used_vocab):
                f.write(f"{token}\n")


class ScoreTextWithLMJob(Job):
    def __init__(
        self,
        text: tk.Path,
        kenlm_model_path: tk.Path,
        symbols_to_remove: Optional[List[str]] = [],
    ):
        self.text = text
        self.kenlm_model_path = kenlm_model_path
        self.symbols_to_remove = symbols_to_remove

        self.total_score = self.output_var("total_score")
        self.normalized_score = self.output_var("normalized_score")
        self.perplexity = self.output_var("perplexity")
        self.individual_line_scores = self.output_path("individual_line_scores.txt")


    def tasks(self):
        yield Task("run",rqmt={"time": 1000, "cpu": 1, "mem": 16})

    def run(self):
        with open(self.text.get_path(), "r") as f:
            raw_text = f.read()

        total_score, individual_line_scores, normalized_score, perplexity, line_scores = score_text(
            raw_text,
            self.kenlm_model_path.get_path(),
            self.symbols_to_remove,
        )

        self.total_score.set(total_score)
        self.normalized_score.set(normalized_score)
        self.perplexity.set(perplexity)

        with open(self.individual_line_scores.get_path(), "w") as f:
            for line, score in individual_line_scores:
                f.write(f"{score:.4f}\t{line}\n")




    

    

# --- Example Usage ---
if __name__ == "__main__":
    # Example raw phonemized text with SIL tokens
    sample_text = """
    <SIL> IH T IH Z D IH F AH K AH L T T UW B IH L IY V DH AE T AH F R EH N CH JH EH N ER AH L W IH DH AH B R IH L Y AH N T R EH K ER D B IH HH AY N D HH IH M SH UH D HH AE V B IH N G IH L T IY AH V S AH CH T R EH CH ER IY S AE K R AH F AY S IH NG HH IH Z M EH N AH N D HH IH Z HH AH N ER <SIL>
    <SIL> HH IH Z F R EH N D Z DH EY W ER N AA T M EH N IY S EY HH IY L AO S T HH IH Z HH EH D W AA Z N IH R L IY K R EY Z IY W IH DH DH AH AH T ER L IY AH N F AO R S IY N D IH F IY T AH V DH AH F R EH N CH B AH T IY V IH N AH M OW M AH N T AH V IH N S AE N AH T IY W UH D HH AA R D L IY AH K AW N T F AO R S AH CH IH K S T R AO R D AH N EH R IY W IY K N AH S <SIL>
    """

    example_model = "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_06_02/work/i6_experiments/users/enrique/experiments/gan/prepare_text/TrainKenLMJob.sZAeRaQX0IsV/output/kenlm.o4.bin"

    total, individual = score_text(sample_text, example_model)
    
    print(f"Total Log10 Probability: {total}")
    for line, score in individual:
        print(f"Score: {score:.4f} | Line: {line.strip()}")
  