from sisyphus import *

from typing import Optional

# from nemo.collections.common.tokenizers.canary_tokenizer import CanaryTokenizer

from i6_experiments.users.zeyer.decoding.beam_search_torch.tools.end_prune_stats import open_job_log


class ComputePruningStatsJob(Job):
    def __init__(self, search_out: tk.Path):
        self.search_out = search_out
        # self.tokenzier = tokenizer

        self.out_num_seqs = self.output_var("num_seqs")
        self.out_num_steps = self.output_var("num_steps")
        self.out_avg_orth_len = self.output_var("avg_orth_len")
        self.out_avg_num_steps_per_seq = self.output_var("avg_num_steps_per_seq")
        self.out_max_act_hyps = self.output_var("max_act_hyps")
        self.out_avg_num_act_hyps_per_step = self.output_var("avg_num_act_hyps_per_step")
        self.out_avg_end_diff_to_orth = self.output_var("avg_end_diff_to_orth")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        # .../output/search_out
        jobdir = "/".join(self.search_out.get_path().split("/")[:-2])
        with open_job_log(jobdir) as (log, log_fn, job_dir):
            job_log = log.read().strip().splitlines()

        # e.g format:
        # {"audio_filepath": "sample_1", "duration": 0.0, "text": "yeah we are going to meet up", "pred_text": "yeah we were going to meet up"}
        # total_len_orth = 0
        # with open(self.search_out.get_path(), "r") as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         ref = eval(line)["text"]
        #         ref_pieces = self.tokenzier.text_to_tokens(ref, lang_id="en")
        #         total_len_orth += len(ref_pieces)

        prev_step: Optional[int] = None
        total_num_seqs = 0
        total_num_seqs_finished = 0
        total_act_num_steps = 0
        total_num_steps = 0
        total_act_hyps = 0
        max_act_hyps = 0
        for line in job_log:
            if "DEBUG: " in line:
                start_idx = line.find("DEBUG: ")
                line = line[start_idx:]
                line = line[len("DEBUG: ") :]

                # if line.startswith("DEBUG: "):
                #     line = line[len("DEBUG: ") :]
                content = eval(f"dict({line})")
                step = content["step"]
                act_beam_sizes = content["act_beam_sizes"]
                assert (
                    isinstance(step, int)
                    and isinstance(act_beam_sizes, list)
                    and all(isinstance(x, int) for x in act_beam_sizes)
                )
                assert (
                    (prev_step is not None) and (step == prev_step + 1)
                ) or step == 0, f"prev_step: {prev_step}, step: {step}"
                if step == 0:
                    total_num_seqs += len(act_beam_sizes)
                    max_act_hyps = max(max_act_hyps, max(act_beam_sizes))

                for i, size in enumerate(act_beam_sizes):
                    # if size == 0:  # finished now
                    #     total_num_seqs_finished += 1
                    #     orth = seq.orth
                    #     orth_pieces = sp.encode(orth, out_type=str)
                    #     total_len_orth += len(orth_pieces)
                    #     cur_seqs[i] = None
                    #     continue
                    if size == 0:
                        continue
                    total_num_steps += 1
                    if step > 0:
                        total_act_num_steps += 1
                        total_act_hyps += size
                prev_step = step

        self.out_num_seqs.set(total_num_seqs)
        self.out_num_steps.set(total_num_steps)
        # self.out_avg_orth_len.set(total_len_orth / total_num_seqs)
        self.out_avg_num_steps_per_seq.set(total_num_steps / total_num_seqs)
        self.out_max_act_hyps.set(max_act_hyps)
        self.out_avg_num_act_hyps_per_step.set(total_act_hyps / total_act_num_steps)
        # self.out_avg_end_diff_to_orth.set((total_num_steps - total_len_orth) / total_num_seqs)
