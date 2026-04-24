from pathlib import Path
from sisyphus import Job, Task, tk
import os
from .common import HF_CACHE_DIR, vllm_server
from datasets import (
    load_dataset,
    load_from_disk,
    Dataset,
    DatasetDict,
    concatenate_datasets,
)
from openai import OpenAI
import json
from i6_experiments.users.dorian_koch.jobs.hf import (
    HfDownloadSplit as implHfDownloadSplit,
    HfMergeShards as implHfMergeShards,
)


class HfDownloadSplit(implHfDownloadSplit):
    pass  # Epic hack to not break hashes


class HfMergeShards(implHfMergeShards):
    pass


def make_dialogue_gen(llm_url, model_name, dialogue_instructions):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "nothing"),
        base_url=llm_url,
    )
    client.models.list()

    def make_dialogue(example):
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "nothing"),
            base_url=llm_url,
        )
        # features: ['question_id', 'question', 'options', 'answer', 'answer_index', 'cot_content', 'category', 'src'],

        # {'question_id': 70, 'question': 'Typical advertising regulatory bodies suggest, for example that adverts must not: encourage _________, cause unnecessary ________ or _____, and must not cause _______ offence.', 'options': ['Safe practices, Fear, Jealousy, Trivial', 'Unsafe practices, Distress, Joy, Trivial', 'Safe practices, Wants, Jealousy, Trivial', 'Safe practices, Distress, Fear, Trivial', 'Unsafe practices, Wants, Jealousy, Serious', 'Safe practices, Distress, Jealousy, Serious', 'Safe practices, Wants, Fear, Serious', 'Unsafe practices, Wants, Fear, Trivial', 'Unsafe practices, Distress, Fear, Serious'], 'answer': 'I', 'answer_index': 8, 'cot_content': '', 'category': 'business', 'src': 'ori_mmlu-business_ethics'}

        question = example["question"]
        options = example["options"]
        answer = example["answer"]
        cot = example["cot_content"] or "No background information available."

        options_text = ""
        for idx, option in enumerate(options):
            options_text += f"{chr(65 + idx)}: {option}\n"
        answer_text = f"{chr(65 + example['answer_index'])}: {answer}"

        user_msg = (
            "Background information: " + cot + "\n"
            "Here is the question: " + question + "\n"
            "The possible answers are: "
            + options_text
            + "The correct answer is: "
            + answer_text
            + ".\n"
        )  # TODO do we really want to give the agent so many different answers?

        user_msg += dialogue_instructions

        messages = [
            {"role": "user", "content": user_msg},
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            seed=example["question_id"],
            response_format={"type": "json_object"},
        )

        prediction = response.choices[0].message.content

        return {"dialogue": prediction}

    return make_dialogue


class HfToDialogue(Job):
    def __init__(
        self,
        *,
        dataset_split_path: tk.Path,
        llm_name: str,
        dialogue_instructions: str,
        shard: int | None = None,
        num_shards: int | None = None,
    ):
        self.dataset_split_path = dataset_split_path
        self.llm_name = llm_name
        self.dialogue_instructions = dialogue_instructions
        self.shard = shard
        self.num_shards = num_shards

        self.out_hf = self.output_path("dialogue_dataset", directory=True)
        self.out_json = self.output_path("json_files")
        self.rqmt = {
            "gpu": 1,
            "cpu": 2,
            "mem": 16,
            "time": 2,
        }

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 2
        return super().hash(d)

    @staticmethod
    def sharded(*, num_shards: int, **kwargs):
        if num_shards == 1:
            return HfToDialogue(**kwargs).out_hf
        assert num_shards > 1
        shards = []
        for shard in range(num_shards):
            shards.append(HfToDialogue(shard=shard, num_shards=num_shards, **kwargs))
        return HfMergeShards(shard_paths=[s.out_hf for s in shards]).out_hf

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        with vllm_server(self.llm_name) as llm_url:
            print("Now loading dataset")
            dataset = load_from_disk(self.dataset_split_path.get())
            if self.shard is not None and self.num_shards is not None:
                dataset = dataset.shard(num_shards=self.num_shards, index=self.shard)

            print("Dataset loaded successfully. Now generating dialogues...")
            dataset = dataset.map(
                make_dialogue_gen(llm_url, self.llm_name, self.dialogue_instructions),
                num_proc=32,
            )
            print("Dialogues generated successfully. Now saving the dataset...")

            dataset.save_to_disk(self.out_hf.get())
            dataset.to_json(self.out_json.get())


class HfDialogueToJsonFile(Job):
    def __init__(
        self,
        *,
        hf_dataset_path: tk.Path,
        split: str | None = None,
        ignore_errors: bool = False,
    ):
        self.hf_dataset_path = hf_dataset_path
        self.split = split
        self.ignore_errors = ignore_errors
        self.out_json = self.output_path("dialogue.jsonl")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        dataset = load_from_disk(self.hf_dataset_path.get())
        if self.split is not None:
            assert type(dataset) is DatasetDict
            dataset = dataset[self.split]
        else:
            assert type(dataset) is Dataset

        num_errors = 0
        # go through the dataset, parse the "dialogue" field as json, and append that json as a single line to a jsonl file
        with open(self.out_json.get(), "w") as f:
            for example in dataset:
                dialogue_str: str = example["dialogue"]
                dialogue_str = dialogue_str.strip()
                try:
                    if dialogue_str.startswith("```json"):
                        dialogue_str = dialogue_str[len("```json") :]
                    if dialogue_str.endswith("```"):
                        dialogue_str = dialogue_str[: -len("```")]
                    dialogue_json = json.loads(dialogue_str)
                    f.write(json.dumps(dialogue_json) + "\n")
                except Exception as e:
                    num_errors += 1
                    if not self.ignore_errors:
                        print("###")
                        print(dialogue_str)
                        print(f"Error parsing dialogue: {e} for {example}")
                        print("###")

        if num_errors > 0 and not self.ignore_errors:
            raise ValueError(
                f"Encountered {num_errors} errors while parsing dialogues. See above for details."
            )
        if num_errors > len(dataset) * 0.1:
            raise ValueError(
                f"Encountered {num_errors} errors while parsing dialogues, which is more than 10% of the dataset. Something might be wrong. See above for details."
            )


# Same as above, but saves to Hf dataset
class HfDialogueCleaner(Job):
    def __init__(
        self,
        *,
        hf_dataset_path: tk.Path,
        split: str | None = None,
        ignore_errors: bool = False,
    ):
        self.hf_dataset_path = hf_dataset_path
        self.split = split
        self.ignore_errors = ignore_errors
        self.out_hf = self.output_path("out_hf", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        dataset = load_from_disk(self.hf_dataset_path.get())
        if self.split is not None:
            assert type(dataset) is DatasetDict
            dataset = dataset[self.split]
        else:
            assert type(dataset) is Dataset

        def clean_dialogue(example):
            dialogue_str: str = example["dialogue"]
            dialogue_str = dialogue_str.strip()

            if dialogue_str.startswith("```json"):
                dialogue_str = dialogue_str[len("```json") :]
            if dialogue_str.endswith("```"):
                dialogue_str = dialogue_str[: -len("```")]

            example["dialogue"] = dialogue_str
            return example

        cleaned = dataset.map(clean_dialogue)

        def filter_dialogue(example):
            dialogue_str: str = example["dialogue"]
            try:
                json.loads(dialogue_str)
                return True
            except Exception as e:
                if not self.ignore_errors:
                    print("###")
                    print(dialogue_str)
                    print(f"Error parsing dialogue: {e} for {example}")
                    print("###")
                return False

        previous_len = len(cleaned)
        filtered = cleaned.filter(filter_dialogue)
        num_errors = previous_len - len(filtered)
        if num_errors > 0 and not self.ignore_errors:
            print(
                f"Encountered {num_errors} errors while parsing dialogues. See above for details."
            )
        if num_errors > len(dataset) * 0.1:
            raise ValueError(
                f"Encountered {num_errors} errors while parsing dialogues, which is more than 10% of the dataset. Something might be wrong. See above for details."
            )
        filtered.save_to_disk(self.out_hf.get())
