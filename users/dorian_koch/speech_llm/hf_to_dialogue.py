from pathlib import Path
from sisyphus import Job, Task, tk
import os
import subprocess
from .common import HF_CACHE_DIR, vllm_server
from datasets import load_dataset
from openai import OpenAI


def make_dialogue_gen(llm_url, model_name):
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

        user_msg += (
            "You are a helpful assistant for converting multiple choice questions into dialogues. "
            "You will be given a multiple choice question, possible answers, and the single correct answer. "
            "Your task is to create a dialogue between a user and an assistant, where the user asks the question and the assistant provides the correct answer with some explanation as to how it arrived at that answer. "
            "Make the dialogue natural and engaging, while ensuring that the correct answer is clearly conveyed by the assistant. "
            "The user may provide some potention options as answers, but the user should not just list them - they should be integrated into the dialogue in a natural way. "
            "Because the dialogue will be spoken by a TTS system, do not use special characters or formatting. "
            "I.e. you must translate any formatting or latex into spoken English words. Keep it natural. "
            "The TTS system can handle  the paralinguistic tags [laugh], [chuckle] and [cough]. "
            "While multiple speaker turns can be used, this is almost never necessary. "
            "Now generate a dialogue between the user and the assistant based on the above information. "
            "The dialogue must be outputted as a JSON array of objects, where each object has a 'speaker' field (either 'user' or 'assistant') and a 'text' field (the content of the speaker's turn). ONLY output the JSON array, and nothing else. Do not include any explanations or other text. "
        )

        messages = [
            {"role": "user", "content": user_msg},
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            seed=example["question_id"],
        )

        prediction = response.choices[0].message.content

        return {"dialogue": prediction}

    return make_dialogue


class HfToDialogue(Job):
    def __init__(
        self,
        *,
        dataset_name: str,
        llm_name: str,
    ):
        self.dataset_name = dataset_name
        self.llm_name = llm_name

        self.out_hf = self.output_path("dialogue_dataset", directory=True)
        self.out_json = self.output_path("dialogue_dataset.json")
        self.rqmt = {
            "gpu": 1,
            "cpu": 2,
            "mem": 16,
            "time": 2,
        }

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 1
        return super().hash(d)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        with vllm_server(self.llm_name) as llm_url:
            dataset = load_dataset(self.dataset_name)
            dataset = dataset["validation"].map(
                make_dialogue_gen(llm_url, self.llm_name), num_proc=4
            )

            dataset.save_to_disk(self.out_hf.get())
            dataset.to_json(self.out_json.get())
