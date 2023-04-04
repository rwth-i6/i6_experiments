from typing import Any, List

from i6_experiments.common.datasets.librispeech.cart import (
    CartQuestionsWithStress,
    CartQuestionsWithoutStress,
)


class DiphoneCartQuestionsWithoutStress(CartQuestionsWithoutStress):
    def __init__(self, max_leaves: int = 12001, min_obs: int = 1000, add_unknown: bool = True):
        super().__init__(max_leaves=max_leaves, min_obs=min_obs, add_unknown=add_unknown)
        self.steps = list(_remove_future_steps(self.steps))


class DiphoneCartQuestionsWithStress(CartQuestionsWithStress):
    def __init__(self, max_leaves: int = 12001, min_obs: int = 1000, add_unknown: bool = True):
        super().__init__(max_leaves=max_leaves, min_obs=min_obs, add_unknown=add_unknown)
        self.steps = list(_remove_future_steps(self.steps))


def _remove_future_steps(steps: List[Any]):
    def remove_future_key(questions):
        for question in questions:
            if "keys" not in question:
                yield question
                continue

            new_key = [k for k in question["keys"].split(" ") if "future" not in k]
            yield {**question, "keys": " ".join(new_key)}

    for step in steps:
        yield {**step, "questions": list(remove_future_key(step["questions"]))}
