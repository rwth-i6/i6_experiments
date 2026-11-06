from typing import Any, AsyncIterator, List, Tuple

from sisyphus import tk

from ......model_pipelines.common.recog import MemristorRecogResult, RecogResult
from ...common import post_recog_memristor_results


ResultBatch = Tuple[List[MemristorRecogResult], List[RecogResult]]


async def run(
    recognition: Any,
    model: Any,
    variants: List[Any],
    regular_variants: List[Any],
    corpora: List[Any],
    selection_corpus: str,
    max_runs: int,
    top_k: int,
    **recog_kwargs: Any,
) -> AsyncIterator[ResultBatch]:
    """Select additional cycles by WER using a memristor recognition module.

    The module must provide run(...) and _run_single_variant(...); recog_kwargs
    are passed to both. Results are yielded as soon as their jobs are created.
    """
    assert max_runs >= 1
    assert top_k >= 1
    assert corpora

    common_kwargs = dict(model=model, corpora=corpora, **recog_kwargs)
    yield recognition.run(variants=regular_variants, max_runs=max_runs, **common_kwargs)

    first_cycle = []
    for variant in variants:
        _, results = recognition._run_single_variant(variant=variant, num_cycles=1, **common_kwargs)
        selection_result = next(result for result in results if result.corpus_name == selection_corpus)
        assert selection_result.wer is not None
        first_cycle.append((variant, results, selection_result.wer))
        yield [], results

    if first_cycle:
        await tk.async_run([wer for _, _, wer in first_cycle])
    selected_indices = set(
        sorted(range(len(first_cycle)), key=lambda index: float(first_cycle[index][2].get()))[:top_k]
    )

    for index, (variant, first_cycle_results, _) in enumerate(first_cycle):
        cycles_results = list(first_cycle_results)
        if index in selected_indices:
            for num_cycles in range(2, max_runs + 1):
                _, results = recognition._run_single_variant(variant=variant, num_cycles=num_cycles, **common_kwargs)
                cycles_results.extend(results)
                yield [], results

        descriptor = f"{model.descriptor}_memristor_{'cache_' if recog_kwargs.get('use_cache') else ''}{variant.descriptor}"
        yield post_recog_memristor_results(descriptor=descriptor, corpora=corpora, recog_results=cycles_results), []
