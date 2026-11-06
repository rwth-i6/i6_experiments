from recipe.experiments.librispeech.groups import run_memristor


async def py():
    return await run_memristor("ctc", "qat_full_model", 8)
