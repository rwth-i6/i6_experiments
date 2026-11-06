from recipe.experiments.librispeech.groups import run_memristor


async def py():
    return await run_memristor("full_ctx_transducer", "qat_encoder_pred", 8)
