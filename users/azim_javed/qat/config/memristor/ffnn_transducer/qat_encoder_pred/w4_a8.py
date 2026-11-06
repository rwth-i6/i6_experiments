from recipe.experiments.librispeech.groups import run_memristor


async def py():
    return await run_memristor("ffnn_transducer", "qat_encoder_pred", 4)
