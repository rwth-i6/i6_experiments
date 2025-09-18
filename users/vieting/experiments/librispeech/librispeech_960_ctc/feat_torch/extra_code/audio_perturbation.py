
def audio_perturbation_v1(audio, sample_rate, random_state, perturbations):
    import sox
    tfm = sox.Transformer()
    for pert_type, config in perturbations.items():
        if random_state.random() < config.get("prob", 1.0):
            factor = random_state.random() * (config.get("max", 1.0) - config.get("min", 1.0)) + config.get("min", 1.0)
            if factor != 1.0:
                if pert_type == "speed":
                    tfm.speed(factor)
                elif pert_type == "tempo":
                    tfm.stretch(factor)
                else:
                    raise NotImplementedError(pert_type)
    audio = tfm.build_array(input_array=audio, sample_rate_in=sample_rate)
    return audio
