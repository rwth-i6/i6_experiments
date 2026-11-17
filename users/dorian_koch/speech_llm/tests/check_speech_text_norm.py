"""Guard: the TTS round-trip check (``speech_text_norm.error_rate``) scores writing differences as 0 and real
differences as errors, in every language ChatterboxInferenceV2 checks by default.

Why: ``ChatterboxInferenceV2`` regenerates every segment whose error is above ``max_err``. Without normalisation a
perfect "Ich habe 3 Katzen" read back by the ASR as "Ich habe drei Katzen" scores 0.25 and is regenerated (wasted GPU,
and turns with numbers are the ones regenerated most), while a normaliser that also equated real differences would
let bad audio through. Each case pairs a reference with a transcript as an ASR writes it.

Run (annotate venv: num2words, cn2an, whisper_normalizer)::

    <annotate venv>/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_speech_text_norm.py
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("speech_text_norm", os.path.join(HERE, "..", "speech_text_norm.py"))
norm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(norm)

#: (lang, reference, transcript): the same words, written differently -> error 0
SAME = [
    ("en", "It costs 25 dollars.", "It costs twenty-five dollars"),
    ("en", "In 1998, about 40% of them left.", "in nineteen ninety eight about forty percent of them left"),
    ("en", "Mr. Smith can't come.", "Mister Smith cannot come"),
    ("de", "Ich habe 3 Katzen.", "Ich habe drei Katzen"),
    ("de", "Das war 1998.", "Das war neunzehnhundertachtundneunzig."),
    ("de", "Es kostet 24,50 €.", "Es kostet vierundzwanzig Komma fünf null Euro"),
    ("de", "Etwa 40 % der Leute", "etwa vierzig Prozent der Leute"),
    ("de", "Die E-Mail kam um 7 Uhr.", "Die E Mail kam um sieben Uhr"),
    ("de", "Wir sind 1.000 Leute.", "Wir sind tausend Leute"),
    ("ru", "У меня 3 кошки.", "у меня три кошки"),
    ("ru", "Всё хорошо!", "Все хорошо"),
    ("zh", "我有3只猫。", "我有三只猫"),
    ("zh", "今天有25个人来了，", "今天有二十五个人来了"),
    ("zh", "大约40%的人", "大约百分之四十的人"),
    ("zh", "两个苹果", "二个苹果"),
    ("fr", "J'ai 3 chats.", "j'ai trois chats"),
]
#: (lang, reference, transcript, expected error): real differences stay errors
DIFF = [
    ("en", "I have three cats.", "I have three dogs.", 0.25),
    ("en", "It costs 25 dollars.", "It costs 35 dollars.", 1 / 3),  # Whisper's normaliser: "it costs $25"
    ("de", "Ich habe drei Katzen.", "Ich habe drei Hunde.", 0.25),
    ("de", "Ich habe 3 Katzen.", "Ich habe vier Katzen.", 0.25),
    ("de", "Das ist gut.", "", 1.0),
    ("ru", "У меня три кошки.", "У меня две кошки.", 1 / 4),
    ("zh", "我有三只猫", "我有四只猫", 0.2),
    ("zh", "今天天气不错", "今天天气不好", 1 / 6),
]

#: (lang, reference, transcript, CER below): a near miss in spelling is small in CER while it costs a whole word in WER
CER_SMALL = [
    ("ru", "Ладно, я постараюсь прийти вовремя.", "Ладно, я постараюсь прийти во время.", 0.05),
    ("de", "Ich kenne keine privaten Kontaktdaten Dritter.", "Ich kenne keine privaten Kontaktdatentritter.", 0.1),
]

fails = []
for lang, ref, hyp, top in CER_SMALL:
    r = norm.error_rate(ref, hyp, lang)
    if not (r["cer"] < top < r["wer"]):
        fails.append(
            f"CER {lang}: {ref!r} vs {hyp!r} -> cer {r['cer']:.3f}, wer {r['wer']:.3f}; want cer < {top} < wer"
        )
for lang, ref, hyp in SAME:
    r = norm.error_rate(ref, hyp, lang)
    if r["err"] != 0:
        fails.append(f"SAME {lang}: {ref!r} vs {hyp!r} -> {r}")
for lang, ref, hyp, want in DIFF:
    r = norm.error_rate(ref, hyp, lang)
    if abs(r["err"] - want) > 1e-6:
        fails.append(f"DIFF {lang}: {ref!r} vs {hyp!r} -> {r['err']:.3f}, want {want:.3f} ({r})")
for f in fails:
    print("FAIL", f)
n = len(SAME) + len(DIFF) + len(CER_SMALL)
print(f"check_speech_text_norm: {n - len(fails)}/{n} cases pass")
sys.exit(1 if fails else 0)
