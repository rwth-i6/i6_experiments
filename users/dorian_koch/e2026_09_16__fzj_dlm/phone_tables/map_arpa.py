"""Automatic target-IPA -> English ARPAbet phone map by articulatory-feature distance (PanPhon).
No hand choices: ARPAbet->IPA is the standard CMUdict convention; each target phone takes the ARPAbet phone
with the smallest weighted feature edit distance (ties: alphabetical). Vowels get primary stress (1), except the
reduced vowels that CMUdict itself writes with stress 0 (AH0 = ə, ER0 = ɚ), which are separate candidates."""
import json, sys, unicodedata
import panphon.distance
ARPA = {"AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "B": "b", "CH": "tʃ", "D": "d", "DH": "ð",
        "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i", "JH": "dʒ", "K": "k",
        "L": "l", "M": "m", "N": "n", "NG": "ŋ", "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
        "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ"}
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
cands = {(k + "1" if k in VOWELS else k): v for k, v in ARPA.items()}
cands["AH0"] = "ə"; cands["ER0"] = "ɚ"
dst = panphon.distance.Distance()
target = sys.argv[1].split()
out = {}
for p in target:
    if p == "spn":
        out[p] = "spn"; continue
    ipa = unicodedata.normalize("NFD", p)
    scores = sorted((dst.weighted_feature_edit_distance(ipa, unicodedata.normalize("NFD", v)), a) for a, v in cands.items())
    out[p] = scores[0][1]
    print(f"{p:4s} -> {scores[0][1]:4s} ({scores[0][0]:.2f})   next: {scores[1][1]} ({scores[1][0]:.2f})")
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False, indent=0)
