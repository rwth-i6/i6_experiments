import re
from typing import List, Tuple, Optional


def clean_string(s: str, custom_subs: Optional[List[Tuple[str, str]]] = None) -> str:
    for pattern, substitution in (
        r"[\!\"\%\,\/\:\;\?\{\}\&]": "",
        "`": "'",
        r"\.(\w)": r"\g<1>",
        r"(\s|\A)\'": r"\g<1>",
        r"(\s|\A)\(": r"\g<1>",
        r"(\s|\A)\)": r"\g<1>",
        r"\(\S*\)": "",
        r"\[\S*\]": "",
        "-HYPHEN": "HYPHEN",
        "--DASH": "DASH",
	    r" *</s>": "",
	    r"<s> *": "",
	    r" *<.*> *": "",
	    r" *< *": "",
	    r" *> *": "",
	    r" *\* *": "",
	    r" *, *": "",
	    r" *\^ *": "",
	    r" *\\ *": "",
	    r" *\| *": "",
	    r" *~ *": "",
	    r" *\[.*\] *": "",
	    r" *\[ *": "",
	    r" *\] *": "",
	    r" *\. *": "",
	    r" *# *": "",
	    r"\$": "dollars",
	    r"(.)\1+": r"\1\1",
    } + custom_subs).items():
        s = re.sub(pattern, substitution, s)
    
    s = " ".join(s.split())
    return s


def lm_cleaning(s: str):
	remove_regexes = [
	    re.compile(expr) for expr in [
	        r" *</s>",
	        r"<s> *",
	        r" *<.*> *",
	        r" *< *",
	        r" *> *",
	        r" *\* *",
	        r" *, *",
	        r" *\^ *",
	        r" *\\ *",
	        r" *\| *",
	        r" *~ *",
	        r" *\[.*\] *",
	        r" *\[ *",
	        r" *\] *",
	        r" *\. *",
	        r" *# *",
	    ]
	]
	replace_regexes = [
	    (re.compile(r"\$"), "dollars"),
	    (r"(.)\1+", r"\1\1"),
	]
	sentence_clean = english_cleaners(s)
	for expr in remove_regexes:
		sentence_clean = re.sub(expr, "", sentence_clean)
	for expr, repl in replace_regexes:
		sentence_clean = re.sub(expr, repl, sentence_clean)
	return sentence_clean