"""German MFA phones <-> english_mfa phones, 1:1 so German labels can be recovered by position.
  make_dict: german_ext.dict -> german_ext_en.dict (English phone set, for `mfa adapt english_mfa`)
  recover  : TextGrids aligned with English labels -> same TextGrids with the German phones restored
The 10 German-only phones map to their nearest single English phone; the table is still built per
GERMAN phone, since recovery restores the German identity of every interval."""
import os, re, sys, collections, json
MAP = {"l̩": "ɫ̩", "øː": "eː", "œ": "ɛ", "ɔʏ": "ɔj", "pf": "f", "ʁ": "ɹ", "ts": "tʃ", "x": "h", "yː": "iː", "ʏ": "ɪ"}
if os.environ.get("PHONE_MAP"):
    MAP = json.load(open(os.environ["PHONE_MAP"], encoding="utf8"))  # full target->model map (e.g. PanPhon ARPAbet)
def en(p): return MAP.get(p, p)
def load_dict(p):
    d = collections.defaultdict(list)
    for l in open(p, encoding="utf8"):
        f = l.rstrip("\n").split("\t"); d[f[0]].append(f[-1].split())
    return d
if sys.argv[1] == "make_dict":
    with open(sys.argv[3], "w", encoding="utf8") as out:
        for l in open(sys.argv[2], encoding="utf8"):
            f = l.rstrip("\n").split("\t"); f[-1] = " ".join(en(p) for p in f[-1].split()); out.write("\t".join(f) + "\n")
    sys.exit()
# recover DICT IN_DIR OUT_DIR
D = load_dict(sys.argv[2]); ok = bad = 0
for r, _, fs in os.walk(sys.argv[3]):
    for f in fs:
        if not f.endswith(".TextGrid"): continue
        txt = open(os.path.join(r, f), encoding="utf8").read()
        items = re.split(r'(?=item \[\d+\]:)', txt)
        def ivs(t): return [(float(a), float(b), c) for a, b, c in re.findall(r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "(.*?)"', t)]
        words = next(ivs(t) for t in items if 'name = "words"' in t)
        pi = next(i for i, t in enumerate(items) if 'name = "phones"' in t)
        phones = ivs(items[pi]); new = [p for _, _, p in phones]
        for ws, we, w in words:
            if not w: continue
            idx = [i for i, (s, e, p) in enumerate(phones) if s >= ws - 1e-6 and e <= we + 1e-6 and p]
            seq = [phones[i][2] for i in idx]
            cand = [pr for pr in D.get(w, []) if [en(p) for p in pr] == seq]
            if cand:
                for i, p in zip(idx, cand[0]): new[i] = p
                ok += 1
            else:
                bad += 1
        it = iter(new)
        items[pi] = re.sub(r'(text = ")(.*?)(")', lambda m: m.group(1) + next(it) + m.group(3), items[pi])
        o = os.path.join(sys.argv[4], os.path.relpath(r, sys.argv[3])); os.makedirs(o, exist_ok=True)
        open(os.path.join(o, f), "w", encoding="utf8").write("".join(items))
print(f"words recovered {ok}, unmatched {bad}")
