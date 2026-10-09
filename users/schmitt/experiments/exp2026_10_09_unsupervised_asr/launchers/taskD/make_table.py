import re, sys, glob
pat = re.compile(r"^  \[(.+?)\] hardened objective ([0-9.]+) \| acc ([0-9.]+)")
init = re.compile(r"^ +0 +(?:tau [0-9.]+ +)?soft [0-9.]+ +hard [0-9.]+ +acc ([0-9.]+)")
for f in sys.argv[1:]:
    rows, pend = [], None
    for line in open(f):
        m = init.match(line)
        if m:
            pend = float(m.group(1))
        m = pat.match(line)
        if m:
            rows.append((m.group(1), float(m.group(2)), pend, float(m.group(3))))
    for lab, hard, a0, a in rows:
        print("| %s | %s | %.5f | %.4f | **%.4f** | %.1f |"
              % (f.split("/")[-1][:-4], lab, hard, a0 if a0 else float("nan"), a, 100 * (1 - a)))
