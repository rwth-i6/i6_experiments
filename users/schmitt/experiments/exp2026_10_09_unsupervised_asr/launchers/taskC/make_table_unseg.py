import re, sys
arms = [("unseg_nocollapse", "no collapse (ablation)"), ("unseg_lz0", "B, lam_z 0"),
        ("unseg_lz1", "B, lam_z 1"), ("unseg_lz10", "B, lam_z 10"), ("unseg_lz100", "B, lam_z 100")]
row = re.compile(r"^  (\d) +([0-9.]+) +([0-9.]+) +([0-9.]+) +([0-9.]+)(.*)$")
print("| arm | seed | hard loss | Z | hyp len | PER % |")
print("|---|---|---|---|---|---|")
for f, desc in arms:
    try:
        txt = open("/u/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/taskC/%s.log" % f).read()
    except IOError:
        continue
    if "  seed      hard loss" not in txt:
        continue
    block = txt[txt.index("  seed      hard loss"):]
    rows = []
    for line in block.splitlines()[1:]:
        m = row.match(line)
        if not m:
            break
        rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)),
                     float(m.group(5)), "loss-selected" in m.group(6)))
    for sd, hard, z, p, hl, sel in sorted(rows):
        print("| %s | %d%s | %.3f | %.4f | %.1f | **%.2f** |"
              % (desc, sd, " **(loss-selected)**" if sel else "", hard, z, hl, p))
