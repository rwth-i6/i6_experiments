from i6_core.report.report import _Report_Type


def gmm_example_report_format(report: _Report_Type) -> str:
    """
    Example report format for a GMM evaluated on dev-clean and dev-other
    :param report:
    :return:
    """
    results = {
        "dev-clean": {
            "Monophone": {},
            "Triphone": {},
            "SAT": {},
            "VTLN": {},
            "VTLN+SAT": {},
        },
        "dev-other": {
            "Monophone": {},
            "Triphone": {},
            "SAT": {},
            "VTLN": {},
            "VTLN+SAT": {},
        },
    }
    for step_name, score in report.items():
        if not step_name.startswith("scorer"):
            continue
        if "dev-clean" in step_name:
            set = "dev-clean"
        else:
            set = "dev-other"
        if "mono" in step_name:
            step = "Monophone"
        elif "tri" in step_name:
            step = "Triphone"
        elif "vtln+sat" in step_name:
            step = "VTLN+SAT"
        elif "sat" in step_name:
            step = "SAT"
        else:
            step = "VTLN"
        if "iter08" in step_name:
            results[set][step]["08"] = score
        elif "iter10" in step_name:
            results[set][step]["10"] = score

    out = []
    out.append(
        f"""Name: {report["name"]}

        Results:"""
    )
    out.append("Step".ljust(20) + "dev-clean".ljust(10) + "dev-other")
    out.append(
        "Monophone".ljust(16)
        + str(results["dev-clean"]["Monophone"]["10"]).ljust(14)
        + str(results["dev-other"]["Monophone"]["10"])
    )
    out.append(
        "Triphone 08".ljust(19)
        + str(results["dev-clean"]["Triphone"]["08"]).ljust(14)
        + str(results["dev-other"]["Triphone"]["08"])
    )
    out.append(
        "Triphone 10".ljust(19)
        + str(results["dev-clean"]["Triphone"]["10"]).ljust(14)
        + str(results["dev-other"]["Triphone"]["10"])
    )
    out.append(
        "VTLN 08".ljust(21)
        + str(results["dev-clean"]["VTLN"]["08"]).ljust(14)
        + str(results["dev-other"]["VTLN"]["08"])
    )
    out.append(
        "VTLN 10".ljust(21)
        + str(results["dev-clean"]["VTLN"]["10"]).ljust(14)
        + str(results["dev-other"]["VTLN"]["10"])
    )
    out.append(
        "SAT 08".ljust(23)
        + str(results["dev-clean"]["SAT"]["08"]).ljust(14)
        + str(results["dev-other"]["SAT"]["08"])
    )
    out.append(
        "SAT 10".ljust(23)
        + str(results["dev-clean"]["SAT"]["10"]).ljust(14)
        + str(results["dev-other"]["SAT"]["10"])
    )
    out.append(
        "VTLN+SAT 08".ljust(17)
        + str(results["dev-clean"]["VTLN+SAT"]["08"]).ljust(14)
        + str(results["dev-other"]["VTLN+SAT"]["08"])
    )
    out.append(
        "VTLN+SAT 10".ljust(17)
        + str(results["dev-clean"]["VTLN+SAT"]["10"]).ljust(14)
        + str(results["dev-other"]["VTLN+SAT"]["10"])
    )

    return "\n".join(out)
