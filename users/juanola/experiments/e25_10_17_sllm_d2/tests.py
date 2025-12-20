from i6_core.report.report import GenerateReportStringJob
from i6_core.tools.download import DownloadJob
from sisyphus import tk
from .constants import SIS_BASE_REPORT_EXTENSION, SIS_ALIASES_REPORTS
from .default_tools import RETURNN_EXE, RETURNN_ROOT
from .experiments_core.reporting.templates.experiment_report_templates import experiment_report_template_v0

ROOT_RETURNN_ROOT = {
    "returnn_exe": RETURNN_EXE,
    "returnn_root": RETURNN_ROOT,
}

def hf_config_download_test(test_name = "hf_config_download_test"):
    download_config_job = DownloadJob("https://huggingface.co/Qwen/Qwen2-0.5B/resolve/main/config.json",
                                      target_filename="config-Qwen2-0_5B.json")
    tk.register_output(f"test/{test_name}", download_config_job.out_file)


def report_job_test_with_results(test_name = "report_job_test1"):
    results = {
        "a1": 12,
        "a2": 10
    }
    report_job = GenerateReportStringJob(report_values=results, report_template=None, compress= False)
    tk.register_output(f"test/{test_name}", report_job.out_report)
    report_job.add_alias(f"{SIS_ALIASES_REPORTS}/{test_name}")

def report_job_test_with_results_and_template(test_name = "report_job_test2"):
    results = {
        "a1": 12,
        "a2": 10
    }
    report_job = GenerateReportStringJob(report_values=results, report_template=experiment_report_template_v0, compress=False)
    tk.register_output(f"test/{test_name}", report_job.out_report)
    #report_job.add_alias(f"test/{test_name}")

def report_job_test_register_report(test_name = "report_job_test3"):
    results1 = {
        "a1": 12,
        "a2": 10
    }

    results2 = {
        "b1": 123,
        "b2": 1330
    }

    results = {
        "res1": results1,
        "res2": results2,
    }

    tk.register_report(
        f"test/{test_name}_report.{SIS_BASE_REPORT_EXTENSION}",
        results,
        required=results,
        update_frequency=900
    )



def mail_test():

    pass