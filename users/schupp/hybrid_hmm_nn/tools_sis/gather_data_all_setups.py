# This script gathers data from several different setups
# All of them have a little different structure, this handels some of the special cases here
from typing import OrderedDict
from pathlib import Path
import subprocess
import os
from datetime import datetime

import argparse
import logging as log
log.basicConfig(level=log.DEBUG)


parser = argparse.ArgumentParser()
parser.add_argument('-se', '--skip-extraction', action="store_true")
args = parser.parse_args()

ABS_TOOLS_PATH = "/u/schupp/setups/ping_setup_refactor_tf23/recipe/i6_experiments/users/schupp/hybrid_hmm_nn/tools_sis"
EXTRACTOR_ROOT = os.getcwd()

PY = "/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python3.8"

# setup_path: absolute path so sis setup
# short_name: ...
# link_to_path: local path to link the setup to
# extract_command: extraction command using my make_py_summary.py data extractor

setups = [
    {
        "short_name" : "tim-old",
        "setup_path" : "/u/schupp/setups/ping_setup_refactor_tf23", 
        "link_to_path" : "setup1_tim_old/",
        # This setup needs no params ( this scipt was first desighned for this setup)
        "extract_command" : f'{PY} tools_linked/make_py_summary.py',  # -ox se_block TODO: remove the filter, it's just for debugging
        "table_command" : f'{PY} tools_linked/make_full_table_02.py'
    },
    {
        "short_name" : "tim-new",
        "setup_path" : "/u/schupp/setups/i6_exp_conformer_rtc", 
        "link_to_path" : "setup2_tim_new/",
        "extract_command" : f'{PY} tools_linked/make_py_summary.py -udf "dev-other:"', # Update data filter command, cause in this setup 'dev-other' is *not* postfixed with '_dev-other'
        "table_command" : f'{PY} tools_linked/make_full_table_02.py'
    },
    { # This has quite some conformer results, also the best baseline I'm currently using was from here
        "short_name" : "ping-old",
        "setup_path" : "/work/asr3/luescher/hiwis/pzheng/librispeech/transformer_conformer_21_11_10", 
        "link_to_path" : "setup3_ping_transformer_conformer_21_11_10",
        "extract_command" : f'{PY} tools_linked/make_py_summary.py -udf "dev-other:"', # TODO: this should use some 'ignore sets' option, because really this only has usefull info on devtain, or look at 'rerecog'
        "table_command" : f'{PY} tools_linked/make_full_table_02.py'
    },
    { # In this one there aren't really man conformer results, and they are mostly also wit SMBR which I don't care about
        "short_name" : "ping-fromscratch",
        "setup_path" : "/work/asr3/luescher/hiwis/pzheng/librispeech/from-scratch", 
        "link_to_path" : "setup4_ping_refactor",
        "extract_command" : f'{PY} tools_linked/make_py_summary.py -udf "dev-other:"',
        "table_command" : f'{PY} tools_linked/make_full_table_02.py'
    },
    # there is also '/work/asr3/luescher/hiwis/pzheng/librispeech/from-scratch_tf23' but it doesn't seem to contain any usefull experiments
    { # 
        "short_name" : "ping-fromscratch-refactored",
        "setup_path" : "/work/asr3/luescher/hiwis/pzheng/librispeech/from-scratch_refactored", 
        "link_to_path" : "setup5_ping_refactor2",
        "extract_command" : f'{PY} tools_linked/make_py_summary.py -udf "dev-other:"',
        "table_command" : f'{PY} tools_linked/make_full_table_02.py'
    },
]

# TODO: remove, just for debugging:
# setups = setups[0:3]
setups = setups[2]

# Links all important setup folders: alias, output ( there is the data we want to gather )
def maybe_initiate_links(setup_def):
    os.chdir(setup_def['link_to_path'])
    path_to_setup = setup_def['setup_path']

    if not os.path.exists("LINKS_CREATED"):
        os.symlink(f'{path_to_setup}/alias', "./alias")
        os.symlink(f'{path_to_setup}/output', "./output")
        os.mkdir("results2") # This is where the reuslts for data extraction are stored ( i.e.: If this is done we can always generate the overviews without extracting )
        os.symlink(ABS_TOOLS_PATH, "./tools_linked") # We need the tools for the data extraction
        os.symlink("./tools_linked/get_wer.py", "./get_wer.py") # We need the tools for the data extraction
        os.symlink("./tools_linked/get_wer_for_set.py", "./get_wer_for_set.py") 
        os.close(os.open("LINKS_CREATED", os.O_CREAT))

    os.chdir(EXTRACTOR_ROOT)

def extract_datas_for_setup(setup_def):
    os.chdir(setup_def['link_to_path'])

    # Run the extraction command
    # We want as much data as possible, we parse: logs, configs, sis worker files, qstat ( might take a while )

    # TODO: handle exeptions, make defaults
    _call = setup_def["extract_command"].split(" ") # Yeah we still assume there are no spaces in names but that served us fine so far
    out = subprocess.check_output(_call).decode('UTF-8') # TODO: check also the return code

    date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    os.close(os.open(f"LAST_EXTRACTED_{date_time}", os.O_CREAT)) # This might allow shipping the extraction at another time

    # TODO: this should also print some sort of 'failure-report'

    os.chdir(EXTRACTOR_ROOT)

    return out


def generate_data_table(setup_def):
    os.chdir(setup_def['link_to_path'])

    # Run the extraction command
    # We want as much data as possible, we parse: logs, configs, sis worker files, qstat ( might take a while )

    # TODO: handle exeptions, make defaults
    _call = setup_def["table_command"].split(" ") # Yeah we still assume there are no spaces in names but that served us fine so far
    out = subprocess.check_output(_call).decode('UTF-8') # TODO: check also the return code

    date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    os.close(os.open(f"LAST_OVERVIEW_GENERATED_{date_time}", os.O_CREAT))

    # TODO: this should also print some sort of 'failure-report'

    os.chdir(EXTRACTOR_ROOT)

    return out

def merge_tables_big_overview(setups): # TODO
    # Tables all the generated overviews and puts them in one big table...
    pass

log.info(setups)
for setup in setups:
    log.info(f"Processing setup: {setup['short_name']}")
    # 1 - create all the links ( if not jet present )
    maybe_initiate_links(setup)

    # 2 - run the data extraction
    if not args.skip_extraction:
        extract_datas_for_setup(setup)

    # 3 - genate a big table, containing all importatnt data for this setup
    generate_data_table(setup)

    # TODO: this sould also generrate plots, for every setup:
    # - learning rates
    # - wers ( over all epochs that are stored )
    # - errors, losses

    # TODO: this should also generate joined plots, from a selction of experiments defined in some datastucture
    # e.g.: a splot with wers per epoch, per variation of learning rate experiment co...

merge_tables_big_overview(setups)