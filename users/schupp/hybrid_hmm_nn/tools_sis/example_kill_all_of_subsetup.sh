for i in $(rnn_py tools/scan_alias_all_status.py -f conformer/l2_drop/ -pid) ; do qdel $i ; done
