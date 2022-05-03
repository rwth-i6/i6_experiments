qstat -xml -u $1 | tr '\n' ' ' | sed 's#<job_list[^>]*>#\n#g' \
	  | sed 's#<[^>]*>##g' | grep " " | column -t
