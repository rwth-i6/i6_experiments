qstat -xml | tr '\n' ' ' | sed 's#<job_list[^>]*>#\n#g' \
	  | sed 's#<[^>]*>##g' | grep " " | column -t
