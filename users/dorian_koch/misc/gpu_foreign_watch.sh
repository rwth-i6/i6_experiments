#!/bin/bash
# Sisyphus worker prefix (settings.py::worker_wrapper):  gpu_foreign_watch.sh <log prefix> -- <worker call...>
#
# Runs the worker unchanged (exec). On a SLURM GPU job it also starts a background watcher that,
# every 60 s, lists the compute processes on THIS job's GPU(s) and logs, once per pid, every one that
# does not belong to this SLURM job: owner, exe path, cwd, command line, start time, and the SLURM
# job it belongs to (id, user, name). Lines go to <log prefix>.<array task id> (= log.run.N), tagged
# `[gpu-foreign]`. Motivation: 2026-09-28 PersonaSelfPlay.EAdY0g1kSd5g OOMed on n25g0004 next to a
# 10.3 GiB process that was not ours.
#
# GPU_FOREIGN_TEST=1 treats this job's own processes as foreign (to show the logging path works).
set -u
log_prefix=$1
shift
[[ ${1:-} == -- ]] && shift

watch_gpus() {
    local worker=$1 log=$2 seen=" " uuids line uuid pid mem name owner exe cwd args start fjob finfo i
    # Sisyphus's init_worker unlinks log.run.N and hard-links it to the engine log, so never hold it
    # open: wait until it exists, then append by path, one line at a time.
    for i in {1..60}; do [[ -e $log ]] && break; sleep 5; done
    sleep 15
    uuids=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES" 2>/dev/null) || return
    [[ -n $uuids ]] || return
    echo "[gpu-foreign] $(date '+%F %T') watching $(hostname -s) GPU(s) ${uuids//$'\n'/ } for processes outside SLURM job $SLURM_JOB_ID" >>"$log"
    while kill -0 "$worker" 2>/dev/null; do
        while IFS= read -r line; do
            IFS=',' read -r uuid pid mem name <<<"$line"
            uuid=${uuid// /} pid=${pid// /} mem=${mem// /} name=${name# }
            [[ -n $pid && $uuids == *"$uuid"* && $seen != *" $pid "* ]] || continue
            if [[ ${GPU_FOREIGN_TEST:-0} != 1 ]] && grep -qs "job_${SLURM_JOB_ID}\b" "/proc/$pid/cgroup"; then
                continue
            fi
            seen+="$pid "
            owner=$(ps -o user= -p "$pid" 2>/dev/null) || owner="? (pid not visible)"
            exe=$(readlink "/proc/$pid/exe" 2>/dev/null) || exe="$name (nvidia-smi; exe unreadable)"
            cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null) || cwd="?"
            args=$(ps -o args= -p "$pid" 2>/dev/null | cut -c1-600)
            start=$(ps -o lstart= -p "$pid" 2>/dev/null)
            fjob=$(grep -so 'job_[0-9]*' "/proc/$pid/cgroup" | head -1)
            finfo="none (not in a SLURM job cgroup)"
            if [[ -n $fjob ]]; then
                finfo="${fjob#job_} $(squeue -h -j "${fjob#job_}" -o '%u %T %j' 2>/dev/null | head -1)"
            fi
            echo "[gpu-foreign] $(date '+%F %T') FOREIGN process on our GPU $uuid: pid=$pid used=${mem}MiB" \
                "owner=${owner// /} exe=$exe cwd=$cwd started='${start}' slurm_job=$finfo cmd=$args" >>"$log"
        done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader,nounits 2>/dev/null)
        sleep 60
    done
}

if [[ -n ${SLURM_JOB_ID:-} && -n ${CUDA_VISIBLE_DEVICES:-} ]] && command -v nvidia-smi >/dev/null 2>&1; then
    watch_gpus $$ "$log_prefix.${SLURM_ARRAY_TASK_ID:-1}" >/dev/null 2>&1 </dev/null &
fi
exec "$@"
