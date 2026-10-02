# 19: EpochData(learningRate=1.0, error={
# ':meta:cpu': 'Intel(R) Xeon(R) Platinum 8468',
# ':meta:device': 'NVIDIA H100',
# ':meta:distributed': None,
# ':meta:effective_learning_rate': 9.23773517305582e-06,
# ':meta:epoch_num_train_steps': 8604,
# ':meta:epoch_train_time_secs': 3418,
# ':meta:global_train_step': 148517,
# ':meta:global_train_step_end': 157121,
# ':meta:hostname': 'w23g0009.hpc.itc.rwth-aachen.de',
# ':meta:returnn': '1.20260123.092204+git.7d9bda18',
# ':meta:time': '2026-02-04-03-18-15 (UTC+0100)',
# ':meta:torch': '2.2.0a0+81ea7a4 (Unknown) (<not-under-git> in /usr/local/lib/python3.10/dist-packages/torch)',
# 'dev_loss_ce': 1.7339062545408277,
# 'dev_loss_ctc-0': 0.33693073066566864,
# 'dev_loss_fer': 0.03911426153870715,
# 'devtrain_loss_ce': 1.7280769520393686,
# 'devtrain_loss_ctc-0': 0.30832190473183263,
# 'devtrain_loss_fer': 0.03754887813052788,
# 'train_loss_ce': 1.829695251429992,
# 'train_loss_ctc-0': 0.5052894207865172,
# 'train_loss_fer': 0.061695749031613865,
# 'train_loss_grad_norm:p2': 12.967069952814095,
# }),
import sys

lr_file = sys.argv[1]

times = []
with open(lr_file, "r") as f:
    for line in f:
        if ":meta:epoch_train_time_secs" in line:
            parts = line.split(":meta:epoch_train_time_secs': ")
            if len(parts) > 1:
                time_part = parts[1].split(",")[0].strip().strip("'").strip()
                try:
                    time_secs = float(time_part)
                    times.append(time_secs)
                except ValueError as e:
                    print(f"Could not convert to float: {time_part}, error: {e}")
                    continue

if times:
    avg_time = sum(times) / len(times)
    print(f"Average time per subepoch: {avg_time:.2f} seconds over {len(times)} subepochs")
    print(f"Total time for all subepochs: {sum(times):.2f} seconds = {sum(times)/3600:.2f} hours")
else:
    print("No epoch train time data found.")
