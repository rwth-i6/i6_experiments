#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# my_dir = os.path.dirname(os.path.abspath(__file__))
# returnn_dir = os.path.dirname(my_dir)
# sys.path.append(returnn_dir)

import argparse
from ftplib import all_errors
import os
import sys
from time import strftime, gmtime

def write_wer(args):

  name = args.job_name
  job_path = 'alias' + '/' + args.prefix + '/' + name + '/' + args.suffix + '/'
  if not os.path.exists(os.path.realpath(job_path)):
    print("job doesn't exit.", file = sys.stdout)
    print('\n', file=sys.stdout)
    exit(-1)

  train_times_str = []
  train_times_h = []

  train_steps = []
  num_params = 0
  max_epoch = 0

  wers = {}
  wers_opt = {}


  optimized_scales = {}

  best_epoch = None
  best_score = None

  def EpochData(learningRate, error):
    return {'learning_rate': learningRate, 'error': error}

  all_errors = {}

  if os.path.exists(job_path + 'work/learning_rates'):

    with open(job_path + 'work/learning_rates') as f:
      text = f.read()

      data = eval(text)

      # multi-loss: sum score
      final_epoch = [k for (k, v) in data.items() if len(v['error']) > 0][-1]
      numScore = sum([1 if k == 'dev_score' or k == 'dev_score_output' else 0 for k in data[final_epoch]['error']])

      for epoch in list(data.keys())[:(final_epoch+1)]:
        score = 0.0
        n = 0
        for k in data[epoch]['error']:
          if k != 'dev_score' and k != 'dev_score_output': continue
          score += data[epoch]['error'][k]
          n += 1  # pretraining might use different losses

        if n == numScore and (best_score is None or score < best_score):
          best_score = score
          best_epoch = epoch

      for epoch in list(data.keys()):
        all_errors[epoch] = data[epoch]["error"]

    with open(job_path + 'work/returnn.log') as file:
      lines = file.readlines()
      max_epoch = 0
      for line in lines:
        if not num_params:
          if line.startswith('net params'):
            num_params = int(line.split()[-1])
        if line.startswith('train epoch') and "finished" in line: # easy fix, problem was log_level = 5
          splits = line.split(' ')
          train_steps.append(int(splits[5]))
          train_times_str.append(splits[7])
          max_epoch = int(splits[2].split(',')[0])

      for time in train_times_str:
        h, m, s = time.split(':')
        train_times_h.append(round(int(h) + (int(m) + int(s)/60) / 60, 3))


  orig_name = name
  for data_set in [args.set]:
    name = orig_name + "_" + data_set
    print("================>" + data_set + " :")
    recog_path = 'output/' + args.prefix + '/recog_' + name + '/'
    optimize_recog_path = 'output/' + args.prefix + '/optimize_recog_' + name + '/'

    if args.recog_suffix:
      splits = args.prefix.split('/')
      splits[-1] = args.recog_suffix # replace the wrong suffix
      recog_path = 'output/' + '/'.join(splits) + '/recog_' + name + '/'
      optimize_recog_path = 'output/' + '/'.join(splits) + '/optimize_recog_' + name + '/'

    if os.path.exists(os.path.realpath(recog_path)):
      reports = sorted(os.listdir(recog_path))
    else:
      reports = None
    if os.path.exists(os.path.realpath(optimize_recog_path)):
      optimize_logs = sorted(os.listdir(optimize_recog_path))
    else:
      optimize_logs = None

    if reports:
      for rep in reports:
        if os.path.exists(os.path.realpath(recog_path+rep)):
          if not os.path.exists(recog_path+rep+'/sclite.dtl'):
            continue
          with open(recog_path+rep+'/sclite.dtl') as file:
            for line in file.readlines():
              if line.startswith('Percent Total Error'):
                wer = line.split()[-2]

          if rep.endswith('-optlm.reports'):
            e = int(rep.split('-')[0].split('_')[-1])
            if float(wers_opt.get(e, '100.0%')[:-1]) > float(wer[:-1]):
              wers_opt[e] = wer
          else:
            if rep.split('.')[0].isdigit():
              e = int(rep.split('.')[0])
              wers[e] = wer

    wer_path = job_path + 'output/wers'

    if optimize_logs:
      for log in optimize_logs:
        if os.path.exists(os.path.realpath(optimize_recog_path+log)) \
          and not os.path.isdir(os.path.realpath(optimize_recog_path+log)):
          with open(os.path.realpath(optimize_recog_path+log)) as file:
            line = file.readline()
            splits = line.split(' ')
            am_scale = float(splits[5])
            lm_scale = float(splits[8])
            wer_in_opt = round(float(splits[-1]), 2)

            if log.split('.')[0].isdigit():
              e = int(log.split('.')[0])

              optimized_scales[e] = (am_scale, lm_scale, str(wer_in_opt)+'%',
                                    round(wer_in_opt - float(wers.get(e, '0%')[:-1]), 2))


    if wers:
      wers = {k: wers[k] for k in sorted(wers)}
    if wers_opt:
      wers_opt = {k: wers_opt[k] for k in sorted(wers_opt)}

    if not train_steps:
      train_steps = [0]
    if not train_times_h:
      train_times_h = [0]

    if not args.print:
      if not args.print_to:
        #sprint_nn_100h_train_20210907
        args.print_to = '_'.join(args.prefix.split('/')) + '_' + strftime("%Y%m%d", gmtime()) + '.txt'

      with open(args.print_to, 'a') as file:

        file.write(name)
        file.write('\n')

        file.write(f'num_params: {round(num_params/1000000, 4)} mio.\n')
        file.write(f'average steps per subepoch: {int(sum(train_steps)/len(train_steps))}\n')

        file.write(f'training time per subepoch: {round(sum(train_times_h) / len(train_times_h), 3)} h\n')
        file.write(f'training time per subepoch: {round(sum(train_times_h)*60 / len(train_times_h), 3)} m\n')

        print('recog:    ', wers, file=file)
        print('rerecog:  ', wers_opt, file=file)


        file.write('\n')

    else:

      print(f'num_params: {round(num_params / 1000000, 4)} mio.',
            file = sys.stdout)
      print(f'average steps per subepoch: {int(sum(train_steps) / len(train_steps))}',
            file = sys.stdout)

      h_per_subepoch = sum(train_times_h) / len(train_times_h)

      print(f'training time per subepoch: {round(h_per_subepoch, 3)} h',
            file = sys.stdout)
      print(f'training time per subepoch: {round(sum(train_times_h) * 60 / len(train_times_h), 3)} m',
            file = sys.stdout)

      print('recog:                  ', wers, file = sys.stdout)
      print('rerecog:                ', wers_opt, file = sys.stdout)
      print('errors:                 ', all_errors, file = sys.stdout) # TODO: print the errors here
      print('optimized am lm scales: ', optimized_scales, file=sys.stdout)


      print('best epoch:', best_epoch, file = sys.stdout)

      epoch_total = 200 if max_epoch <= 200 else 500 if max_epoch <= 500 else 600 if max_epoch <= 600 else 1000

      if epoch_total != max_epoch:

        print('\n', file=sys.stdout)
        print(f'finished epoch: {max_epoch}', file=sys.stdout)
        print(f'remaining time: {round(h_per_subepoch * (epoch_total - max_epoch), 3)} h', file=sys.stdout)

      print('##########################################################################', file=sys.stdout)

    wer_path = job_path + 'output/wers'

    if not os.path.exists(os.path.realpath(wer_path)) or args.overwrite:
      with open(os.path.realpath(wer_path), 'w') as file:
        print('recog:    ', wers, file=file)
        print('rerecog:  ', wers_opt, file=file)

        print('optimized am lm scales:     ', optimized_scales, file=file)
        print(f'num_params: {round(num_params / 1000000, 4)} mio.\n', file=file)

def main():

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--job_name", type=str, default="", help="name of the job", required=True)
  parser.add_argument(
      "--suffix", type=str, default="train.job")
  parser.add_argument(
    "--prefix", type=str, default="sprint_nn/train/conformer/switch")

  parser.add_argument(
    "--set", type=str, default="dev-other")

  parser.add_argument(
    "--recog_suffix", type=str, default="")
  parser.add_argument(
    "--print", type=bool, default=False, nargs="?",
      const=True
  )
  parser.add_argument(
    "--overwrite", type=bool, default=True, nargs="?",
    const=True
  )
  parser.add_argument(
    "--print_to", type=str, default="")

  args = parser.parse_args()

  write_wer(args)



if __name__ == "__main__":
  main()
