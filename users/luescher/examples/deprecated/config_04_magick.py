from sisyphus import *
Path = tk.Path

import sys

sys.setrecursionlimit(2500)

# ------------------------------ Recipes ------------------------------

import recipe.setups.michel.swb.baseline as swb_setups
import recipe.sprint as sprint
import recipe.am as am

# ------------------------------ Init ------------------------------

gs.ALIAS_AND_OUTPUT_SUBDIR = __file__[7:-3]
sprint.flow.FlowNetwork.default_flags = { 'cache_mode': 'bundle' }

s = swb_setups.SWBSystem()
s.cart_args['max_leaves'] = 9001

with tk.block("Baseline"):
  #s.run(steps={'init', 'monophone', 'triphone', 'vtln', 'sat', 'vtln+sat', 'nn'})
  s.run(steps={'init', 'nn'})

# ----------------------------- Add Oracle files  --------------------------
commonfiles = {
        "corpus": "/u/tuske/work/ASR/switchboard/corpus/xml/train.corpus.gz",
        "features": "/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.train.bundle",
        "lexicon": "/u/tuske/work/ASR/switchboard/corpus/train.lex.v1_0_3.ci.gz",
        "alignment": "/work/asr2/zeyer/setups-data/switchboard/2016-01-28--crnn/tuske__2016_01_28__align.combined.train",
        "cart": "/u/tuske/work/ASR/switchboard/initalign/data/cart-9000"
}

s.add_overlay('train','train_magick')
s.add_overlay('eval','eval_magick')
s.add_overlay('eval1','eval1_magick')

s.alignments['train_magick']['magick_alignment'] = sprint.NamedFlowAttribute('alignment', "/work/asr2/zeyer/setups-data/switchboard/2016-01-28--crnn/tuske__2016_01_28__align.combined.train")
s.cart_trees['magick'] = "/u/tuske/work/ASR/switchboard/initalign/data/cart-9000"
s.mixtures['train_magick']['split0'] = '/u/tuske/work/ASR/switchboard/singles/mfcc/data/0-split0-9000.mix'

for csp in [s.csp['train_magick'],s.csp['eval_magick'],s.csp['eval1_magick']]:
  csp.acoustic_model_config.state_tying.type            = 'cart'
  csp.acoustic_model_config.state_tying.file            = s.cart_trees['magick']
  csp.acoustic_model_config.allophones.add_all          = True
  csp.acoustic_model_post_config.allophones.add_from_file    = ""
  csp.acoustic_model_config.allophones.add_from_lexicon = False


import recipe.features as features
#Train Features
s.feature_caches ['train_magick']['gt'] = '/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.train.$(TASK)'
s.feature_bundles['train_magick']['gt'] = '/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.train.bundle'
feature_path = sprint.FlagDependentFlowAttribute('cache_mode', { 'task_dependent' : s.feature_caches ['train_magick']['gt'],
                                                                 'bundle'         : s.feature_bundles['train_magick']['gt'] })

s.feature_flows['train_magick']['gt'] = features.basic_cache_flow(feature_path)
#Eval features
s.feature_caches ['eval_magick']['gt'] = '/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.dev.$(TASK)'
s.feature_bundles['eval_magick']['gt'] = '/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.dev.bundle'
feature_path = sprint.FlagDependentFlowAttribute('cache_mode', { 'task_dependent' : s.feature_caches ['eval_magick']['gt'],
                                                                 'bundle'         : s.feature_bundles['eval_magick']['gt'] })
s.feature_flows['eval_magick']['gt'] = features.basic_cache_flow(feature_path)

s.feature_caches ['eval1_magick']['gt'] = '/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.hub5e_01.$(TASK)'
s.feature_bundles['eval1_magick']['gt'] = '/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.hub5e_01.bundle'
feature_path = sprint.FlagDependentFlowAttribute('cache_mode', { 'task_dependent' : s.feature_caches ['eval1_magick']['gt'],
                                                                 'bundle'         : s.feature_bundles['eval1_magick']['gt'] })
s.feature_flows['eval1_magick']['gt'] = features.basic_cache_flow(feature_path)


# normalize features
s.normalize('train_magick','gt',['train_magick','eval_magick'])

# Set eval corpus
import recipe.corpus       as corpus_recipes
s.concurrent['eval_magick'] = s.concurrent['eval']
s.corpora['eval_magick'] = s.corpora['eval']
s.corpora['eval_magick'].corpus_file = Path('/u/tuske/work/ASR/switchboard/corpus/xml/dev.corpus.gz')
j = corpus_recipes.SegmentCorpus(s.corpora['eval_magick'].corpus_file, s.concurrent['eval_magick'])
s.csp['eval_magick'].corpus_config.file = '/u/tuske/work/ASR/switchboard/corpus/xml/dev.corpus.gz'
s.csp['eval_magick'].segment_path = j.segment_path
#s.set_corpus('eval_magick', s.corpora['eval_magick'], s.concurrent['eval_magick'], j.segment_path)
s.jobs['eval_magick']['segment_corpus'] = j
#tk.register_output('%s.corpus.gz' % 'eval_magick', s.corpora['eval_magick'].corpus_file)

s.stm_files['eval_magick'] = Path('/u/tuske/bin/switchboard/hub5e_00.2.stm')
s.glm_files['eval_magick'] = s.glm_files['eval']

s.scorer_args['eval_magick'] = { 'ref': s.stm_files['eval_magick'], 'glm': s.glm_files['eval_magick'] }
s.scorer_hyp_arg['eval_magick'] = s.scorer_hyp_arg['eval']
s.scorers['eval_magick'] = s.scorers['eval']


s.concurrent['eval1_magick'] = s.concurrent['eval1']
s.corpora['eval1_magick'] = s.corpora['eval1']
s.corpora['eval1_magick'].corpus_file = Path('/u/tuske/work/ASR/switchboard/corpus/xml/hub5e_01.corpus.gz')
j = corpus_recipes.SegmentCorpus(s.corpora['eval1_magick'].corpus_file, s.concurrent['eval1_magick'])
s.csp['eval1_magick'].corpus_config.file = '/u/tuske/work/ASR/switchboard/corpus/xml/hub5e_01.corpus.gz'
s.csp['eval1_magick'].segment_path = j.segment_path
s.jobs['eval1_magick']['segment_corpus'] = j


s.stm_files['eval1_magick'] = Path('/u/tuske/bin/switchboard/hub5e_01.2.stm')
s.glm_files['eval1_magick'] = s.glm_files['eval1']

s.scorer_args['eval1_magick'] = { 'ref': s.stm_files['eval1_magick'], 'glm': s.glm_files['eval1_magick'] }
s.scorer_hyp_arg['eval1_magick'] = s.scorer_hyp_arg['eval1']
s.scorers['eval1_magick'] = s.scorers['eval1']


# ------------------------------ NN training ------------------------------

import recipe.crnn.helpers.beck as crnn_helpers

def make_config(num_input=16, layers=None, l2=0.0, lr=1e-4, dropout=0.1, **kwargs):
  return crnn_helpers.blstm_config(num_input, crnn_helpers.blstm_network(layers, dropout, l2), lr, **kwargs)

def make_ffconfig(num_input=16, layers=None, activation='relu', l2=0.0, lr=1e-4, dropout=0.1, **kwargs):
  conf = crnn_helpers.feed_forward_config(num_input, crnn_helpers.mlp_network(layers,activation, dropout,l2),lr,**kwargs)
  conf.__delitem__('pretrain')
  return conf

def nn(name, crnn_config, prior_scale=0.30, am_scale=3.0, lm_scale=14.1,
       prior_mixtures=('train_magick', 'split0'), training_args=None, recognition_args=None, search_parameters=None, **kwargs):
  sp = { 'beam-pruning' : 20 }
  if search_parameters is not None:
    sp.update(search_parameters)
  r_args = { 'corpus': 'eval_magick',
            'pronunciation_scale'   : am_scale,
            'lm_scale'              : lm_scale,
            'search_parameters'     : sp,
            'lattice_to_ctm_kwargs' : { 'fill_empty_segments' : True, 'best_path_algo': 'bellman-ford' },
            'flow'                  : training_args.get('feature_flow','mfcc+context1') }
  if recognition_args is not None:
      r_args.update(recognition_args)
  if training_args is None:
    training_args = {}
  training_args['alignment'] = ('train_magick','magick_alignment', -1)
  s.nn(name             = name,
       training_args    = training_args,
       crnn_config      = crnn_config,
       scorer_args      = { 'prior_scale' : prior_scale,
                            'prior_mixtures': prior_mixtures,
                            'feature_dimension': crnn_config['num_outputs']['data'][0]},
       recognition_args = r_args,
       **kwargs)

b = make_config(num_input  = 40,
                layers     = 5*[500],
                l2         = 0.01,
                lr         = 0.0005,
                dropout    = 0.1,
                batch_size = 5000,
                adam       = False,
                nadam      = True,
                gradient_clip = 0,
                learning_rate_control = "newbob_multi_epoch",
                update_on_device = True,
                cache_size = "0",
                batching = "random",
                chunking = "50:25",
                truncation = -1,
                pretrain = "default",
                pretrain_construction_algo = "from_input",
                gradient_noise = 0.3,
                learning_rate_control_relative_error_relative_lr = True,
                newbob_multi_num_epochs = 6,
                newbob_multi_update_interval = 1,
)

import copy
b2 = copy.deepcopy(b)
b2["partition_epochs"] = {'train': 6, 'dev': 1}


baseline = make_config(num_input=40,
                layers=5 * [500],
                l2=0.01,
                lr=0.0005,
                dropout=0.1,
                batch_size=5000,
                adam=True,
                gradient_clip=0,
                learning_rate_control="newbob_multi_epoch",
                update_on_device=True,
                cache_size="0",
                batching="random",
                chunking="50:25",
                truncation=-1,
                gradient_noise=0.3,
                learning_rate_control_relative_error_relative_lr=True,
                newbob_multi_num_epochs=6,
                newbob_multi_update_interval=1,
                optimizer_epsilon = 0.1,
                use_tensorflow = True,
                multiprocessing = True,
                partition_epochs = {'train': 6, 'dev': 1}
                )
del baseline['max_seq_length']
baseline_sorted = copy.deepcopy(baseline)
baseline_sorted["batching"]="sorted"
baseline_seq = copy.deepcopy(baseline)
baseline_seq["learning_rate"] = 0.000005
baseline_seq_sorted = copy.deepcopy(baseline_seq)
baseline_seq_sorted["batching"]="sorted"

c = make_config(num_input  = 40,
                layers     = 4*[512],
                l2         = 0.001,
                lr         = 0.00025,
                dropout    = 0.1,
                batch_size = 10000)

d = make_ffconfig(num_input     = 40,
                  layers        = 6*[1024],
                  l2            = 0.001,
                  batch_size    = 10000,
                  gradient_clip = 10,
                  window        = 9)


#nn('magick_first', c, training_args={'feature_corpus':'train_magick', 'feature_flow':'gt'})
nn('magick_albert1', b, training_args = {'feature_corpus':'train_magick','feature_flow': 'gt', 'num_classes':9001, 'mem_rqmt': 16,'time_rqmt': 120})
nn('magick_albert2', b2, training_args = {'feature_corpus':'train_magick','feature_flow': 'gt', 'num_classes':9001, 'mem_rqmt': 16,'time_rqmt': 80, 'num_epochs': 96}, epochs = [16, 32, 48, 64, 72, 80, 88, 96])
nn('magick_baseline', baseline, training_args = {'feature_corpus':'train_magick','feature_flow': 'gt', 'num_classes':9001, 'mem_rqmt': 16, 'num_epochs': 96, 'time_rqmt': 120}, epochs = [12,24,36,48,60,72,84,96])
#nn('magick_ffnn', d, training_args={'feature_corpus':'train_magick','feature_flow':'gt'})

#### retrain with multiple rng seeds ##
dev_size = s.init_nn_args['dev_size']
name = s.init_nn_args['name']
corpus = s.init_nn_args['corpus']

old_train_segments = s.csp['%s_train' % name].segment_path
old_dev_segments   = s.csp['%s_dev' % name].segment_path

with tk.block("Train with shuffled segments"):
    for i,rng_seed in enumerate([7593347,77496457,1636039,34789693,8384293]): # these are some random primes i picked
        all_segments = s.jobs[corpus]['all_segments_%s' % name]
        new_segments = corpus_recipes.ShuffleAndSplitSegments(segment_file = all_segments.single_segment_files[1],
                                                              split        = { 'train': 1.0 - dev_size, 'dev': dev_size },
                                                              shuffle_seed = rng_seed)

        s.jobs[corpus]['segment%i' % i] = new_segments

        s.csp['%s_train' % name].segment_path = new_segments.new_segments['train']
        s.csp['%s_dev' % name].segment_path = new_segments.new_segments['dev']
        nn('magick_baseline%d' % i, baseline, training_args = {'feature_corpus':'train_magick','feature_flow': 'gt', 'num_classes':9001, 'mem_rqmt': 16, 'num_epochs': 96, 'time_rqmt': 120}, epochs = [12,24,36,48,60,72,84,96])
        nn('magick_baseline%d_eval1' % i, baseline, training_args={'feature_corpus': 'train_magick', 'feature_flow': 'gt', 'num_classes': 9001, 'mem_rqmt': 16, 'num_epochs': 96, 'time_rqmt': 120}, recognition_args={'corpus':'eval1_magick'}, epochs=[84, 96])

    nn('magick_baseline%d_norm' % i, baseline, training_args = {'feature_corpus':'train_magick','feature_flow': 'gt+norm', 'num_classes':9001, 'mem_rqmt': 16, 'num_epochs': 96, 'time_rqmt': 120}, epochs = [12,24,36,48,60,72,84,96])

# Try my own sorted segment list
with tk.block("Train with sorted segments"):
    for suffix in [""]+list(range(5)):
        s.csp['%s_train' % name].segment_path = Path("/u/michel/setups/SWB_sis/dependencies/seg_train_sorted%s" % suffix)
        s.csp['%s_dev' % name].segment_path = Path("/u/michel/setups/SWB_sis/dependencies/seg_cv")
        nn('magick_baseline_sorted%s' % suffix, baseline_sorted, training_args = {'feature_corpus':'train_magick','feature_flow': 'gt', 'num_classes':9001, 'mem_rqmt': 16, 'num_epochs': 96, 'time_rqmt': 120}, epochs = [12,24,36,48,60,72,84,96])
        nn('magick_baseline_sorted%s_eval1' % suffix, baseline_sorted, training_args={'feature_corpus': 'train_magick', 'feature_flow': 'gt', 'num_classes': 9001, 'mem_rqmt': 16, 'num_epochs': 96, 'time_rqmt': 120}, recognition_args={'corpus':'eval1_magick'}, epochs=[84, 96])

    nn('magick_baseline_sorted%s_norm' % suffix, baseline_sorted, training_args = {'feature_corpus':'train_magick','feature_flow': 'gt+norm', 'num_classes':9001, 'mem_rqmt': 16, 'num_epochs': 96, 'time_rqmt': 120}, epochs = [12,24,36,48,60,72,84,96])

    s.csp['%s_train' % name].segment_path = Path("/u/michel/setups/SWB_sis/dependencies/seg_train_isorted")
    s.csp['%s_dev' % name].segment_path = Path("/u/michel/setups/SWB_sis/dependencies/seg_cv")
    nn('magick_baseline_isorted', baseline_sorted, training_args = {'feature_corpus':'train_magick','feature_flow': 'gt', 'num_classes':9001, 'mem_rqmt': 16, 'num_epochs': 96, 'time_rqmt': 120}, epochs = [12,24,36,48,60,72,84,96])


    s.csp['%s_train' % name].segment_path = old_train_segments
    s.csp['%s_dev' % name].segment_path = old_dev_segments

# Used wrong sorting option, try again
with tk.block("Train with sorted segments"):
    for suffix in list(range(2)):
        s.csp['%s_train' % name].corpus_config.segment_order = Path("/u/michel/setups/SWB_sis/dependencies/seg_train_sorted%s" % suffix)
        s.csp['%s_dev' % name].segment_path = Path("/u/michel/setups/SWB_sis/dependencies/seg_cv")
        nn('magick_baseline_sorted%s' % str(suffix+5), baseline_sorted, training_args = {'feature_corpus':'train_magick','feature_flow': 'gt', 'num_classes':9001, 'mem_rqmt': 16, 'num_epochs': 96, 'time_rqmt': 120}, epochs = [12,24,36,48,60,72,84,96])
        nn('magick_baseline_sorted%s_eval1' % str(suffix+5), baseline_sorted, training_args={'feature_corpus': 'train_magick', 'feature_flow': 'gt', 'num_classes': 9001, 'mem_rqmt': 16, 'num_epochs': 96, 'time_rqmt': 120}, recognition_args={'corpus':'eval1_magick'}, epochs=[84, 96])

    del s.csp['%s_train' % name].corpus_config["segment-order"]
    s.csp['%s_train' % name].segment_path = old_train_segments
    s.csp['%s_dev' % name].segment_path = old_dev_segments


#--------------------------- Extract Allophones ---------------------------------
from recipe.allophones import StoreAllophones

j = StoreAllophones(s.csp['train_magick'])
s.jobs['train_magick']['extract_allophones'] = j

tk.register_output("allophones",j.allophone_file)


# ------------------------------ Sequence Training ------------------------------
from experimental.michel.sequence_training import add_accuracy_output

# optimize lm scale for 4-gram LM
with tk.block("Optimize LM scale"):
  s.add_overlay('eval_magick','eval_seq')
  s.scorer_args['eval_seq']     = s.scorer_args['eval_magick']
  s.scorer_hyp_arg['eval_seq']  = s.scorer_hyp_arg['eval_magick']
  s.scorers['eval_seq']         = s.scorers['eval_magick']
  s.add_overlay('train_magick','train_seq')
  for corpus in ['train_seq','eval_seq']:
    s.csp[corpus].lexicon_config.file = '/home/tuske/work/ASR/switchboard/corpus/train.lex.v1_0_4.ci.gz'
    s.csp[corpus].language_model_config       = sprint.SprintConfig()
    s.csp[corpus].language_model_config.type  = 'ARPA'
    s.csp[corpus].language_model_config.file  = '/u/tuske/work/ASR/switchboard/corpus/lm/data/mylm2/swb.fsh.4gr.voc30k.LM.gz' #'/work/asr2/golik/20150106-swb-init-train/step070.recognition/lm/lower_order/swb.fsh.2gr.voc30k.LM.gz'
    s.csp[corpus].language_model_config.scale = 12.1

  s.recog('baseline_seq_opt','eval_seq','gt',s.feature_scorers['train_magick']['magick_baseline1-84'],6.0,12.1)
  s.optimize_am_lm('recog_baseline_seq_opt','eval_seq',6.0,12.1)
  s.jobs['eval_seq']['optimize_recog_baseline_seq_opt'].rqmt = { 'time': 48, 'cpu': 1, 'mem': 5 }
  s.csp['train_seq'].language_model_config.scale = s.jobs['eval_seq']['optimize_recog_baseline_seq_opt'].best_lm_score

  lattice_options = {'lm' : {'file' : '/u/tuske/work/ASR/switchboard/corpus/lm/data/mylm2/swb.fsh.4gr.voc30k.LM.gz', #'/work/asr2/golik/20150106-swb-init-train/step070.recognition/lm/lower_order/swb.fsh.2gr.voc30k.LM.gz',
                             'type' : 'ARPA', 'scale' : s.jobs['eval_seq']['optimize_recog_baseline_seq_opt'].best_lm_score}}

with tk.block("Sequence Training Baseline1"):
    # lattice generation
    s.generate_lattices('baseline1', 'train_magick',
                        s.feature_scorers['train_magick']['magick_baseline1-84'],
                        s.feature_flows['train_magick']['gt'], lattice_options)

    # add prior file
    feature_scorer = copy.deepcopy(s.feature_scorers['train_magick']['magick_baseline1-84'])
    feature_scorer.config.priori_scale = 0.0
    score_features = am.ScoreFeaturesJob(csp=s.csp['train_magick'],
                                         feature_flow=s.feature_flows['train_seq']['gt'],
                                         feature_scorer=feature_scorer,
                                         normalize=True,
                                         plot_prior=True,
                                         rtf=20.0)
    s.feature_scorers['train_magick']['magick_baseline1-84-prior'] = copy.deepcopy(s.feature_scorers['train_magick']['magick_baseline1-84'])
    s.feature_scorers['train_magick']['magick_baseline1-84-prior'].config.priori_scale = 0.6
    s.feature_scorers['train_magick']['magick_baseline1-84-prior'].config.prior_file = score_features.prior

    # seq train random order
    new_segments = s.jobs[s.init_nn_args['corpus']]['segment1']
    s.csp['%s_train' % name].segment_path = new_segments.new_segments['train']
    s.csp['%s_dev' % name].segment_path = new_segments.new_segments['dev']

    baseline_seq_sorted, add_sprint_conf, add_print_post_conf = add_accuracy_output(s.csp['train_seq'],baseline_seq_sorted,s.lattice_bundles['train_magick']['state-accuracy_baseline1'],
                                                                                    s.alignments['train_magick']['state-accuracy_baseline1'],s.feature_scorers['train_magick']['magick_baseline1-84'],
                                                                                    s.feature_flows['train_magick']['gt'],import_model=s.nn_models['train_magick']['magick_baseline1'][84].model)

    t_args = {'additional_sprint_config_files': add_sprint_conf,
              'additional_sprint_post_config_files': add_print_post_conf,
              'feature_corpus': 'train_magick',
              'feature_flow': 'gt',
              'num_classes': 9001,
              'time_rqmt': 120,
              'num_epochs': 16,
              'mem_rqmt': 16}

    nn('baseline1_seq_rand_from_rand', baseline_seq_sorted, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16])
    nn('baseline1_seq_rand_from_rand_eval1', baseline_seq_sorted, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16],
       recognition_args={'corpus': 'eval1_magick'})

    # activate sorted segment order
    s.csp['%s_train' % name].corpus_config.segment_order = Path(
        "/u/michel/setups/SWB_sis/dependencies/seg_train_sorted1")
    nn('baseline1_seq_sort_from_rand', baseline_seq_sorted, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16])
    nn('baseline1_seq_sort_from_rand_eval1', baseline_seq_sorted, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16],
       recognition_args={'corpus': 'eval1_magick'})
    del s.csp['%s_train' % name].corpus_config["segment-order"]

    #activate Prior
    baseline_seq_sorted, add_sprint_conf_p, add_print_post_conf_p = add_accuracy_output(s.csp['train_seq'],baseline_seq_sorted, s.lattice_bundles['train_magick']['state-accuracy_baseline1'],
                                                                                    s.alignments['train_magick']['state-accuracy_baseline1'],s.feature_scorers['train_magick']['magick_baseline1-84-prior'],
                                                                                    s.feature_flows['train_magick']['gt'], import_model=s.nn_models['train_magick']['magick_baseline1'][84].model)
    t_args['additional_sprint_config_files']=add_sprint_conf_p
    nn('baseline1_seq_rand_from_rand-prior', baseline_seq_sorted, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16])
    nn('baseline1_seq_rand_from_rand-prior_eval1', baseline_seq_sorted, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16],
       recognition_args={'corpus': 'eval1_magick'})

    # activate sorted segment order
    s.csp['%s_train' % name].corpus_config.segment_order = Path(
        "/u/michel/setups/SWB_sis/dependencies/seg_train_sorted1")
    nn('baseline1_seq_sort_from_rand-prior', baseline_seq_sorted, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16])
    nn('baseline1_seq_sort_from_rand-prior_eval1', baseline_seq_sorted, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16],
       recognition_args={'corpus': 'eval1_magick'})
    del s.csp['%s_train' % name].corpus_config["segment-order"]



with tk.block("Sequence Training Baseline1 with larger lattice"):
    tmpconf = sprint.SprintConfig()
    tmpconf.pronunciation_scale = s.jobs['eval_seq']['optimize_recog_baseline_seq_opt'].best_am_score
    new_lattice_options = {'lm' : {'file' : '/u/tuske/work/ASR/switchboard/corpus/lm/data/mylm2/swb.fsh.4gr.voc30k.LM.gz', #'/work/asr2/golik/20150106-swb-init-train/step070.recognition/lm/lower_order/swb.fsh.2gr.voc30k.LM.gz',
                             'type' : 'ARPA', 'scale' : s.jobs['eval_seq']['optimize_recog_baseline_seq_opt'].best_lm_score},
                       'raw-denominator_options': {'search_parameters': {'beam-pruning': 20,
                                                                         'word-end-pruning': 0.8},
                                                   'model_combination_config': tmpconf},
                       'denominator_options': {'search_options': {'pruning-threshold': 18}}
                       }

    # lattice generation
    s.generate_lattices('baseline1_l', 'train_magick',
                        s.feature_scorers['train_magick']['magick_baseline1-84'],
                        s.feature_flows['train_magick']['gt'], new_lattice_options)


    # seq train random order
    new_segments = s.jobs[s.init_nn_args['corpus']]['segment1']
    s.csp['%s_train' % name].segment_path = new_segments.new_segments['train']
    s.csp['%s_dev' % name].segment_path = new_segments.new_segments['dev']

    baseline_seq_sorted, add_sprint_conf, add_print_post_conf = add_accuracy_output(s.csp['train_seq'],baseline_seq_sorted,s.lattice_bundles['train_magick']['state-accuracy_baseline1_l'],
                                                                                    s.alignments['train_magick']['state-accuracy_baseline1_l'],s.feature_scorers['train_magick']['magick_baseline1-84'],
                                                                                    s.feature_flows['train_magick']['gt'],import_model=s.nn_models['train_magick']['magick_baseline1'][84].model)

    t_args = {'additional_sprint_config_files': add_sprint_conf,
              'additional_sprint_post_config_files': add_print_post_conf,
              'feature_corpus': 'train_magick',
              'feature_flow': 'gt',
              'num_classes': 9001,
              'time_rqmt': 120,
              'num_epochs': 16,
              'mem_rqmt': 16}

    nn('baseline1_seq_rand_from_rand_large', baseline_seq_sorted, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16])
    nn('baseline1_seq_rand_from_rand_lagre_eval1', baseline_seq_sorted, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16],
       recognition_args={'corpus': 'eval1_magick'})

with tk.block("Sequence Training Baseline1s"):
    # lattice generation
    s.generate_lattices('baseline_sorted1', 'train_magick', s.feature_scorers['train_magick']['magick_baseline_sorted1-84'], s.feature_flows['train_magick']['gt'],lattice_options)

    # seq train random order
    s.csp['%s_train' % name].segment_path = Path("/u/michel/setups/SWB_sis/dependencies/seg_train_sorted1")
    s.csp['%s_dev' % name].segment_path = Path("/u/michel/setups/SWB_sis/dependencies/seg_cv")

    baseline_seq_sorted, add_sprint_conf, add_print_post_conf = add_accuracy_output(s.csp['train_seq'], baseline_seq_sorted, s.lattice_bundles['train_magick']['state-accuracy_baseline_sorted1'],s.alignments['train_magick']['state-accuracy_baseline_sorted1'],
                                                                      s.feature_scorers['train_magick']['magick_baseline_sorted1-84'], s.feature_flows['train_magick']['gt'],
                                                                      import_model=s.nn_models['train_magick']['magick_baseline_sorted1'][84].model)

    t_args = {'additional_sprint_config_files':       add_sprint_conf,
            'additional_sprint_post_config_files':  add_print_post_conf,
            'feature_corpus':'train_magick',
            'feature_flow': 'gt',
            'num_classes':9001,
            'time_rqmt': 120,
            'num_epochs':                           16,
            'mem_rqmt':                             16}

    nn('baseline_seq_rand_from_rand', baseline_seq_sorted, training_args=t_args, epochs=[2,4,6,8,12,16])
    nn('baseline_seq_rand_from_rand_eval1', baseline_seq_sorted, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16], recognition_args = { 'corpus' : 'eval1_magick'})

    # activate sorted segment order
    s.csp['%s_train' % name].corpus_config.segment_order = Path("/u/michel/setups/SWB_sis/dependencies/seg_train_sorted1")
    nn('baseline_seq_sort_from_rand', baseline_seq_sorted, training_args=t_args, epochs=[2,4,6,8,12,16])
    nn('baseline_seq_sort_from_rand_eval1', baseline_seq_sorted, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16], recognition_args = { 'corpus' : 'eval1_magick'})
    del s.csp['%s_train' % name].corpus_config["segment-order"]


with tk.block("Sequence Training Alberts Model"):
    model_file = Path('/u/zeyer/setups/switchboard/2016-12-27--tf-crnn/data-train/dropout01.l2_1e_2.5l.n500.max_seqs40.grad_noise03.adam.oeps01.lr05e_3.nbm6.nbrl.grad_clip_inf/net-model/network.080.meta')
    config_file = Path('/u/michel/setups/SWB_sis/dependencies/dropout01.l2_1e_2.5l.n500.max_seqs40.grad_noise03.adam.oeps01.lr05e_3.nbm6.nbrl.grad_clip_inf.config')
    prior_mixture = Path('/u/tuske/work/ASR/switchboard/singles/mfcc/data/0-split0-9000.mix')
    import recipe.crnn as crnn
    model_albert = crnn.CRNNModel(config_file,model_file,80)
    feature_scorer_albert = sprint.CRNNScorer(40,9001,prior_mixture, model_albert)

    s.recog_and_optimize('albert_baseline','eval_magick','gt',feature_scorer_albert,3.0,14.1)

    # lattice generation
    s.generate_lattices('lattice.albert', 'train_magick', feature_scorer_albert,
                        s.feature_flows['train_magick']['gt'], lattice_options)

    # add prior file
    feature_scorer = copy.deepcopy(feature_scorer_albert)
    feature_scorer.config.priori_scale = 0.0
    score_features = am.ScoreFeaturesJob(csp=s.csp['train_magick'],
                                         feature_flow=s.feature_flows['train_seq']['gt'],
                                         feature_scorer=feature_scorer,
                                         normalize=True,
                                         plot_prior=True,
                                         rtf=20.0)
    feature_scorer_albert_prior = copy.deepcopy(feature_scorer_albert)
    feature_scorer_albert_prior.config.priori_scale = 0.6
    feature_scorer_albert_prior.config.prior_file = score_features.prior

    # seq train random order
    s.csp['%s_train' % name].segment_path = Path("/u/michel/setups/SWB_sis/dependencies/seg_train")
    s.csp['%s_dev' % name].segment_path = Path("/u/michel/setups/SWB_sis/dependencies/seg_cv")

    baseline_seq_sorted, add_sprint_conf, add_print_post_conf = add_accuracy_output(s.csp['train_seq'],baseline_seq_sorted,s.lattice_bundles['train_magick']['state-accuracy_lattice.albert'],
                                                                                    s.alignments['train_magick']['state-accuracy_lattice.albert'],feature_scorer_albert_prior,
                                                                                    s.feature_flows['train_magick']['gt'],import_model=model_albert.model)

    t_args = {'additional_sprint_config_files': add_sprint_conf,
              'additional_sprint_post_config_files': add_print_post_conf,
              'feature_corpus': 'train_magick',
              'feature_flow': 'gt',
              'num_classes': 9001,
              'time_rqmt': 120,
              'num_epochs': 16,
              'mem_rqmt': 16}

    nn('albert_seq', baseline_seq_sorted, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16])
    nn('albert_seq_eval1', baseline_seq_sorted, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16],
       recognition_args={'corpus': 'eval1_magick'})

    baseline_seq_sorted, add_sprint_conf, add_print_post_conf = add_accuracy_output(s.csp['train_seq'],baseline_seq_sorted,s.lattice_bundles['train_magick']['state-accuracy_lattice.albert'],
                                                                                    s.alignments['train_magick']['state-accuracy_lattice.albert'],feature_scorer_albert,
                                                                                    s.feature_flows['train_magick']['gt'],import_model=model_albert.model)

    t_args['additional_sprint_config_files'] = add_sprint_conf

    nn('albert_seq_noprior', baseline_seq_sorted, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16])
    nn('albert_seq_noprior_eval1', baseline_seq_sorted, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16],
       recognition_args={'corpus': 'eval1_magick'})

# # ---------- Best random ce-baseline -------------
# with tk.block("Sequence Training Baseline1"):
#     # lattice generation
#     s.generate_lattices('baseline1', 'train_magick', s.feature_scorers['train_magick']['magick_baseline1-84'], s.feature_flows['train_magick']['gt'],lattice_options)
#
#     # Estimate sdm with dim 16
#     #s.single_density_mixtures('sdm_16','train','mfcc',('train', 'train_tri', -1))
#
#     # random seq train starting from random model
#     new_segments = s.jobs[s.init_nn_args['corpus']]['segment1']
#     s.csp['%s_train' % name].segment_path = new_segments.new_segments['train']
#     s.csp['%s_dev' % name].segment_path = new_segments.new_segments['dev']
#
#     baseline_seq, add_sprint_conf, add_print_post_conf = add_accuracy_output(s.csp['train_seq'], baseline_seq, s.lattice_bundles['train_magick']['state-accuracy_baseline1'],s.alignments['train_magick']['state-accuracy_baseline1'],
#
#                                                                       s.feature_scorers['train_magick']['magick_baseline1-84'], s.feature_flows['train_magick']['gt'],
#                                                                       import_model=s.nn_models['train_magick']['magick_baseline1'][84].model,ce_smoothing=0.1)
#
#     t_args = {'additional_sprint_config_files':       add_sprint_conf,
#             'additional_sprint_post_config_files':  add_print_post_conf,
#             'use_python_control':                   True,
#             'feature_corpus':'train_magick',
#             'feature_flow': 'gt',
#             'num_classes':9001,
#             'time_rqmt': 120,
#             'num_epochs':                           16,
#             'mem_rqmt':                             16}
#
#     nn('baseline_seq_rand_from_rand', baseline_seq, training_args=t_args, epochs=[2,4,6,8,12,16])
#     nn('baseline_seq_rand_from_rand_eval1', baseline_seq, training_args=t_args, epochs=[2,4,6,8,12,16], recognition_args = { 'corpus' : 'eval1_magick'})
#
#    # random seq train starting from sorted model
#    baseline_seq['import_model_train_epoch1'] = str(s.nn_models['train_magick']['magick_baseline_sorted1'][84].model)[:-5]
#    nn('baseline_seq_rand_from_sorted', baseline_seq, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16])
#    nn('baseline_seq_rand_from_sorted_eval1', baseline_seq, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16],recognition_args={'corpus': 'eval1_magick'})

# # ---------- Best sorted ce-baseline -------------
# with tk.block("Sequence Training Baseline1 sorted"):
#     # lattice generation
#     s.generate_lattices('baseline_sorted1', 'train_magick', s.feature_scorers['train_magick']['magick_baseline_sorted1-84'], s.feature_flows['train_magick']['gt'],lattice_options)
#
#     # sorted seq train starting from sorted model
#     s.csp['%s_train' % name].segment_path = Path("/u/michel/setups/SWB_sis/dependencies/seg_train_sorted1")
#     s.csp['%s_dev' % name].segment_path = Path("/u/michel/setups/SWB_sis/dependencies/seg_cv")
#
#     baseline_seq_sorted, add_sprint_conf, add_print_post_conf = add_accuracy_output(s.csp['train_seq'], baseline_seq_sorted, s.lattice_bundles['train_magick']['state-accuracy_baseline_sorted1'],s.alignments['train_magick']['state-accuracy_baseline_sorted1'],
#                                                                       s.feature_scorers['train_magick']['magick_baseline_sorted1-84'], s.feature_flows['train_magick']['gt'],
#                                                                       import_model=s.nn_models['train_magick']['magick_baseline_sorted1'][84].model)
#
#     t_args = {'additional_sprint_config_files':       add_sprint_conf,
#             'additional_sprint_post_config_files':  add_print_post_conf,
#             'feature_corpus':'train_magick',
#             'feature_flow': 'gt',
#             'num_classes':9001,
#             'time_rqmt': 120,
#             'num_epochs':                           16,
#             'mem_rqmt':                             16}
#
#     nn('baseline_seq_sorted_from_sorted', baseline_seq_sorted, training_args=t_args, epochs=[2,4,6,8,12,16])
#     nn('baseline_seq_sorted_from_sorted_eval1', baseline_seq_sorted, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16], recognition_args = { 'corpus' : 'eval1_magick'})
#
#     # sorted seq train starting from sorted model
#     baseline_seq_sorted['import_model_train_epoch1'] = str(s.nn_models['train_magick']['magick_baseline1'][84].model)[:-5]
#     nn('baseline_seq_sorted_sorted_from_rand', baseline_seq_sorted, training_args=t_args, epochs=[2,4,6,8,12,16])
#     nn('baseline_seq_sorted_sorted_from_rand_eval1', baseline_seq_sorted, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16], recognition_args = { 'corpus' : 'eval1_magick'})


# optimize lm scale for 2-gram LM
with tk.block("Optimize LM scale 2g"):
  s.add_overlay('eval_magick','eval_seq')
  s.scorer_args['eval_seq']     = s.scorer_args['eval_magick']
  s.scorer_hyp_arg['eval_seq']  = s.scorer_hyp_arg['eval_magick']
  s.scorers['eval_seq']         = s.scorers['eval_magick']
  s.add_overlay('train_magick','train_seq')
  for corpus in ['train_seq','eval_seq']:
    s.csp[corpus].lexicon_config.file = '/home/tuske/work/ASR/switchboard/corpus/train.lex.v1_0_4.ci.gz'
    s.csp[corpus].language_model_config       = sprint.SprintConfig()
    s.csp[corpus].language_model_config.type  = 'ARPA'
    s.csp[corpus].language_model_config.file  = '/u/michel/setups/SWB/seq.train/dependencies/mylm/swb.fsh.2gr.voc30k.LM.gz' #'/work/asr2/golik/20150106-swb-init-train/step070.recognition/lm/lower_order/swb.fsh.2gr.voc30k.LM.gz'
    s.csp[corpus].language_model_config.scale = 12.1

  s.recog('baseline_seq_opt_2g','eval_seq','gt',s.feature_scorers['train_magick']['magick_baseline1-84'],6.0,12.1)
  s.optimize_am_lm('recog_baseline_seq_opt_2g','eval_seq',6.0,12.1)
  s.jobs['eval_seq']['optimize_recog_baseline_seq_opt_2g'].rqmt = { 'time': 48, 'cpu': 1, 'mem': 5 }
  s.csp['train_seq'].language_model_config.scale = s.jobs['eval_seq']['optimize_recog_baseline_seq_opt_2g'].best_lm_score

  lattice_options = {'lm' : {'file' : '/u/michel/setups/SWB/seq.train/dependencies/mylm/swb.fsh.2gr.voc30k.LM.gz',
                             'type' : 'ARPA', 'scale' : s.jobs['eval_seq']['optimize_recog_baseline_seq_opt_2g'].best_lm_score}}


# ---------- Best random ce-baseline -------------
with tk.block("Sequence Training Baseline1 2g"):
    # lattice generation
    s.generate_lattices('baseline1', 'train_magick', s.feature_scorers['train_magick']['magick_baseline1-84'], s.feature_flows['train_magick']['gt'],lattice_options)

    # model training
    new_segments = s.jobs[s.init_nn_args['corpus']]['segment1']
    s.csp['%s_train' % name].segment_path = new_segments.new_segments['train']
    s.csp['%s_dev' % name].segment_path = new_segments.new_segments['dev']

    baseline_seq, add_sprint_conf, add_print_post_conf = add_accuracy_output(s.csp['train_seq'], baseline_seq, s.lattice_bundles['train_magick']['state-accuracy_baseline1'],s.alignments['train_magick']['state-accuracy_baseline1'],

                                                                      s.feature_scorers['train_magick']['magick_baseline1-84'], s.feature_flows['train_magick']['gt'],
                                                                      import_model=s.nn_models['train_magick']['magick_baseline1'][84].model,ce_smoothing=0.1)

    t_args = {'additional_sprint_config_files': add_sprint_conf,
             'additional_sprint_post_config_files': add_print_post_conf,
             'use_python_control': True,
             'feature_corpus': 'train_magick',
             'feature_flow': 'gt',
             'num_classes': 9001,
             'time_rqmt': 120,
             'num_epochs': 16,
             'mem_rqmt': 16}

    nn('baseline_seq_2g', baseline_seq, training_args=t_args, epochs=[2,4,6,8,12,16])
    nn('baseline_seq_2g_eval1', baseline_seq, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16], recognition_args = { 'corpus' : 'eval1_magick'})


    baseline_seq, add_sprint_conf, add_print_post_conf = add_accuracy_output(s.csp['train_seq'], baseline_seq, s.lattice_bundles['train_magick']['state-accuracy_baseline1'],
                                                                             s.alignments['train_magick']['state-accuracy_baseline1'],s.feature_scorers['train_magick']['magick_baseline1-84-prior'],
                                                                             s.feature_flows['train_magick']['gt'], import_model=s.nn_models['train_magick']['magick_baseline1'][84].model,
                                                                             ce_smoothing=0.1)

    t_args = {'additional_sprint_config_files': add_sprint_conf,
              'additional_sprint_post_config_files': add_print_post_conf,
              'use_python_control': True,
              'feature_corpus': 'train_magick',
              'feature_flow': 'gt',
              'num_classes': 9001,
              'time_rqmt': 120,
              'num_epochs': 16,
              'mem_rqmt': 16}

    nn('baseline_seq_2g-prior', baseline_seq, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16])
    nn('baseline_seq_2g-prior_eval1', baseline_seq, training_args=t_args, epochs=[2, 4, 6, 8, 12, 16],
       recognition_args={'corpus': 'eval1_magick'})
