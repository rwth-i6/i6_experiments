from recipe.setups.mann.librispeech import LibriNNSystem
import recipe.sprint as sprint

#config
lm = "/work/asr3/irie/data/librispeech/lm/seq_train/960hr/kn2.no_pruning.gz"
corpus = '/work/speech/golik/setups/librispeech/resources/corpus/corpus.' + corpus_file + '.corpus.gz'

# ------------------------------ Init ------------------------------

gs.ALIAS_AND_OUTPUT_SUBDIR = __file__[7:-3]

feature_mapping = {corpus_file: PREFIX_PATH + "FeatureExtraction.Gammatone.tp4cEAa0YLIP/output/gt.cache.",
                        'dev-clean' : "/u/michel/setups/librispeech/work/features/extraction/FeatureExtraction.Gammatone.DA0TtL8MbCKI/output/gt.cache.",
                        "test-clean" : "/u/luescher/setups/librispeech/2018-12-18--960h/work/features/extraction/FeatureExtraction.Gammatone.INa6z5A4JvZ5/output/gt.cache.",
                        "dev-other" : "/u/luescher/setups/librispeech/2018-12-18--960h/work/features/extraction/FeatureExtraction.Gammatone.qrINHi3yh3GH/output/gt.cache.",
                        "test-other" : "/u/luescher/setups/librispeech/2018-12-18--960h/work/features/extraction/FeatureExtraction.Gammatone.qqN3kYqQ6QHF/output/gt.cache.",
                       }

features = {cv : {'caches': util.MultiPath(path + "$(TASK)",
                                {i: path + "{}".format(i) for i in range(librispeech.concurrent['train'])}, cached=True),
                  'bundle': Path(path + "bundle", cached=True)}
                for cv, path in feature_mapping.items()
           }

with tk.block("Init"):
  # set files from chris
  PREFIX_PATH1K = "/work/asr3/luescher/setups-data/librispeech/best-model/960h_2019-04-10/"
  PREFIX_PATH = "/work/asr3/luescher/setups-data/librispeech/best-model/100h_2019-04-10/"
  allophones_file = PREFIX_PATH + "StoreAllophones.34VPSakJyy0U/output/allophones"
  alignment_file = PREFIX_PATH + "AlignmentJob.Mg44tFDRPnuh/output/alignment.cache.bundle"
  cart_file = PREFIX_PATH + "EstimateCartJob.knhvHK9ONIOC/output/cart.tree.xml.gz"

  librispeech.feature = {corpus_file: {'caches': util.MultiPath(PREFIX_PATH + "FeatureExtraction.Gammatone.tp4cEAa0YLIP/output/gt.cache.$(TASK)",
                                                                      {i:PREFIX_PATH + "FeatureExtraction.Gammatone.tp4cEAa0YLIP/output/gt.cache.{}".format(i) for i in range(librispeech.concurrent['train'])}, cached=True),
                                       'bundle': Path(PREFIX_PATH + "FeatureExtraction.Gammatone.tp4cEAa0YLIP/output/gt.cache.bundle", cached=True)
                                            },
                         'dev-clean' : {'caches': util.MultiPath("/u/michel/setups/librispeech/work/features/extraction/FeatureExtraction.Gammatone.DA0TtL8MbCKI/output/gt.cache.$(TASK)",
                                                        {i:"/u/michel/setups/librispeech/work/features/extraction/FeatureExtraction.Gammatone.DA0TtL8MbCKI/output/gt.cache.{}".format(i) 
                                                            for i in range(librispeech.concurrent['dev'])}, cached=True),
                                        'bundle': Path("/u/michel/setups/librispeech/work/features/extraction/FeatureExtraction.Gammatone.DA0TtL8MbCKI/output/gt.cache.bundle", cached=True)
                                        },
                         "test-clean" : {'caches': util.MultiPath("/u/luescher/setups/librispeech/2018-12-18--960h/work/features/extraction/FeatureExtraction.Gammatone.INa6z5A4JvZ5/output/gt.cache.$(TASK)",
                                                                      {i:"/u/luescher/setups/librispeech/2018-12-18--960h/work/features/extraction/FeatureExtraction.Gammatone.INa6z5A4JvZ5/output/gt.cache.{}".format(i) for i in range(librispeech.concurrent['test'])}, cached=True),
                                         'bundle': Path("/u/luescher/setups/librispeech/2018-12-18--960h/work/features/extraction/FeatureExtraction.Gammatone.INa6z5A4JvZ5/output/gt.cache.bundle", cached=True)
                                        },
                         "dev-other" : {'caches': util.MultiPath("/u/luescher/setups/librispeech/2018-12-18--960h/work/features/extraction/FeatureExtraction.Gammatone.qrINHi3yh3GH/output/gt.cache.$(TASK)",
                                                                      {i:"/u/luescher/setups/librispeech/2018-12-18--960h/work/features/extraction/FeatureExtraction.Gammatone.qrINHi3yh3GH/output/gt.cache.{}".format(i) for i in range(librispeech.concurrent['dev'])}, cached=True),
                                        'bundle': Path("/u/luescher/setups/librispeech/2018-12-18--960h/work/features/extraction/FeatureExtraction.Gammatone.qrINHi3yh3GH/output/gt.cache.bundle", cached=True)
                                        },
                         "test-other" : {'caches': util.MultiPath("/u/luescher/setups/librispeech/2018-12-18--960h/work/features/extraction/FeatureExtraction.Gammatone.qqN3kYqQ6QHF/output/gt.cache.$(TASK)",
                                                                      {i:"/u/luescher/setups/librispeech/2018-12-18--960h/work/features/extraction/FeatureExtraction.Gammatone.qqN3kYqQ6QHF/output/gt.cache.{}".format(i) for i in range(librispeech.concurrent['test'])}, cached=True),
                                         'bundle': Path("/u/luescher/setups/librispeech/2018-12-18--960h/work/features/extraction/FeatureExtraction.Gammatone.qqN3kYqQ6QHF/output/gt.cache.bundle", cached=True)
                                        }
                         }


  librispeech.alignment = {corpus_file: Path(alignment_file, cached=True),
                           "dev-clean" : None,
                           "test-clean" : None,
                           "dev-other": None,
                           "test-other": None
                           }

  librispeech.mixture = {c: Path(PREFIX_PATH + "EstimateMixturesJob.accumulate.dctnjFBP8hos/output/am.mix",cached=True) for c in s.subcorpus_mapping.values()}
  librispeech.cart = {c: Path(cart_file, cached=True) for c in s.subcorpus_mapping.values()}
  librispeech.allophones = {c: allophones_file for c in s.subcorpus_mapping.values()}
  librispeech.segment_whitelist = {c: None for c in s.subcorpus_mapping.values()}
  librispeech.segment_blacklist = {c: None for c in s.subcorpus_mapping.values()}
  for c, v in s.subcorpus_mapping.items():
    s.set_initial_system(corpus=c, 
            feature=librispeech.feature[v], 
            alignment=librispeech.alignment[v],
            prior_mixture=Path(PREFIX_PATH + "EstimateMixturesJob.accumulate.dctnjFBP8hos/output/am.mix",cached=True),
            cart=librispeech.cart[v],
            allophones=allophones_file)

  # set scorer
  librispeech.scorers = {c: scoring.Sclite for c in librispeech.stm_path.keys()}
  librispeech.scorer_args = {c: {'ref': librispeech.stm_path[c]} for c in librispeech.stm_path.keys()}
  s.set_scorer()
  # costa
  for c in s.subcorpus_mapping.keys():
    s.costa(c, **s.costa_args)

  s.run(steps={'nn_init'})