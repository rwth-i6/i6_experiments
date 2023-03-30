import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
from resemblyzer import VoiceEncoder, preprocess_wav
from i6_core.lib import corpus
from i6_private.users.rossenbach.lib.hdf import SimpleHDFWriter
import numpy as np

embeddings = {}
encoder = VoiceEncoder()
sil_prep_corpus = corpus.Corpus()
sil_prep_corpus.load("/work/asr3/rossenbach/schuemann/sis_work/i6_core/audio/ffmpeg/BlissFfmpegJob.yi48w07CgSW6/output/corpus.xml.gz")
for recording in sil_prep_corpus.all_recordings():
  wav = preprocess_wav(recording.audio)
  embeddings[recording.fullname()] = encoder.embed_utterance(wav)
  if len(embeddings) % 200 == 0:
    print(len(embeddings))

print(len(embeddings))

hdf_writer = SimpleHDFWriter(
  "test.hdf", dim=256, ndim=2
)

for name, embedding in embeddings.items():
  hdf_writer.insert_batch(
    np.asarray([[embedding]], dtype="float32"), [1], [name]
  )
hdf_writer.close()