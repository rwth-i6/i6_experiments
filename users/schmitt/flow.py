from i6_core.features.common import get_input_node_type
import i6_core.rasr as rasr


def get_raw_wav_feature_flow(
  audio_format="wav",
  dc_detection=True,
  dc_params={
    "min-dc-length": 0.01,
    "max-dc-increment": 0.9,
    "min-non-dc-segment-length": 0.021,
  },
  input_options=None,
  scale_input=None,
):
  """
    Copied from i6_core.features.common.samples_flow and only replaced "network:samples" by "network:features".
  """
  net = rasr.FlowNetwork()

  net.add_output("features")
  net.add_param(["input-file", "start-time", "end-time", "track"])

  input_opts = {
    "file": "$(input-file)",
    "start-time": "$(start-time)",
    "end-time": "$(end-time)",
  }

  if input_options is not None:
    input_opts.update(**input_options)

  input_node_type = get_input_node_type(audio_format)

  samples = net.add_node("audio-input-file-" + input_node_type, "samples", input_opts)
  if input_node_type == "ffmpeg":
    samples_out = samples
  else:
    demultiplex = net.add_node("generic-vector-s16-demultiplex", "demultiplex", track="$(track)")
    net.link(samples, demultiplex)

    convert = net.add_node("generic-convert-vector-s16-to-vector-f32", "convert")
    net.link(demultiplex, convert)
    samples_out = convert

  if scale_input:
    scale = net.add_node("generic-vector-f32-multiplication", "scale", value=str(scale_input))
    net.link(samples_out, scale)
    pre_dc_out = scale
  else:
    pre_dc_out = samples_out

  if dc_detection:
    dc_detection = net.add_node("signal-dc-detection", "dc-detection", dc_params)
    net.link(pre_dc_out, dc_detection)
    net.link(dc_detection, "network:features")
  else:
    net.link(pre_dc_out, "network:features")

  return net


def get_raw_wav_feature_flow_w_alignment(
  audio_format="wav",
  dc_detection=True,
  dc_params={
    "min-dc-length": 0.01,
    "max-dc-increment": 0.9,
    "min-non-dc-segment-length": 0.021,
  },
  input_options=None,
  scale_input=None,
):
  """
    Copied from i6_core.features.common.samples_flow and only replaced "network:samples" by "network:features".
  """
  net = get_raw_wav_feature_flow(
    audio_format=audio_format,
    dc_detection=dc_detection,
    dc_params=dc_params,
    input_options=input_options,
    scale_input=scale_input,
  )
  net.add_output("alignments")
  net.add_param(["id", "orthography"])
  net.add_node("generic-aggregation-vector-f32", "aggregate")
  net.link("scale", "aggregate")
  net.add_node("speech-seq2seq-alignment", "alignment", id="$(id)", orthography="$(orthography)")
  net.link("aggregate", "alignment")
  net.add_node("generic-cache", "alignment-cache", id="$(id)")
  net.link("alignment", "alignment-cache")
  net.link("alignment-cache", "network:alignments")

  return net


