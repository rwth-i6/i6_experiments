"""Parakeet TDT encoder, ONNX predictor/joint and public recognition jobs."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sisyphus import Job, Task


BLANK = 8192
EOS = 3
DURATIONS = (0, 1, 2, 3, 4)
FRAME_SHIFT = .08


def load_model(repo, device):
    from transformers import AutoProcessor, ParakeetForTDT

    from .hf_benchmark import check_transformers_version

    check_transformers_version()
    model = ParakeetForTDT.from_pretrained(repo, dtype=torch.float32).eval().to(device)
    processor = AutoProcessor.from_pretrained(repo)
    config = model.config
    fe = processor.feature_extractor
    if (config.vocab_size != BLANK + 1 or config.blank_token_id != BLANK
            or tuple(config.durations) != DURATIONS or model.generation_config.decoder_start_token_id != BLANK
            or model.generation_config.eos_token_id not in (EOS, None)
            or fe.sampling_rate != 16000 or config.encoder_config.subsampling_factor * fe.hop_length != 1280):
        raise ValueError('Parakeet checkpoint inventory, starting token or frame shift differs')
    return model, processor


class PredictorUpdate(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, h_in, c_in, token):
        embedded = self.decoder.embedding(token.long()[:, None])
        out, (h, c) = self.decoder.lstm(embedded, (h_in.transpose(0, 1).contiguous(),
                                                   c_in.transpose(0, 1).contiguous()))
        projected = self.decoder.decoder_projector(out[:, -1])
        return h.transpose(0, 1), c.transpose(0, 1), projected


class PredictorInit(torch.nn.Module):
    def __init__(self, update, layers, hidden, blank):
        super().__init__()
        self.update = update
        self.register_buffer('state', torch.zeros(1, layers, hidden))
        self.register_buffer('token', torch.tensor([blank], dtype=torch.int32))

    def forward(self):
        return self.update(self.state, self.state, self.token)


class JointCosts(torch.nn.Module):
    def __init__(self, joint, vocab_size):
        super().__init__()
        self.joint, self.vocab_size = joint, vocab_size

    def forward(self, encoder_state, pred_out):
        logits = self.joint(decoder_hidden_states=pred_out, encoder_hidden_states=encoder_state)
        token, duration = logits[:, :self.vocab_size], logits[:, self.vocab_size:]
        return torch.cat((-token.log_softmax(-1), -duration.log_softmax(-1)), dim=-1)


def export_bundle(model, files):
    """Use batch-first metadata and the int32 token input expected by RASR."""
    import onnx

    config = model.config
    update = PredictorUpdate(model.decoder)
    init = PredictorInit(update, config.num_decoder_layers, config.decoder_hidden_size, config.blank_token_id)
    joint = JointCosts(model.joint, config.vocab_size)
    states = ('h', 'c', 'pred_out')
    inputs = ('h_in', 'c_in', 'token')
    outputs = ('h_out', 'c_out', 'pred_out')
    mapping = dict(zip(inputs[:2] + outputs, states[:2] + states))
    h, c, pred = init()
    encoder = torch.zeros(1, config.decoder_hidden_size)
    exports = (
        (init, (), files['init'], (), states, dict(zip(states, states)), {}),
        (update, (h, c, init.token), files['update'], inputs, outputs, mapping,
         {name: {0: 'batch'} for name in inputs + outputs}),
        (joint, (encoder, pred), files['joint'], ('encoder_state', 'pred_out'), ('scores',),
         {'pred_out': 'pred_out'}, {'pred_out': {0: 'batch'}, 'scores': {0: 'batch'}}),
    )
    for module, args, path, input_names, output_names, metadata, axes in exports:
        torch.onnx.export(module, args, path, input_names=list(input_names), output_names=list(output_names),
                          dynamic_axes=axes, opset_version=17, dynamo=False)
        graph = onnx.load(path)
        onnx.helper.set_model_props(graph, metadata)
        onnx.checker.check_model(graph)
        onnx.save(graph, path)


@torch.inference_mode()
def verify_bundle(model, files, encoder, tokens):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sessions = {k: ort.InferenceSession(path, sess_options=opts, providers=['CPUExecutionProvider'])
                for k, path in files.items()}
    h, c, pred = sessions['init'].run(None, {})
    expected = model.decoder(torch.tensor([[BLANK]]), cache=None)[:, -1]
    np.testing.assert_allclose(pred, expected.numpy(), rtol=1e-4, atol=1e-5)
    batch = len(tokens)
    labels = np.asarray(tokens, dtype=np.int32)
    h, c, pred = sessions['update'].run(None, dict(h_in=np.repeat(h, batch, 0),
                                                   c_in=np.repeat(c, batch, 0), token=labels))
    history = torch.tensor([[BLANK, int(t)] for t in labels])
    expected = model.decoder(history, cache=None)[:, -1]
    np.testing.assert_allclose(pred, expected.numpy(), rtol=1e-4, atol=1e-5)
    h, c, pred = sessions['update'].run(None, dict(h_in=h, c_in=c, token=labels[::-1].copy()))
    history = torch.cat((history, torch.tensor(labels[::-1].copy())[:, None]), dim=1)
    expected = model.decoder(history, cache=None)[:, -1]
    np.testing.assert_allclose(pred, expected.numpy(), rtol=1e-4, atol=1e-5)
    max_difference = 0.
    for frame in encoder[:8]:
        costs, = sessions['joint'].run(None, dict(encoder_state=frame[None].numpy(), pred_out=pred))
        logits = model.joint(decoder_hidden_states=expected, encoder_hidden_states=frame[None])
        expected_costs = torch.cat((-logits[:, :BLANK+1].log_softmax(-1),
                                    -logits[:, BLANK+1:].log_softmax(-1)), dim=-1).numpy()
        np.testing.assert_allclose(costs, expected_costs, rtol=1e-4, atol=2e-5)
        max_difference = max(max_difference, float(np.max(np.abs(costs - expected_costs))))
    return max_difference


def waveform_inputs(processor, waveform, device):
    return processor([waveform], sampling_rate=16000, return_tensors='pt').to(device)


def encoder_frames(model, inputs):
    output = model.get_audio_features(**inputs)
    length = int(output.attention_mask[0].sum())
    if length < 1:
        raise ValueError('Empty TDT encoder output')
    return output.pooler_output[0, :length].float().contiguous()


class ExportParakeetTdtJob(Job):
    """Export and compare CPU ONNX and native PyTorch on real encoder states."""

    def __init__(self, repo, gate_dataset, *, version):
        self.repo, self.gate_dataset, self.version = repo, gate_dataset, version
        self.out_models = {k: self.output_path(k + '.onnx') for k in ('init', 'update', 'joint')}
        self.out_vocab = self.output_path('vocab.txt')
        self.out_gate = self.output_path('onnx-parity.json')

    def tasks(self):
        yield Task('run', rqmt=dict(cpu=2, mem=24, time=4))

    @torch.inference_mode()
    def run(self):
        from datasets import load_from_disk
        from .hf_benchmark import audio_array

        torch.set_num_threads(2)
        model, processor = load_model(self.repo.get_path(), 'cpu')
        pieces = processor.tokenizer.convert_ids_to_tokens(list(range(BLANK)))
        pieces.append('[BLANK]')
        if len(set(pieces)) != BLANK + 1 or any(not p or any(c.isspace() for c in p) or '▁' in p[1:] for p in pieces):
            raise ValueError('TDT vocab-text labels are not unique single pieces')
        Path(self.out_vocab.get_path()).write_text('\n'.join(pieces) + '\n', encoding='utf-8')
        files = {key: path.get_path() for key, path in self.out_models.items()}
        export_bundle(model, files)
        sample = load_from_disk(self.gate_dataset.get_path())[0]
        inputs = waveform_inputs(processor, audio_array(sample['audio']), 'cpu')
        encoder = encoder_frames(model, inputs)
        tokens = processor.tokenizer.encode("This is a test.", add_special_tokens=False)[:3]
        difference = verify_bundle(model, files, encoder, tokens)
        Path(self.out_gate.get_path()).write_text(json.dumps(dict(passed=True, id=str(sample['id']),
            frames=len(encoder), max_cost_difference=difference, blank=BLANK, durations=DURATIONS,
            frame_shift_seconds=FRAME_SHIFT), indent=2) + '\n')


def decode_tokens(model, processor, ids):
    eos = model.generation_config.eos_token_id
    return processor.decode([i for i in ids if i not in (BLANK, eos)], skip_special_tokens=True,
                            group_tokens=False).strip()


def trace_result(trace, vocabulary, model, processor, frames):
    tokens = [vocabulary[item.lemma] for item in trace if item.lemma]
    times = [(int(item.start_time), int(item.end_time)) for item in trace]
    ends = [end for _, end in times]
    final_time = ends[-1] if ends else None
    if (any(start < 0 or end < start for start, end in times)
            or ends != sorted(ends) or (final_time is not None and final_time > frames + max(DURATIONS))
            or not all(np.isfinite(float(item.am_score) + float(item.lm_score)) for item in trace)):
        raise ValueError(f'Invalid TDT traceback times or scores on {frames} encoder frames')
    return dict(text=decode_tokens(model, processor, tokens), token_ids=tokens,
                encoder_frames=frames, traceback_end=final_time, traceback_times=times,
                reached_encoder_end=final_time is not None and final_time >= frames)


class ParakeetTdtRecogJob(Job):
    def __init__(self, repo, dataset, *, version, config=None, binary=None, vocabulary=None,
                 onnx_gate=None, search_gate=None, auxiliary=None, kind='native', probe_size=None):
        self.repo, self.dataset, self.version = repo, dataset, version
        self.config, self.binary, self.vocabulary = config, binary, vocabulary
        self.onnx_gate, self.search_gate = onnx_gate, search_gate
        self.auxiliary, self.kind, self.probe_size = auxiliary, kind, probe_size
        self.out_hypotheses = self.output_path('hypotheses.jsonl')
        self.out_runtime = self.output_path('runtime.json')
        self.out_probe = self.output_path('probe.json')
        self.out_log = self.output_path('rasr.log')

    def tasks(self):
        yield Task('run', rqmt=dict(cpu=2, mem=32, gpu=1, gpu_mem=48, time=24,
                                   sbatch_args=['-p', 'gpu_48gb']))

    @torch.inference_mode()
    def run(self):
        from datasets import load_from_disk
        from .hf_benchmark import audio_array

        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        for gate in (self.onnx_gate, self.search_gate):
            if gate is not None and not json.loads(Path(gate.get_path()).read_text())['passed']:
                raise ValueError('TDT validation did not pass')
        model, processor = load_model(self.repo.get_path(), 'cuda')
        search, auxiliary = None, None
        if self.config is not None:
            sys.path.insert(0, self.binary.get_path())
            import librasr

            if self.auxiliary is not None:
                from .rnnt_llm import register_rnnt_llm_scorer
                if self.kind == 'speech':
                    from .qwen_speech import QwenSpeechScorer
                    auxiliary = QwenSpeechScorer(self.auxiliary.get_path(), batch_size=8)
                else:
                    from .dynamic_qwen_tuned import TunedQwenScorer
                    auxiliary = TunedQwenScorer(self.auxiliary.get_path(), scale=1., reward=0.,
                                                formatting='original', terminal_period=False)
                register_rnnt_llm_scorer(librasr, 'parakeet-tdt-auxiliary', lambda _: auxiliary,
                                         case_variants=('original',))
            cfg = librasr.Configuration()
            if not cfg.set_from_file(self.config.get_path()):
                raise ValueError('Cannot load TDT search configuration')
            for channel in ('log', 'warning', 'error', 'statistics'):
                cfg.set(f'*.{channel}.channel', self.out_log.get_path())
            librasr.init_logging(cfg)
            search = librasr.SearchAlgorithm(config=cfg)
            vocabulary = {piece: i for i, piece in enumerate(
                Path(self.vocabulary.get_path()).read_text(encoding='utf-8').split('\n')[:-1])}
        dataset = load_from_disk(self.dataset.get_path())
        if self.probe_size is not None:
            dataset = dataset.select(range(min(self.probe_size, len(dataset))))
        elapsed = audio_seconds = 0.
        probe = []
        with open(self.out_hypotheses.get_path(), 'w') as dst:
            for index, sample in enumerate(dataset):
                waveform = audio_array(sample['audio'])
                torch.cuda.synchronize()
                start = time.perf_counter()
                inputs = waveform_inputs(processor, waveform, 'cuda')
                if search is None:
                    generated = model.generate(**inputs)
                    ids = (generated.sequences if hasattr(generated, 'sequences') else generated)[0].tolist()
                    result = dict(text=decode_tokens(model, processor, ids), token_ids=ids)
                else:
                    if self.kind == 'speech':
                        auxiliary.set_audio(auxiliary.prepare(waveform))
                        if index == 0:
                            auxiliary.validate()
                    features = encoder_frames(model, inputs).cpu().numpy()
                    result = trace_result(search.recognize_segment(features), vocabulary, model, processor, len(features))
                torch.cuda.synchronize()
                elapsed += time.perf_counter() - start
                audio_seconds += len(waveform) / 16000
                if self.probe_size is not None:
                    raw = model.generate(**inputs, return_dict_in_generate=True)
                    expected = decode_tokens(model, processor, raw.sequences[0].tolist())
                    probe.append(dict(id=str(sample['id']), native_text=expected, rasr_text=result['text'],
                                      exact=expected == result['text'], encoder_frames=result['encoder_frames'],
                                      traceback_end=result['traceback_end']))
                dst.write(json.dumps(dict(id=str(sample['id']), **result), ensure_ascii=False) + '\n')
        if not audio_seconds:
            raise ValueError('Empty TDT benchmark')
        if self.probe_size is not None and not any(r['rasr_text'] and r['native_text'] for r in probe):
            raise ValueError('No nonempty TDT probe transcript')
        Path(self.out_probe.get_path()).write_text(json.dumps(dict(passed=True, segments=probe,
            exact_matches=sum(r['exact'] for r in probe)), indent=2) + '\n')
        Path(self.out_runtime.get_path()).write_text(json.dumps(dict(elapsed_seconds=elapsed, audio_seconds=audio_seconds,
            rtf=elapsed/audio_seconds, num_segments=len(dataset)), indent=2) + '\n')
