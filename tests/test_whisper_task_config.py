import io
import json
import logging
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

sys.modules.setdefault("pyjson5", types.SimpleNamespace(decode_io=json.load))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import faster_whisper_transwithai_chickenrice.infer as infer_module
from faster_whisper_transwithai_chickenrice.backends.base import (
    BackendRequest,
    BackendResult,
    BackendSegment,
    ModelDescriptor,
)
from faster_whisper_transwithai_chickenrice.backends.ct2 import CTranslate2Backend
from faster_whisper_transwithai_chickenrice.infer import (
    AudioChunk,
    Inference,
    Segment,
    SpeechSpan,
    create_contiguous_chunks,
    enforce_segment_timeline,
    require_local_runtime_assets,
    select_best_compute_type,
    vad_segments_to_clip_timestamps,
)
from faster_whisper_transwithai_chickenrice.vad_manager import VadConfig, WhisperVadModel, WhisperVADOnnxWrapper


def make_args(config_path: Path, task: str | None = None) -> Namespace:
    return Namespace(
        generation_config=str(config_path),
        task=task,
        vad_threshold=None,
        vad_min_speech_duration_ms=None,
        vad_min_silence_duration_ms=None,
        vad_speech_pad_ms=None,
        merge_segments=None,
        merge_max_gap_ms=None,
        merge_max_duration_ms=None,
        smart_split_with_vad=None,
        target_chunk_duration_s=None,
    )


def make_full_args(config_path: Path, model_path: Path, *, compute_type: str = "auto") -> Namespace:
    args = make_args(config_path)
    args.model_name_or_path = str(model_path)
    args.device = "cpu"
    args.compute_type = compute_type
    args.cpu_threads = 12
    args.vad_threads = 4
    args.enable_batching = False
    args.batch_size = None
    args.max_batch_size = 8
    args.overwrite = False
    args.output_dir = None
    args.audio_suffixes = "wav,mp3"
    args.sub_formats = "srt,vtt,lrc"
    return args


class WhisperTaskConfigTests(unittest.TestCase):
    def load_config(self, config_text: str, task: str | None = None):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "generation_config.json5"
            config_path.write_text(config_text, encoding="utf-8")

            inference = Inference.__new__(Inference)
            config, _segment_merge_options, _smart_split_options = inference._load_generation_config(
                make_args(config_path, task=task)
            )
            return config

    def test_uses_task_from_generation_config(self) -> None:
        config = self.load_config('{"task": "transcribe"}')

        self.assertEqual(config["task"], "transcribe")

    def test_cli_task_overrides_generation_config(self) -> None:
        config = self.load_config('{"task": "transcribe"}', task="translate")

        self.assertEqual(config["task"], "translate")

    def test_cli_task_takes_precedence_over_invalid_config_task(self) -> None:
        config = self.load_config('{"task": "caption"}', task="transcribe")

        self.assertEqual(config["task"], "transcribe")

    def test_invalid_config_task_fails_without_cli_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid Whisper task"):
            self.load_config('{"task": "caption"}')

    def test_defaults_to_translate_without_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "missing.json5"
            inference = Inference.__new__(Inference)
            config, _segment_merge_options, _smart_split_options = inference._load_generation_config(
                make_args(config_path)
            )

        self.assertEqual(config["task"], "translate")


class CpuRuntimePolicyTests(unittest.TestCase):
    def test_cpu_auto_prefers_int8(self) -> None:
        fake_ct2 = types.SimpleNamespace(
            get_supported_compute_types=lambda _device: {"float32", "int8", "int8_float32"},
            get_cuda_device_count=lambda: 0,
        )
        with mock.patch.object(infer_module, "ctranslate2", fake_ct2):
            self.assertEqual(select_best_compute_type("cpu"), "int8")

    def test_explicit_compute_type_and_threads_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            args = make_full_args(root / "config.json5", root / "model", compute_type="float32")
            with (
                mock.patch.object(infer_module, "require_local_runtime_assets"),
                mock.patch.object(Inference, "_setup_vad_injection"),
            ):
                inference = Inference(args)

        self.assertEqual(inference.compute_type, "float32")
        self.assertEqual(inference.cpu_threads, 12)
        self.assertEqual(inference.vad_threads, 4)

    def test_cpu_threads_are_passed_to_whisper_model(self) -> None:
        model = mock.Mock()
        model.transcribe.return_value = (
            [mock.Mock(start=0.0, end=1.2, text="translated")],
            mock.Mock(duration=1.0, duration_after_vad=1.0, language="ja"),
        )
        model_class = mock.Mock(return_value=model)
        backend = CTranslate2Backend(
            ModelDescriptor(
                backend="ct2",
                profile="translate",
                variant="int8",
                path=Path("/models/translate"),
            ),
            device="cpu",
            compute_type="int8",
            cpu_threads=12,
            model_class=model_class,
            batched_pipeline_class=mock.Mock(),
        )
        result = backend.transcribe(BackendRequest(audio="audio.mp3", language="ja", task="translate"))

        model_class.assert_called_once_with("/models/translate", device="cpu", compute_type="int8", cpu_threads=12)
        self.assertEqual(result.segments[0].end, 1.0)

    def test_missing_main_model_fails_before_writing_subtitles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            subtitle = root / "output.srt"
            with self.assertRaisesRegex(RuntimeError, "inference was not started"):
                require_local_runtime_assets(root / "missing-model")
            self.assertFalse(subtitle.exists())

    def test_all_existing_outputs_skip_without_loading_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audio = root / "sample.wav"
            audio.write_bytes(b"audio")
            (root / "sample.srt").write_text("existing", encoding="utf-8")
            args = make_full_args(root / "config.json5", root / "missing-model")
            args.sub_formats = "srt"
            inference = Inference(args)
            with mock.patch.object(inference, "_ensure_runtime_ready") as ensure:
                status = inference.generates([str(audio)])

        self.assertEqual(status, infer_module.EXIT_OK)
        ensure.assert_not_called()


class VadRuntimePolicyTests(unittest.TestCase):
    def test_cpu_vad_excludes_coreml_and_cuda(self) -> None:
        class FakeSessionOptions:
            inter_op_num_threads = 0
            intra_op_num_threads = 0

        class FakeSession:
            def __init__(self, _path, *, providers, sess_options):
                self.providers = providers
                self.sess_options = sess_options

            def get_inputs(self):
                return [types.SimpleNamespace(name="input")]

            def get_outputs(self):
                return [types.SimpleNamespace(name="output")]

            def get_providers(self):
                return self.providers

        fake_ort = types.SimpleNamespace(
            SessionOptions=FakeSessionOptions,
            InferenceSession=FakeSession,
            get_available_providers=lambda: [
                "CoreMLExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        extractor = mock.Mock()
        fake_transformers = types.SimpleNamespace(
            WhisperFeatureExtractor=types.SimpleNamespace(from_pretrained=extractor)
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            model_path = root / "whisper_vad.onnx"
            metadata_path = root / "whisper_vad_metadata.json"
            model_path.write_bytes(b"onnx")
            metadata_path.write_text("{}", encoding="utf-8")
            (root / "whisper-base").mkdir()
            (root / "whisper-base" / "preprocessor_config.json").write_text("{}", encoding="utf-8")
            with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort, "transformers": fake_transformers}):
                wrapper = WhisperVADOnnxWrapper(str(model_path), str(metadata_path), force_cpu=True, num_threads=4)

        self.assertEqual(wrapper.session.get_providers(), ["CPUExecutionProvider"])
        self.assertEqual(wrapper.session.sess_options.inter_op_num_threads, 1)
        self.assertEqual(wrapper.session.sess_options.intra_op_num_threads, 4)
        extractor.assert_called_once_with(str((root / "whisper-base").resolve()), local_files_only=True)

    def test_missing_vad_file_raises_during_initialization(self) -> None:
        config = VadConfig(onnx_model_path="/definitely/missing/vad.onnx")
        with self.assertRaisesRegex(FileNotFoundError, "macos_doctor"):
            WhisperVadModel(config)


class MainCompatibilityTests(unittest.TestCase):
    def test_main_accepts_streams_without_reconfigure(self) -> None:
        args = Namespace(console=False, log_level="INFO", base_dirs=[])
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch.object(infer_module, "parse_arguments", return_value=args),
            mock.patch.object(sys, "stdout", io.StringIO()),
            mock.patch.object(sys, "stderr", io.StringIO()),
            mock.patch("os.getcwd", return_value=tmp_dir),
            mock.patch.object(infer_module.logging, "FileHandler", return_value=logging.NullHandler()),
        ):
            self.assertEqual(infer_module.main(), infer_module.EXIT_NO_INPUT)


class VadClipTimestampTests(unittest.TestCase):
    def test_converts_sample_indices_to_clip_timestamp_pairs(self) -> None:
        clips = vad_segments_to_clip_timestamps([{"start": 16_000, "end": 32_000}])

        self.assertEqual(clips, [1.0, 2.0])

    def test_converts_sample_indices_to_batched_clip_dicts(self) -> None:
        clips = vad_segments_to_clip_timestamps([{"start": 16_000, "end": 32_000}], batched=True)

        self.assertEqual(clips, [{"start": 1.0, "end": 2.0}])

    def test_prepare_transcription_uses_vad_clips_when_smart_split_disabled(self) -> None:
        inference = Inference.__new__(Inference)
        inference.generation_config = {
            "language": "ja",
            "task": "translate",
            "vad_filter": True,
            "vad_parameters": {"threshold": 0.5},
        }
        inference.smart_split_options = infer_module.SmartSplitOptions(enabled=False)
        inference.vad_manager = mock.Mock()
        inference.vad_manager.get_speech_timestamps.return_value = [{"start": 16_000, "end": 32_000}]

        with mock.patch.object(infer_module, "decode_audio", return_value=[0] * 32_000):
            audio, config, duration_after_vad = inference._prepare_transcription("audio.mp3", batched=False)

        self.assertEqual(len(audio), 32_000)
        self.assertEqual(duration_after_vad, 1.0)
        self.assertFalse(config["vad_filter"])
        self.assertEqual(config["clip_timestamps"], [1.0, 2.0])
        self.assertEqual(config["beam_size"], 1)
        self.assertFalse(config["condition_on_previous_text"])


class SmartSplitTests(unittest.TestCase):
    def test_create_contiguous_chunks_prefers_silence_near_target_end(self) -> None:
        chunks = create_contiguous_chunks(
            [
                SpeechSpan(0.0, 8.0),
                SpeechSpan(12.0, 20.0),
                SpeechSpan(26.0, 35.0),
            ],
            max_duration=30.0,
            total_duration=40.0,
        )

        self.assertEqual(chunks[0], AudioChunk(0, 0.0, 23.0))
        self.assertEqual(chunks[1], AudioChunk(1, 23.0, 40.0))

    def test_enforce_segment_timeline_caps_and_removes_overlaps(self) -> None:
        segments = [
            Segment(start=1_000, end=30_000, text="first"),
            Segment(start=2_000, end=4_000, text="overlap"),
            Segment(start=5_000, end=5_000, text="empty"),
        ]

        normalized = enforce_segment_timeline(segments, max_duration_ms=20_000)

        self.assertEqual(normalized, [Segment(start=1_000, end=21_000, text="first")])

    def test_smart_chunk_transcription_runs_internal_vad_per_chunk(self) -> None:
        inference = Inference.__new__(Inference)
        inference.generation_config = {
            "language": "ja",
            "task": "translate",
            "vad_filter": True,
            "vad_parameters": {"threshold": 0.5},
        }
        inference.smart_split_options = infer_module.SmartSplitOptions(enabled=True, target_chunk_duration_s=2.0)
        inference.vad_manager = mock.Mock()
        inference.vad_manager.get_speech_timestamps.return_value = [
            {"start": 0, "end": 16_000},
            {"start": 48_000, "end": 64_000},
        ]

        class FakeBackend:
            def __init__(self) -> None:
                self.calls = []

            def transcribe(self, request: BackendRequest) -> BackendResult:
                self.calls.append((len(request.audio), dict(request.options)))
                index = len(self.calls)
                return BackendResult(
                    segments=[BackendSegment(start=0.0, end=0.5, text=f"chunk {index}")],
                    duration=len(request.audio) / 16_000,
                    duration_after_vad=0.5,
                    language="ja",
                    backend="fake",
                )

        model = FakeBackend()
        task = mock.Mock(audio_path="audio.mp3")

        with mock.patch.object(infer_module, "decode_audio", return_value=[0.0] * 80_000):
            segments, info = inference._transcribe_smart_chunks(model, task)

        self.assertEqual(len(model.calls), 3)
        self.assertTrue(all(call_config["vad_filter"] for _length, call_config in model.calls))
        self.assertNotIn("clip_timestamps", model.calls[0][1])
        self.assertEqual(segments[0], Segment(start=0, end=500, text="chunk 1"))
        self.assertEqual(segments[1], Segment(start=2_000, end=2_500, text="chunk 2"))
        self.assertEqual(segments[2], Segment(start=4_000, end=4_500, text="chunk 3"))
        self.assertEqual(info.duration, 5.0)
        self.assertEqual(info.duration_after_vad, 1.5)

    def test_smart_chunk_silence_logs_and_skips_model(self) -> None:
        inference = Inference.__new__(Inference)
        inference.generation_config = {
            "language": "ja",
            "task": "translate",
            "vad_filter": True,
            "vad_parameters": {"threshold": 0.5},
        }
        inference.smart_split_options = infer_module.SmartSplitOptions(enabled=True, target_chunk_duration_s=30.0)
        inference.vad_manager = mock.Mock()
        inference.vad_manager.get_speech_timestamps.return_value = []
        model = mock.Mock()
        task = mock.Mock(audio_path="silence.wav")

        with (
            mock.patch.object(infer_module, "decode_audio", return_value=[0.0] * 32_000),
            self.assertLogs(infer_module.logger, level="INFO") as logs,
        ):
            segments, info = inference._transcribe_smart_chunks(model, task)

        self.assertEqual(segments, [])
        self.assertEqual(info.duration, 2.0)
        self.assertEqual(info.duration_after_vad, 0)
        model.transcribe.assert_not_called()
        self.assertTrue(any("No speech detected" in message for message in logs.output))

    def test_mlx_smart_chunks_use_outer_vad_clips_and_skip_silent_chunks(self) -> None:
        inference = Inference.__new__(Inference)
        inference.backend_name = "mlx"
        inference.generation_config = {
            "language": "ja",
            "task": "translate",
            "vad_filter": True,
        }
        inference.smart_split_options = infer_module.SmartSplitOptions(enabled=True, target_chunk_duration_s=2.0)
        inference.vad_manager = mock.Mock()
        inference.vad_manager.get_speech_timestamps.return_value = [
            {"start": 0, "end": 16_000},
            {"start": 48_000, "end": 64_000},
        ]

        class FakeBackend:
            def __init__(self) -> None:
                self.calls: list[BackendRequest] = []

            def transcribe(self, request: BackendRequest) -> BackendResult:
                self.calls.append(request)
                return BackendResult(
                    segments=[BackendSegment(start=0.0, end=0.5, text="chunk")],
                    duration=len(request.audio) / 16_000,
                    duration_after_vad=None,
                    language="ja",
                    backend="mlx",
                )

        backend = FakeBackend()
        task = mock.Mock(audio_path="audio.mp3")

        with mock.patch.object(infer_module, "decode_audio", return_value=[0.0] * 80_000):
            segments, result = inference._transcribe_smart_chunks(backend, task)

        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(backend.calls[0].options["clip_timestamps"], [0.0, 1.0])
        self.assertEqual(backend.calls[1].options["clip_timestamps"], [1.0, 2.0])
        self.assertEqual(result.metrics["chunk_count"], 2.0)
        self.assertEqual([segment.start for segment in segments], [0, 2_000])


if __name__ == "__main__":
    unittest.main()
