import json
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
from faster_whisper_transwithai_chickenrice.infer import (
    AudioChunk,
    Inference,
    Segment,
    SpeechSpan,
    create_contiguous_chunks,
    enforce_segment_timeline,
    vad_segments_to_clip_timestamps,
)


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

        class FakeModel:
            def __init__(self) -> None:
                self.calls = []

            def transcribe(self, audio, **config):
                self.calls.append((len(audio), dict(config)))
                index = len(self.calls)
                segment = mock.Mock(start=0.0, end=0.5, text=f"chunk {index}")
                return [segment], mock.Mock(duration=len(audio) / 16_000, duration_after_vad=0.5)

        model = FakeModel()
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


if __name__ == "__main__":
    unittest.main()
