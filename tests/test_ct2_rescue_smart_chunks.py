import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import faster_whisper_transwithai_chickenrice.infer as infer_module
from faster_whisper_transwithai_chickenrice.backends.base import (
    BackendCapabilities,
    BackendRequest,
    BackendResult,
    BackendSegment,
    ModelDescriptor,
)
from faster_whisper_transwithai_chickenrice.backends.factory import BackendSelection
from faster_whisper_transwithai_chickenrice.infer import (
    AudioChunk,
    Inference,
    InferenceTask,
    SegmentMergeOptions,
    SpeechSpan,
    _LazyCT2RescueBackend,
    _validate_ct2_rescue_candidate,
)
from faster_whisper_transwithai_chickenrice.mlx_diagnostics import CT2RescueDecision


class FakeBackend:
    capabilities = BackendCapabilities(
        backend="fake",
        supports_translate=True,
        supports_transcribe=True,
        supports_word_timestamps=False,
        supports_batching=False,
    )
    descriptor = ModelDescriptor(
        backend="fake",
        profile="translate",
        variant="test",
        path=Path("/fake/model"),
    )
    batching_enabled = False
    ignored_options: tuple[str, ...] = ()

    def __init__(self, responses: list[BackendResult | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[BackendRequest] = []
        self.closed = False

    def transcribe(self, request: BackendRequest) -> BackendResult:
        self.calls.append(request)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        self.closed = True


def backend_result(
    backend: str,
    text: str,
    *,
    start: float = 0.1,
    end: float = 0.9,
    duration: float = 2.0,
    duration_after_vad: float | None = None,
    metrics: dict[str, float] | None = None,
) -> BackendResult:
    segments = [BackendSegment(start=start, end=end, text=text)] if text else []
    return BackendResult(
        segments=segments,
        duration=duration,
        duration_after_vad=duration_after_vad,
        language="ja",
        backend=backend,
        metrics=dict(metrics or {}),
        diagnostics={},
    )


class CT2RescueSmartChunkTests(unittest.TestCase):
    @staticmethod
    def _inference(*, backend_name: str = "mlx", chunk_count: int = 2) -> Inference:
        inference = Inference.__new__(Inference)
        inference.backend_name = backend_name
        inference.profile = None
        inference._warned_backend_options = set()
        inference.generation_config = {
            "language": "ja",
            "task": "translate",
            "vad_filter": True,
            "vad_parameters": {"threshold": 0.5},
            "beam_size": 1,
            "condition_on_previous_text": False,
            "mlx_use_outer_vad_clips": False,
            "mlx_safe_retry_without_clips": True,
            "mlx_debug_diagnostics": False,
            "mlx_sampling_seed": 0,
            "fp16": True,
            "sample_len": 128,
            "verbose": True,
        }
        duration = chunk_count * 2.0
        audio = [0.0] * int(duration * infer_module.WHISPER_SAMPLING_RATE)
        chunks = [AudioChunk(index, index * 2.0, (index + 1) * 2.0) for index in range(chunk_count)]
        speech_spans = [SpeechSpan(index * 2.0, index * 2.0 + 1.0) for index in range(chunk_count)]
        inference.__dict__["_plan_smart_chunks"] = mock.Mock(
            return_value=(audio, chunks, float(chunk_count), speech_spans)
        )
        return inference

    @staticmethod
    def _task() -> InferenceTask:
        return InferenceTask(audio_path="audio.wav", sub_prefix="unused", sub_formats=[])

    def test_normal_mlx_chunks_never_create_ct2_backend(self) -> None:
        inference = self._inference()
        mlx = FakeBackend([backend_result("mlx", "mlx one"), backend_result("mlx", "mlx two")])
        ct2 = FakeBackend([])
        factory = mock.Mock(return_value=ct2)
        holder = _LazyCT2RescueBackend(factory)

        with mock.patch.object(
            infer_module,
            "decide_ct2_rescue",
            return_value=CT2RescueDecision(False, "no_objective_high_risk_anomaly"),
        ) as decide:
            segments, result = inference._transcribe_smart_chunks(
                mlx,
                self._task(),
                ct2_rescue_holder=holder,
            )

        self.addCleanup(holder.close)
        self.assertEqual([segment.text for segment in segments], ["mlx one", "mlx two"])
        self.assertEqual(decide.call_count, 2)
        for call in decide.call_args_list:
            self.assertIs(call.args[1], call.args[0].diagnostics)
            self.assertTrue(call.kwargs["outer_vad_has_speech"])
            self.assertFalse(call.kwargs["ct2_rescue_already_attempted"])
        factory.assert_not_called()
        self.assertEqual(ct2.calls, [])
        self.assertNotIn("ct2_rescue_call_count", result.metrics)

    def test_triggered_chunks_reuse_one_ct2_backend_and_replace_whole_chunks(self) -> None:
        inference = self._inference()
        mlx = FakeBackend(
            [
                backend_result("mlx", "mlx one", metrics={"runtime_call_count": 2.0}),
                backend_result("mlx", "mlx two", metrics={"runtime_call_count": 2.0}),
            ]
        )
        ct2 = FakeBackend(
            [
                backend_result(
                    "ct2",
                    "ct2 one",
                    start=0.25,
                    end=0.75,
                    duration_after_vad=2.0,
                    metrics={"inference_seconds": 0.1},
                ),
                backend_result(
                    "ct2",
                    "ct2 two",
                    start=0.1,
                    end=0.6,
                    duration_after_vad=2.0,
                    metrics={"inference_seconds": 0.2},
                ),
            ]
        )
        factory = mock.Mock(return_value=ct2)
        holder = _LazyCT2RescueBackend(factory)

        with mock.patch.object(
            infer_module,
            "decide_ct2_rescue",
            return_value=CT2RescueDecision(True, "output_truncated"),
        ):
            segments, result = inference._transcribe_smart_chunks(
                mlx,
                self._task(),
                ct2_rescue_holder=holder,
            )

        self.addCleanup(holder.close)
        factory.assert_called_once_with()
        self.assertEqual(len(ct2.calls), 2)
        self.assertTrue(all(mlx.calls[index].audio is ct2.calls[index].audio for index in range(2)))
        self.assertEqual([segment.text for segment in segments], ["ct2 one", "ct2 two"])
        self.assertEqual([segment.start for segment in segments], [250, 2_100])
        self.assertEqual(result.metrics["runtime_call_count"], 4.0)
        self.assertEqual(result.metrics["ct2_rescue_trigger_count"], 2.0)
        self.assertEqual(result.metrics["ct2_rescue_call_count"], 2.0)
        self.assertEqual(result.metrics["ct2_rescue_accept_count"], 2.0)
        self.assertAlmostEqual(result.metrics["ct2_rescue_inference_seconds"], 0.3)
        self.assertEqual(result.duration_after_vad, 2.0)
        for request in ct2.calls:
            self.assertFalse(request.options["vad_filter"])
            self.assertNotIn("vad_parameters", request.options)
            self.assertNotIn("clip_timestamps", request.options)
            self.assertNotIn("fp16", request.options)
            self.assertNotIn("sample_len", request.options)
            self.assertNotIn("verbose", request.options)
            self.assertFalse(any(name.startswith("mlx_") for name in request.options))
        rescue_records = [chunk["ct2_rescue"] for chunk in result.diagnostics["chunks"]]
        self.assertTrue(all(record["accepted"] for record in rescue_records))
        self.assertTrue(all(record["reason"] == "ct2_rescue_accepted" for record in rescue_records))

    def test_ct2_inference_failure_keeps_mlx_and_continues_with_next_chunk(self) -> None:
        inference = self._inference()
        mlx = FakeBackend([backend_result("mlx", "keep mlx"), backend_result("mlx", "second mlx")])
        ct2 = FakeBackend(
            [
                RuntimeError("synthetic CT2 failure"),
                backend_result("ct2", "second ct2"),
            ]
        )
        holder = _LazyCT2RescueBackend(mock.Mock(return_value=ct2))

        with mock.patch.object(
            infer_module,
            "decide_ct2_rescue",
            return_value=CT2RescueDecision(True, "output_truncated"),
        ):
            segments, result = inference._transcribe_smart_chunks(
                mlx,
                self._task(),
                ct2_rescue_holder=holder,
            )

        self.addCleanup(holder.close)
        self.assertEqual([segment.text for segment in segments], ["keep mlx", "second ct2"])
        self.assertEqual(len(mlx.calls), 2)
        self.assertEqual(len(ct2.calls), 2)
        self.assertEqual(result.metrics["ct2_rescue_call_count"], 2.0)
        self.assertEqual(result.metrics["ct2_rescue_reject_count"], 1.0)
        self.assertEqual(result.metrics["ct2_rescue_accept_count"], 1.0)
        self.assertEqual(result.diagnostics["chunks"][0]["ct2_rescue"]["reason"], "ct2_inference_failed")
        self.assertEqual(result.diagnostics["chunks"][1]["ct2_rescue"]["reason"], "ct2_rescue_accepted")

    def test_invalid_ct2_candidate_keeps_mlx_result(self) -> None:
        inference = self._inference(chunk_count=1)
        mlx = FakeBackend([backend_result("mlx", "keep mlx")])
        ct2 = FakeBackend([backend_result("ct2", "out of range", start=0.1, end=2.5)])
        holder = _LazyCT2RescueBackend(mock.Mock(return_value=ct2))

        with mock.patch.object(
            infer_module,
            "decide_ct2_rescue",
            return_value=CT2RescueDecision(True, "output_truncated"),
        ):
            segments, result = inference._transcribe_smart_chunks(
                mlx,
                self._task(),
                ct2_rescue_holder=holder,
            )

        self.addCleanup(holder.close)
        self.assertEqual([segment.text for segment in segments], ["keep mlx"])
        self.assertEqual(len(ct2.calls), 1)
        rescue = result.diagnostics["chunks"][0]["ct2_rescue"]
        self.assertFalse(rescue["accepted"])
        self.assertEqual(rescue["reason"], "ct2_invalid_timestamp")

    def test_tail_without_speech_support_keeps_mlx_result(self) -> None:
        inference = self._inference(chunk_count=1)
        mlx = FakeBackend([backend_result("mlx", "keep mlx", metrics={"runtime_call_count": 2.0})])
        ct2 = FakeBackend([backend_result("ct2", "unsupported tail", start=1.0, end=2.0)])
        holder = _LazyCT2RescueBackend(mock.Mock(return_value=ct2))

        with mock.patch.object(
            infer_module,
            "decide_ct2_rescue",
            return_value=CT2RescueDecision(True, "output_truncated"),
        ):
            segments, result = inference._transcribe_smart_chunks(
                mlx,
                self._task(),
                ct2_rescue_holder=holder,
            )

        self.addCleanup(holder.close)
        self.assertEqual([segment.text for segment in segments], ["keep mlx"])
        self.assertEqual(len(mlx.calls), 1)
        self.assertEqual(len(ct2.calls), 1)
        self.assertEqual(result.metrics["runtime_call_count"], 2.0)
        self.assertEqual(result.metrics["ct2_rescue_call_count"], 1.0)
        self.assertEqual(result.metrics["ct2_rescue_reject_count"], 1.0)
        self.assertNotIn("ct2_rescue_accept_count", result.metrics)
        rescue = result.diagnostics["chunks"][0]["ct2_rescue"]
        self.assertFalse(rescue["accepted"])
        self.assertEqual(rescue["reason"], "ct2_tail_without_speech_support")

    def test_unavailable_ct2_factory_is_not_retried_for_later_chunks(self) -> None:
        inference = self._inference()
        mlx = FakeBackend([backend_result("mlx", "one"), backend_result("mlx", "two")])
        factory = mock.Mock(side_effect=RuntimeError("CT2 unavailable"))
        holder = _LazyCT2RescueBackend(factory)

        with mock.patch.object(
            infer_module,
            "decide_ct2_rescue",
            return_value=CT2RescueDecision(True, "output_truncated"),
        ):
            segments, result = inference._transcribe_smart_chunks(
                mlx,
                self._task(),
                ct2_rescue_holder=holder,
            )

        self.assertEqual([segment.text for segment in segments], ["one", "two"])
        factory.assert_called_once_with()
        self.assertNotIn("ct2_rescue_call_count", result.metrics)
        self.assertEqual(result.metrics["ct2_rescue_reject_count"], 2.0)

    def test_non_mlx_primary_backend_never_enters_rescue(self) -> None:
        inference = self._inference(backend_name="ct2", chunk_count=1)
        primary = FakeBackend([backend_result("ct2", "primary")])
        rescue = FakeBackend([])
        factory = mock.Mock(return_value=rescue)
        holder = _LazyCT2RescueBackend(factory)

        with mock.patch.object(infer_module, "decide_ct2_rescue") as decide:
            segments, _result = inference._transcribe_smart_chunks(
                primary,
                self._task(),
                ct2_rescue_holder=holder,
            )

        self.addCleanup(holder.close)
        self.assertEqual([segment.text for segment in segments], ["primary"])
        decide.assert_not_called()
        factory.assert_not_called()

    def test_candidate_gate_accepts_only_minimally_valid_ct2_results(self) -> None:
        valid = backend_result("ct2", "valid")
        empty = backend_result("ct2", "")
        nonfinite = backend_result("ct2", "bad", end=math.inf)
        hard_loop = backend_result("ct2", "晚安晚安晚安")
        unsupported_tail = backend_result("ct2", "unsupported", start=1.0, end=2.0)
        partially_supported_tail = backend_result("ct2", "noise", start=0.9, end=1.9)
        brief_silence = backend_result("ct2", "brief silence", start=0.1, end=0.9)

        self.assertEqual(
            _validate_ct2_rescue_candidate(
                None,
                chunk_duration=2.0,
                outer_vad_has_speech=True,
                outer_vad_speech_ranges=[(0.0, 1.0)],
            ),
            (False, "ct2_invalid_result"),
        )
        self.assertEqual(
            _validate_ct2_rescue_candidate(
                valid,
                chunk_duration=2.0,
                outer_vad_has_speech=True,
                outer_vad_speech_ranges=[(0.0, 1.0)],
            ),
            (True, "ct2_rescue_accepted"),
        )
        self.assertEqual(
            _validate_ct2_rescue_candidate(
                empty,
                chunk_duration=2.0,
                outer_vad_has_speech=True,
                outer_vad_speech_ranges=[(0.0, 1.0)],
            ),
            (False, "ct2_empty_output"),
        )
        self.assertEqual(
            _validate_ct2_rescue_candidate(
                nonfinite,
                chunk_duration=2.0,
                outer_vad_has_speech=True,
                outer_vad_speech_ranges=[(0.0, 1.0)],
            ),
            (False, "ct2_invalid_timestamp"),
        )
        self.assertEqual(
            _validate_ct2_rescue_candidate(
                hard_loop,
                chunk_duration=2.0,
                outer_vad_has_speech=True,
                outer_vad_speech_ranges=[(0.0, 1.0)],
            ),
            (False, "ct2_hard_loop"),
        )
        self.assertEqual(
            _validate_ct2_rescue_candidate(
                unsupported_tail,
                chunk_duration=2.0,
                outer_vad_has_speech=True,
                outer_vad_speech_ranges=[(0.0, 1.0)],
            ),
            (False, "ct2_tail_without_speech_support"),
        )
        self.assertEqual(
            _validate_ct2_rescue_candidate(
                partially_supported_tail,
                chunk_duration=2.0,
                outer_vad_has_speech=True,
                outer_vad_speech_ranges=[(0.9, 1.48)],
            ),
            (False, "ct2_tail_without_speech_support"),
        )
        self.assertEqual(
            _validate_ct2_rescue_candidate(
                brief_silence,
                chunk_duration=2.0,
                outer_vad_has_speech=True,
                outer_vad_speech_ranges=[(0.1, 0.45), (0.55, 0.9)],
            ),
            (True, "ct2_rescue_accepted"),
        )

    def test_rescue_backend_factory_uses_ct2_profile_defaults_without_mlx_path(self) -> None:
        inference = Inference.__new__(Inference)
        inference.profile = mock.Mock()
        inference.model_name_or_path = "/fake/mlx/model"
        inference.cpu_threads = 12
        selection = BackendSelection(
            requested="ct2",
            selected="ct2",
            descriptor=ModelDescriptor(
                backend="ct2",
                profile="translate",
                variant="int8",
                path=Path("/fake/ct2"),
            ),
            device="cpu",
        )
        backend = FakeBackend([])

        with (
            mock.patch.object(infer_module, "select_backend", return_value=selection) as select,
            mock.patch.object(infer_module, "create_backend", return_value=backend) as create,
        ):
            created = inference._create_ct2_rescue_backend()

        self.assertIs(created, backend)
        select.assert_called_once_with("ct2", inference.profile, variant=None, model_path=None)
        create.assert_called_once_with(
            selection,
            device="cpu",
            compute_type="int8",
            cpu_threads=12,
            enable_batching=False,
        )

    def test_generates_closes_created_rescue_backend_at_task_end(self) -> None:
        inference = Inference.__new__(Inference)
        inference.generation_config = {"task": "translate"}
        inference.backend_selection = BackendSelection(
            requested="mlx",
            selected="mlx",
            descriptor=ModelDescriptor(
                backend="mlx",
                profile="translate",
                variant="fp16",
                path=Path("/fake/mlx"),
            ),
        )
        inference.backend_name = "mlx"
        inference.profile = None
        inference.device = "gpu"
        inference.compute_type = "float16"
        inference.cpu_threads = 1
        inference.enable_batching = False
        inference.batch_size = 0
        inference.max_batch_size = 8
        inference.segment_merge_options = SegmentMergeOptions(enabled=False)
        inference.vad_injected = False
        inference.sub_writers = {}
        primary = FakeBackend([])
        rescue = FakeBackend([])

        with tempfile.TemporaryDirectory() as tmp_dir:
            task = InferenceTask(
                audio_path=str(Path(tmp_dir) / "audio.wav"),
                sub_prefix=str(Path(tmp_dir) / "output"),
                sub_formats=[],
            )

            def transcribe_smart_chunks(
                passed_backend: FakeBackend,
                passed_task: InferenceTask,
                *,
                ct2_rescue_holder: _LazyCT2RescueBackend | None,
            ) -> tuple[list[infer_module.Segment], BackendResult]:
                self.assertIs(passed_backend, primary)
                self.assertIs(passed_task, task)
                assert ct2_rescue_holder is not None
                self.assertIs(ct2_rescue_holder.get(), rescue)
                return [], backend_result("mlx", "")

            with (
                mock.patch.object(inference, "_scan", return_value=[task]),
                mock.patch.object(inference, "_ensure_runtime_ready"),
                mock.patch.object(inference, "_should_use_smart_split", return_value=True),
                mock.patch.object(inference, "_transcribe_smart_chunks", side_effect=transcribe_smart_chunks),
                mock.patch.object(inference, "_create_ct2_rescue_backend", return_value=rescue) as create_rescue,
                mock.patch.object(inference, "_log_duration"),
                mock.patch.object(infer_module, "create_backend", return_value=primary),
            ):
                status = inference.generates(["input"])

        self.assertEqual(status, infer_module.EXIT_OK)
        create_rescue.assert_called_once_with()
        self.assertTrue(rescue.closed)
        self.assertTrue(primary.closed)


if __name__ == "__main__":
    unittest.main()
