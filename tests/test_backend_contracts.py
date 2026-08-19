import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faster_whisper_transwithai_chickenrice.backends.base import (
    BackendConfigurationError,
    BackendRequest,
    BackendResult,
    BackendUnavailableError,
    ModelDescriptor,
    UnsupportedBackendOptionError,
)
from faster_whisper_transwithai_chickenrice.backends.factory import (
    BackendAvailability,
    probe_backend,
    select_backend,
)
from faster_whisper_transwithai_chickenrice.backends.mlx import MLXBackend
from faster_whisper_transwithai_chickenrice.backends.option_mapping import (
    map_ct2_options,
    map_mlx_options,
)
from faster_whisper_transwithai_chickenrice.profiles import get_profile


class OptionMappingTests(unittest.TestCase):
    def test_ct2_keeps_supported_options_and_removes_request_fields(self) -> None:
        mapped = map_ct2_options(
            {
                "language": "ja",
                "task": "translate",
                "beam_size": 1,
                "repetition_penalty": 1.1,
                "smart_split_with_vad": True,
                "mlx_use_outer_vad_clips": False,
                "mlx_safe_retry_without_clips": True,
                "mlx_debug_diagnostics": False,
                "mlx_sampling_seed": 0,
            }
        )

        self.assertEqual(mapped, {"beam_size": 1, "repetition_penalty": 1.1})

    def test_mlx_beam_size_one_uses_greedy_compatibility_mapping(self) -> None:
        mapping = map_mlx_options(
            {
                "beam_size": 1,
                "condition_on_previous_text": False,
                "repetition_penalty": 1.1,
                "vad_filter": True,
                "mlx_use_outer_vad_clips": False,
                "mlx_safe_retry_without_clips": True,
                "mlx_debug_diagnostics": False,
                "mlx_sampling_seed": 0,
            },
            task="translate",
        )

        self.assertNotIn("beam_size", mapping.options)
        self.assertTrue(mapping.options["fp16"])
        self.assertFalse(mapping.options["word_timestamps"])
        self.assertEqual(mapping.ignored, ("repetition_penalty", "vad_filter"))

    def test_mlx_rejects_unimplemented_beam_search(self) -> None:
        with self.assertRaisesRegex(UnsupportedBackendOptionError, "beam search"):
            map_mlx_options({"beam_size": 2}, task="translate")

    def test_mlx_strict_mode_rejects_unknown_options(self) -> None:
        with self.assertRaisesRegex(UnsupportedBackendOptionError, "unknown_option"):
            map_mlx_options({"unknown_option": True}, task="translate", strict_unknown=True)


class MLXBackendAdapterTests(unittest.TestCase):
    @staticmethod
    def _backend(transcribe: mock.Mock) -> MLXBackend:
        return MLXBackend(
            ModelDescriptor(
                backend="mlx",
                profile="translate",
                variant="fp16",
                path=Path("/models/mlx/translate/fp16"),
            ),
            transcribe_fn=transcribe,
        )

    @staticmethod
    def _request(options: dict[str, object] | None = None) -> BackendRequest:
        return BackendRequest(
            audio=[0.0] * 32_000,
            language="ja",
            task="translate",
            options=dict(options or {}),
        )

    @staticmethod
    def _runtime_result(
        text: str,
        *,
        fallback_exhausted: bool = False,
    ) -> dict[str, object]:
        final_failure_reasons = ["logprob"] if fallback_exhausted else []
        return {
            "language": "ja",
            "segments": [{"start": 0.25, "end": 1.5, "text": text}],
            "runtime_diagnostics": {
                "schema_version": 2,
                "fallback_exhausted": fallback_exhausted,
                "timestamp_violation_count": 0,
                "timestamp_overrun_seconds": 0.0,
                "windows": [
                    {
                        "seek_before": 0,
                        "seek_after": 200,
                        "seek_clip_start": 0,
                        "seek_clip_end": 200,
                        "total_content_frames": 200,
                        "content_frames": 200,
                        "padding_frames": 2_800,
                        "attempts": [],
                        "fallback_exhausted": fallback_exhausted,
                        "final_failure_reasons": final_failure_reasons,
                        "timestamp_violation_count": 0,
                        "timestamp_overrun_seconds": 0.0,
                    }
                ],
            },
        }

    def test_missing_runtime_diagnostics_safely_downgrades_without_blind_repair(self) -> None:
        transcribe = mock.Mock(
            return_value={
                "language": "ja",
                "segments": [
                    {"start": 0.25, "end": 1.5, "text": " 中文 "},
                ],
            }
        )
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request({"beam_size": 1, "repetition_penalty": 1.1}))

        self.assertEqual(result.backend, "mlx")
        self.assertEqual(result.duration, 2.0)
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].text, "中文")
        kwargs = transcribe.call_args.kwargs
        self.assertNotIn("beam_size", kwargs)
        self.assertNotIn("repetition_penalty", kwargs)
        self.assertEqual(kwargs["path_or_hf_repo"], "/models/mlx/translate/fp16")
        self.assertEqual(transcribe.call_count, 1)
        self.assertEqual(result.metrics["runtime_call_count"], 1.0)
        self.assertTrue(all(isinstance(value, (int, float)) for value in result.metrics.values()))
        self.assertEqual(result.diagnostics["sample_count"], 32_000)
        self.assertTrue(result.diagnostics["unresolved_anomaly"])
        self.assertEqual(result.diagnostics["selection_reason"], "evidence_incomplete")
        self.assertEqual(result.diagnostics["result_status"], "degraded_unresolved")

    def test_incomplete_runtime_schema_keeps_production_without_blind_repair(self) -> None:
        runtime_result = self._runtime_result("正常结果")
        runtime_result["runtime_diagnostics"] = {
            "schema_version": 1,
            "fallback_exhausted": False,
            "windows": [],
        }
        transcribe = mock.Mock(return_value=runtime_result)
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request())

        self.assertEqual(transcribe.call_count, 1)
        self.assertEqual(result.diagnostics["selected_result"], "initial")
        self.assertEqual(result.diagnostics["selection_reason"], "evidence_incomplete")
        self.assertTrue(result.diagnostics["initial"]["diagnostics_incomplete"])
        self.assertTrue(result.diagnostics["unresolved_anomaly"])

    def test_project_flags_are_not_forwarded_to_mlx_runtime(self) -> None:
        transcribe = mock.Mock(return_value=self._runtime_result("正常结果"))
        backend = self._backend(transcribe)

        backend.transcribe(
            self._request(
                {
                    "mlx_use_outer_vad_clips": False,
                    "mlx_safe_retry_without_clips": True,
                    "mlx_debug_diagnostics": True,
                    "mlx_sampling_seed": 42,
                }
            )
        )

        kwargs = transcribe.call_args.kwargs
        self.assertNotIn("mlx_use_outer_vad_clips", kwargs)
        self.assertNotIn("mlx_safe_retry_without_clips", kwargs)
        self.assertNotIn("mlx_debug_diagnostics", kwargs)
        self.assertNotIn("mlx_sampling_seed", kwargs)
        self.assertTrue(kwargs["diagnostic_details"])
        self.assertEqual(kwargs["sampling_seed"], 42)

    def test_invalid_sampling_seed_is_rejected_before_runtime_call(self) -> None:
        transcribe = mock.Mock(return_value=self._runtime_result("正常结果"))
        backend = self._backend(transcribe)

        with self.assertRaisesRegex(BackendConfigurationError, "mlx_sampling_seed"):
            backend.transcribe(self._request({"mlx_sampling_seed": 1.5}))

        transcribe.assert_not_called()

    def test_fallback_exhaustion_retries_once_without_clips(self) -> None:
        transcribe = mock.Mock(
            side_effect=[
                self._runtime_result("初次结果", fallback_exhausted=True),
                self._runtime_result("重试结果"),
            ]
        )
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request({"clip_timestamps": [0.25, 1.75]}))

        self.assertEqual(transcribe.call_count, 2)
        initial_kwargs = dict(transcribe.call_args_list[0].kwargs)
        repair_kwargs = dict(transcribe.call_args_list[1].kwargs)
        self.assertEqual(initial_kwargs.pop("clip_timestamps"), [0.25, 1.75])
        self.assertNotIn("clip_timestamps", repair_kwargs)
        self.assertEqual(repair_kwargs.pop("temperature"), 0.0)
        self.assertEqual(initial_kwargs, repair_kwargs)
        self.assertEqual(result.segments[0].text, "重试结果")
        self.assertEqual(result.diagnostics["repair_reason"], "fallback_exhausted")
        self.assertEqual(result.diagnostics["repair_mode"], "full_t0")
        self.assertEqual(result.diagnostics["selected_result"], "repair")
        self.assertTrue(result.diagnostics["selected_not_worse"])
        self.assertFalse(result.diagnostics["unresolved_anomaly"])
        self.assertEqual(result.diagnostics["result_status"], "clean")

    def test_high_confidence_loop_retries_once_without_clips(self) -> None:
        transcribe = mock.Mock(
            side_effect=[
                self._runtime_result("还算还算还算还算"),
                self._runtime_result("恢复正常"),
            ]
        )
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request({"clip_timestamps": [0.25, 1.75]}))

        self.assertEqual(transcribe.call_count, 2)
        self.assertEqual(result.diagnostics["repair_reason"], "high_confidence_loop")
        self.assertEqual(result.segments[0].text, "恢复正常")

    def test_loop_detection_runs_before_backend_segment_filtering(self) -> None:
        transcribe = mock.Mock(
            side_effect=[
                {
                    "language": "ja",
                    "segments": [{"start": 1.0, "end": 1.0, "text": "还算还算还算还算"}],
                    "runtime_diagnostics": {"fallback_exhausted": False},
                },
                self._runtime_result("恢复正常"),
            ]
        )
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request({"clip_timestamps": [0.25, 1.75]}))

        self.assertEqual(transcribe.call_count, 2)
        self.assertEqual(result.diagnostics["repair_reason"], "high_confidence_loop")
        self.assertEqual(result.segments[0].text, "恢复正常")

    def test_worse_repair_never_overwrites_initial_and_never_triggers_third_call(self) -> None:
        repair_text = "晚安晚安晚安"
        transcribe = mock.Mock(
            side_effect=[
                self._runtime_result("初次结果", fallback_exhausted=True),
                self._runtime_result(repair_text),
            ]
        )
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request({"clip_timestamps": [0.25, 1.75]}))

        self.assertEqual(transcribe.call_count, 2)
        self.assertEqual(result.segments[0].text, "初次结果")
        self.assertEqual(result.diagnostics["selected_result"], "initial")
        self.assertEqual(result.diagnostics["selection_reason"], "repair_not_strictly_better")
        self.assertTrue(result.diagnostics["repair_rejected"])
        self.assertTrue(result.diagnostics["selected_not_worse"])
        self.assertTrue(result.diagnostics["unresolved_anomaly"])
        self.assertEqual(result.metrics["unresolved_anomaly"], 1.0)
        self.assertEqual(result.diagnostics["result_status"], "degraded_unresolved")

    def test_fallback_exhausted_repair_without_hard_loop_improvement_keeps_initial(self) -> None:
        initial = self._runtime_result("初次结果", fallback_exhausted=True)
        initial["segments"] = [{"start": 0.25, "end": 2.5, "text": "初次结果"}]
        initial["runtime_diagnostics"]["timestamp_violation_count"] = 1
        initial["runtime_diagnostics"]["timestamp_overrun_seconds"] = 0.5
        initial["runtime_diagnostics"]["windows"][0]["timestamp_violation_count"] = 1
        initial["runtime_diagnostics"]["windows"][0]["timestamp_overrun_seconds"] = 0.5
        repair = self._runtime_result("修复新增尾句", fallback_exhausted=True)
        transcribe = mock.Mock(side_effect=[initial, repair])
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request())

        initial_record = result.diagnostics["initial"]
        repair_record = result.diagnostics["repair"]
        self.assertIsNotNone(repair_record)
        initial_quality = initial_record["quality"]
        repair_quality = repair_record["quality"]
        self.assertLess(repair_quality["comparison_key"], initial_quality["comparison_key"])
        self.assertEqual(repair_quality["hard_loop_severity"], initial_quality["hard_loop_severity"])
        self.assertTrue(repair_record["fallback_exhausted"])
        self.assertEqual(transcribe.call_count, 2)
        self.assertEqual(result.metrics["runtime_call_count"], 2.0)
        self.assertEqual(result.diagnostics["selected_result"], "initial")
        self.assertEqual(
            result.diagnostics["selection_reason"],
            "repair_fallback_without_hard_loop_improvement",
        )
        self.assertTrue(result.diagnostics["repair_rejected"])
        self.assertEqual(result.diagnostics["selected_quality"], initial_quality)
        self.assertTrue(result.diagnostics["selected_not_worse"])
        self.assertEqual(result.segments[0].text, "初次结果")

    def test_fallback_exhausted_repair_with_strict_hard_loop_improvement_can_win(self) -> None:
        initial = self._runtime_result("还算还算还算还算", fallback_exhausted=True)
        repair = self._runtime_result("还算还算还算", fallback_exhausted=True)
        transcribe = mock.Mock(side_effect=[initial, repair])
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request())

        initial_record = result.diagnostics["initial"]
        repair_record = result.diagnostics["repair"]
        self.assertIsNotNone(repair_record)
        initial_quality = initial_record["quality"]
        repair_quality = repair_record["quality"]
        self.assertLess(repair_quality["hard_loop_severity"], initial_quality["hard_loop_severity"])
        self.assertLess(repair_quality["comparison_key"], initial_quality["comparison_key"])
        self.assertTrue(repair_record["fallback_exhausted"])
        self.assertEqual(transcribe.call_count, 2)
        self.assertEqual(result.metrics["runtime_call_count"], 2.0)
        self.assertEqual(result.diagnostics["selected_result"], "repair")
        self.assertEqual(result.diagnostics["selection_reason"], "repair_strictly_better")
        self.assertFalse(result.diagnostics["repair_rejected"])
        self.assertTrue(result.diagnostics["unresolved_anomaly"])
        self.assertEqual(result.segments[0].text, "还算还算还算")

    def test_full_chunk_hard_loop_gets_one_full_t0_repair_candidate(self) -> None:
        transcribe = mock.Mock(
            side_effect=[
                self._runtime_result("还算还算还算还算"),
                self._runtime_result("恢复正常"),
            ]
        )
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request())

        self.assertEqual(transcribe.call_count, 2)
        self.assertEqual(result.diagnostics["initial_clip_mode"], "full_chunk")
        self.assertEqual(transcribe.call_args_list[1].kwargs["temperature"], 0.0)
        self.assertNotIn("clip_timestamps", transcribe.call_args_list[1].kwargs)
        self.assertEqual(result.diagnostics["selected_result"], "repair")
        self.assertFalse(result.diagnostics["unresolved_anomaly"])

    def test_non_full_clips_without_anomaly_are_not_retried(self) -> None:
        transcribe = mock.Mock(return_value=self._runtime_result("正常结果"))
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request({"clip_timestamps": [0.25, 1.75]}))

        self.assertEqual(transcribe.call_count, 1)
        self.assertEqual(transcribe.call_args.kwargs["sampling_seed"], 0)
        self.assertEqual(result.diagnostics["initial_clip_mode"], "clips")
        self.assertFalse(result.diagnostics["unresolved_anomaly"])

    def test_full_fixed_t0_initial_is_not_repeated_as_an_identical_repair(self) -> None:
        transcribe = mock.Mock(return_value=self._runtime_result("还算还算还算还算"))
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request({"temperature": 0.0}))

        self.assertEqual(transcribe.call_count, 1)
        self.assertEqual(result.diagnostics["initial_clip_mode"], "full_chunk")
        self.assertEqual(result.diagnostics["selected_result"], "initial")
        self.assertEqual(result.diagnostics["selection_reason"], "repair_not_distinct")
        self.assertIsNone(result.diagnostics["repair"])
        self.assertTrue(result.diagnostics["unresolved_anomaly"])

    def test_timestamp_overrun_triggers_one_full_t0_repair(self) -> None:
        initial = self._runtime_result("越界结果")
        initial["segments"] = [{"start": 0.25, "end": 2.5, "text": "越界结果"}]
        initial["runtime_diagnostics"]["timestamp_violation_count"] = 1
        initial["runtime_diagnostics"]["timestamp_overrun_seconds"] = 0.5
        initial["runtime_diagnostics"]["windows"][0]["timestamp_violation_count"] = 1
        initial["runtime_diagnostics"]["windows"][0]["timestamp_overrun_seconds"] = 0.5
        transcribe = mock.Mock(side_effect=[initial, self._runtime_result("恢复正常")])
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request())

        self.assertEqual(transcribe.call_count, 2)
        self.assertEqual(transcribe.call_args_list[1].kwargs["temperature"], 0.0)
        self.assertEqual(result.diagnostics["repair_reason"], "timestamp_violation")
        self.assertEqual(result.diagnostics["selected_result"], "repair")
        self.assertEqual(result.segments[0].text, "恢复正常")

    def test_adapter_clamps_timestamp_overrun_and_records_evidence_when_repair_disabled(self) -> None:
        transcribe = mock.Mock(
            return_value={
                "language": "ja",
                "segments": [{"start": 0.25, "end": 2.5, "text": "正常结果"}],
                "runtime_diagnostics": {
                    "schema_version": 2,
                    "fallback_exhausted": False,
                    "timestamp_violation_count": 1,
                    "timestamp_overrun_seconds": 0.5,
                    "windows": [
                        {
                            "seek_before": 0,
                            "seek_after": 200,
                            "seek_clip_start": 0,
                            "seek_clip_end": 200,
                            "total_content_frames": 200,
                            "content_frames": 200,
                            "padding_frames": 2_800,
                            "attempts": [],
                            "fallback_exhausted": False,
                            "final_failure_reasons": [],
                            "timestamp_violation_count": 1,
                            "timestamp_overrun_seconds": 0.5,
                        }
                    ],
                },
            }
        )
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request({"mlx_safe_retry_without_clips": False}))

        self.assertEqual(transcribe.call_count, 1)
        self.assertEqual(result.segments[0].end, 2.0)
        self.assertEqual(result.metrics["adapter_timestamp_clamp_count"], 1.0)
        self.assertEqual(result.diagnostics["adapter"]["adapter_timestamp_clamp_count"], 1)
        self.assertEqual(
            result.diagnostics["initial"]["quality"]["invalid_timestamp_count"],
            1,
        )

    def test_full_initial_fallback_exhaustion_runs_different_full_t0_repair_once(self) -> None:
        transcribe = mock.Mock(
            side_effect=[
                self._runtime_result("初次结果", fallback_exhausted=True),
                self._runtime_result("修复结果"),
            ]
        )
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request({"clip_timestamps": [0.0, 2.0]}))

        self.assertEqual(transcribe.call_count, 2)
        self.assertEqual(result.diagnostics["initial_clip_mode"], "full_chunk")
        self.assertEqual(transcribe.call_args_list[1].kwargs["temperature"], 0.0)
        self.assertNotIn("clip_timestamps", transcribe.call_args_list[1].kwargs)
        self.assertEqual(result.diagnostics["selected_result"], "repair")
        self.assertFalse(result.diagnostics["unresolved_anomaly"])

    def test_retry_can_be_disabled_and_control_option_is_not_forwarded(self) -> None:
        transcribe = mock.Mock(return_value=self._runtime_result("初次结果", fallback_exhausted=True))
        backend = self._backend(transcribe)

        result = backend.transcribe(
            self._request(
                {
                    "clip_timestamps": [0.25, 1.75],
                    "mlx_safe_retry_without_clips": False,
                }
            )
        )

        self.assertEqual(transcribe.call_count, 1)
        self.assertNotIn("mlx_safe_retry_without_clips", transcribe.call_args.kwargs)
        self.assertEqual(result.metrics["safe_retry_count"], 0.0)
        self.assertTrue(result.diagnostics["unresolved_anomaly"])

    def test_b6_spaced_short_repetition_is_soft_and_never_rewritten_or_deleted(self) -> None:
        transcribe = mock.Mock(
            return_value={
                "language": "ja",
                "segments": [
                    {"start": 13.22, "end": 14.22, "text": "晚安"},
                    {"start": 17.94, "end": 18.94, "text": "晚安"},
                ],
                "runtime_diagnostics": {
                    "schema_version": 2,
                    "fallback_exhausted": False,
                    "timestamp_violation_count": 0,
                    "timestamp_overrun_seconds": 0.0,
                    "windows": [
                        {
                            "seek_before": 0,
                            "seek_after": 2_411,
                            "seek_clip_start": 0,
                            "seek_clip_end": 2_411,
                            "total_content_frames": 2_411,
                            "content_frames": 2_411,
                            "padding_frames": 589,
                            "attempts": [],
                            "fallback_exhausted": False,
                            "final_failure_reasons": [],
                            "timestamp_violation_count": 0,
                            "timestamp_overrun_seconds": 0.0,
                        }
                    ],
                },
            }
        )
        backend = self._backend(transcribe)

        request = BackendRequest(
            audio=[0.0] * 385_760,
            language="ja",
            task="translate",
            options={},
        )
        result = backend.transcribe(request)

        self.assertEqual(transcribe.call_count, 1)
        self.assertEqual([segment.text for segment in result.segments], ["晚安", "晚安"])
        self.assertFalse(result.diagnostics["unresolved_anomaly"])
        self.assertTrue(result.diagnostics["soft_review_required"])
        self.assertEqual(result.diagnostics["selection_reason"], "soft_review_only")
        self.assertEqual(result.diagnostics["result_status"], "review_required")

    def test_seek_nonprogress_triggers_one_repair(self) -> None:
        initial = self._runtime_result("初次结果")
        initial["runtime_diagnostics"]["windows"][0]["seek_after"] = 0
        transcribe = mock.Mock(side_effect=[initial, self._runtime_result("修复结果")])
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request())

        self.assertEqual(transcribe.call_count, 2)
        self.assertEqual(result.diagnostics["repair_reason"], "seek_invariant")
        self.assertEqual(result.diagnostics["selected_result"], "repair")

    def test_repeated_frame_range_triggers_one_repair(self) -> None:
        initial = self._runtime_result("初次结果")
        repeated_window = dict(initial["runtime_diagnostics"]["windows"][0])
        initial["runtime_diagnostics"]["windows"].append(repeated_window)
        transcribe = mock.Mock(side_effect=[initial, self._runtime_result("修复结果")])
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request())

        self.assertEqual(transcribe.call_count, 2)
        self.assertEqual(result.diagnostics["repair_reason"], "repeated_frame_range")
        self.assertEqual(result.diagnostics["selected_result"], "repair")

    def test_invalid_zero_length_segment_triggers_one_repair(self) -> None:
        initial = self._runtime_result("坏片段")
        initial["segments"] = [{"start": 1.0, "end": 1.0, "text": "坏片段"}]
        transcribe = mock.Mock(side_effect=[initial, self._runtime_result("修复结果")])
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request())

        self.assertEqual(transcribe.call_count, 2)
        self.assertEqual(result.diagnostics["repair_reason"], "timestamp_violation")
        self.assertIn("invalid_segment", result.diagnostics["initial"]["repair_reasons"])
        self.assertEqual(result.diagnostics["initial"]["quality"]["invalid_segment_count"], 1)
        self.assertEqual(result.diagnostics["selected_result"], "repair")

    def test_empty_production_output_triggers_one_repair(self) -> None:
        initial = self._runtime_result("")
        initial["segments"] = []
        transcribe = mock.Mock(side_effect=[initial, self._runtime_result("修复结果")])
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request())

        self.assertEqual(transcribe.call_count, 2)
        self.assertEqual(result.diagnostics["repair_reason"], "empty_output")
        self.assertEqual(result.diagnostics["selected_result"], "repair")

    def test_explicitly_truncated_output_triggers_one_repair(self) -> None:
        initial = self._runtime_result("被截断")
        initial["output_truncated"] = True
        transcribe = mock.Mock(side_effect=[initial, self._runtime_result("修复结果")])
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request())

        self.assertEqual(transcribe.call_count, 2)
        self.assertEqual(result.diagnostics["repair_reason"], "output_truncated")
        self.assertEqual(result.diagnostics["selected_result"], "repair")

    def test_incomplete_repair_evidence_cannot_overwrite_complete_production(self) -> None:
        initial = self._runtime_result("还算还算还算还算")
        repair = self._runtime_result("看似正常")
        repair["runtime_diagnostics"] = {"schema_version": 1, "windows": []}
        transcribe = mock.Mock(side_effect=[initial, repair])
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request())

        self.assertEqual(transcribe.call_count, 2)
        self.assertEqual(result.diagnostics["selected_result"], "initial")
        self.assertTrue(result.diagnostics["repair_rejected"])
        self.assertTrue(result.diagnostics["unresolved_anomaly"])

    def test_complete_but_more_severe_hard_loop_cannot_replace_incomplete_initial(self) -> None:
        initial = self._runtime_result("还算还算还算还算")
        initial["runtime_diagnostics"] = {"schema_version": 1, "windows": []}
        repair = self._runtime_result("晚安晚安晚安晚安晚安")
        transcribe = mock.Mock(side_effect=[initial, repair])
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request())

        self.assertEqual(transcribe.call_count, 2)
        self.assertEqual(result.diagnostics["selected_result"], "initial")
        self.assertTrue(result.diagnostics["repair_rejected"])
        self.assertTrue(result.diagnostics["unresolved_anomaly"])

    def test_equal_quality_repair_keeps_initial(self) -> None:
        repeated = self._runtime_result("还算还算还算还算")
        transcribe = mock.Mock(side_effect=[repeated, repeated])
        backend = self._backend(transcribe)

        result = backend.transcribe(self._request())

        self.assertEqual(transcribe.call_count, 2)
        self.assertEqual(result.diagnostics["selected_result"], "initial")
        self.assertEqual(result.diagnostics["selection_reason"], "repair_not_strictly_better")
        self.assertTrue(result.diagnostics["selected_not_worse"])

    def test_backend_result_diagnostics_default_is_empty_for_ct2(self) -> None:
        result = BackendResult(
            segments=[],
            duration=0.0,
            duration_after_vad=None,
            language="ja",
            backend="ct2",
        )

        self.assertEqual(result.diagnostics, {})


class BackendFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = get_profile("translate")
        self.mlx_descriptor = self.profile.descriptor("mlx")
        self.ct2_descriptor = self.profile.descriptor("ct2")

    def test_auto_falls_back_only_during_preflight(self) -> None:
        unavailable_mlx = BackendAvailability(
            backend="mlx",
            available=False,
            descriptor=self.mlx_descriptor,
            reasons=("Metal unavailable",),
        )
        available_ct2 = BackendAvailability(
            backend="ct2",
            available=True,
            descriptor=self.ct2_descriptor,
            device="cpu",
        )
        with mock.patch(
            "faster_whisper_transwithai_chickenrice.backends.factory.probe_backend",
            side_effect=[unavailable_mlx, available_ct2],
        ):
            selection = select_backend("auto", self.profile)

        self.assertEqual(selection.selected, "ct2")
        self.assertIn("Metal unavailable", selection.fallback_reason)

    def test_explicit_mlx_never_falls_back_to_ct2(self) -> None:
        unavailable_mlx = BackendAvailability(
            backend="mlx",
            available=False,
            descriptor=self.mlx_descriptor,
            reasons=("Metal unavailable",),
        )
        with (
            mock.patch(
                "faster_whisper_transwithai_chickenrice.backends.factory.probe_backend",
                return_value=unavailable_mlx,
            ) as probe,
            self.assertRaisesRegex(BackendUnavailableError, "Metal unavailable"),
        ):
            select_backend("mlx", self.profile)

        self.assertEqual(probe.call_count, 1)

    def test_probe_without_runtime_check_is_ci_safe(self) -> None:
        with (
            mock.patch(
                "faster_whisper_transwithai_chickenrice.backends.factory.validate_mlx_model",
                return_value=([], []),
            ),
            mock.patch(
                "faster_whisper_transwithai_chickenrice.backends.factory.platform.machine",
                return_value="arm64",
            ),
        ):
            availability = probe_backend(
                self.profile,
                "mlx",
                check_runtime=False,
                verify_hashes=False,
            )

        self.assertTrue(availability.available)
        self.assertEqual(availability.device, "gpu")


class ProfileSchemaTests(unittest.TestCase):
    def test_transcribe_assets_are_isolated_from_translate_assets(self) -> None:
        translate = get_profile("translate").descriptor("mlx")
        transcribe = get_profile("transcribe").descriptor("mlx")

        self.assertNotEqual(translate.path, transcribe.path)
        self.assertIn("translate", translate.path.parts)
        self.assertIn("transcribe", transcribe.path.parts)


if __name__ == "__main__":
    unittest.main()
