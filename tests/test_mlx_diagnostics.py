import json
import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faster_whisper_transwithai_chickenrice.backends.base import (
    BackendResult,
    BackendSegment,
)
from faster_whisper_transwithai_chickenrice.mlx_diagnostics import (
    MLX_DEBUG_MAX_ATTEMPTS,
    MLX_DEBUG_MAX_BYTES,
    MLX_DEBUG_MAX_WINDOWS,
    MLX_EVENT_REQUIRED_FIELDS,
    CT2RescueDecision,
    analyze_runtime_result,
    build_mlx_diagnostic_events,
    decide_ct2_rescue,
    detect_high_confidence_loops,
    evaluate_candidate_quality,
    limit_mlx_diagnostic_events,
    mlx_diagnostic_events_size,
    normalize_loop_text,
    runtime_fallback_exhausted,
    serialize_mlx_diagnostic_event,
)


class LoopNormalizationTests(unittest.TestCase):
    def test_normalizes_width_case_and_punctuation(self) -> None:
        self.assertEqual(normalize_loop_text(" Ａb，Ｃ！１２ "), "abc12")


class LoopDetectionTests(unittest.TestCase):
    @staticmethod
    def _rules(text: str) -> set[str]:
        result = detect_high_confidence_loops([{"start": 0.0, "end": 1.0, "text": text}])
        return {str(finding["rule"]) for finding in result.findings}

    def test_single_character_requires_four_consecutive_repeats(self) -> None:
        self.assertNotIn("consecutive_unit", self._rules("哈哈哈"))
        self.assertIn("consecutive_unit", self._rules("哈哈哈哈"))

    def test_two_or_three_character_unit_requires_three_repeats(self) -> None:
        self.assertNotIn("consecutive_unit", self._rules("还算还算"))
        self.assertIn("consecutive_unit", self._rules("还算还算还算"))
        self.assertIn("consecutive_unit", self._rules("abcabcabc"))

    def test_four_character_unit_requires_two_repeats(self) -> None:
        self.assertIn("consecutive_unit", self._rules("测试循环测试循环"))

    def test_adjacent_identical_segments_are_detected(self) -> None:
        result = detect_high_confidence_loops(
            [
                {"start": 0.0, "end": 1.0, "text": "今天的天气很好"},
                {"start": 1.0, "end": 2.0, "text": "今天的天气很好"},
            ]
        )

        self.assertIn("adjacent_segment_similarity", {finding["rule"] for finding in result.findings})

    def test_adjacent_segments_at_similarity_threshold_are_detected(self) -> None:
        result = detect_high_confidence_loops(
            [
                {"start": 0.0, "end": 1.0, "text": "abcdefghij"},
                {"start": 1.0, "end": 2.0, "text": "abcdefghiX"},
            ]
        )

        adjacent = [finding for finding in result.findings if finding["rule"] == "adjacent_segment_similarity"]
        self.assertEqual(len(adjacent), 1)
        self.assertGreaterEqual(adjacent[0]["similarity"], 0.88)

    def test_b6_spaced_short_repetition_is_soft_review_evidence(self) -> None:
        result = detect_high_confidence_loops(
            [
                {"start": 13.22, "end": 14.22, "text": "晚安"},
                {"start": 17.94, "end": 18.94, "text": "晚安"},
            ],
            duration=24.11,
        )

        self.assertFalse(result.loop_detected)
        self.assertTrue(result.review_required)
        self.assertEqual(len(result.soft_findings), 1)
        self.assertEqual(result.soft_findings[0]["severity"], "soft")
        self.assertAlmostEqual(result.soft_findings[0]["gap_seconds"], 3.72)

    def test_spaced_short_soft_rule_is_not_a_wanan_word_allowlist(self) -> None:
        result = detect_high_confidence_loops(
            [
                {"start": 2.0, "end": 3.0, "text": "再见"},
                {"start": 6.72, "end": 7.72, "text": "再见"},
            ],
            duration=10.0,
        )

        self.assertFalse(result.loop_detected)
        self.assertTrue(result.review_required)
        self.assertAlmostEqual(result.soft_findings[0]["gap_seconds"], 3.72)

    def test_b6_unheard_tail_repetition_remains_hard_alongside_soft_repeat(self) -> None:
        result = detect_high_confidence_loops(
            [
                {"start": 0.1, "end": 0.3, "text": "晚安"},
                {"start": 1.4, "end": 1.6, "text": "晚安"},
                {"start": 1.7, "end": 1.95, "text": "永远永远永远永远永远"},
            ],
            duration=2.0,
        )

        self.assertTrue(result.loop_detected)
        self.assertTrue(result.review_required)
        self.assertGreaterEqual(len(result.hard_findings), 1)
        self.assertEqual(len(result.soft_findings), 1)

    def test_b1_dense_out_of_bounds_kuku_pair_remains_hard(self) -> None:
        result = detect_high_confidence_loops(
            [
                {"start": 24.98, "end": 25.98, "text": "哭哭"},
                {"start": 26.22, "end": 27.22, "text": "哭哭"},
            ],
            duration=26.11,
        )

        self.assertTrue(result.loop_detected)
        self.assertEqual(result.hard_findings[0]["severity"], "hard")
        self.assertFalse(result.hard_findings[0]["bounds_valid"])

    def test_a1_long_phrase_pairs_remain_hard(self) -> None:
        for text in ("按的几号呢按的几号呢", "按这个就好按这个就好"):
            with self.subTest(text=text):
                result = detect_high_confidence_loops([{"start": 0.0, "end": 1.0, "text": text}])

                self.assertTrue(result.loop_detected)
                self.assertIn("consecutive_unit", {finding["rule"] for finding in result.hard_findings})

    def test_dense_or_out_of_bounds_adjacent_repetition_is_hard(self) -> None:
        dense = detect_high_confidence_loops(
            [
                {"start": 0.1, "end": 0.8, "text": "短句"},
                {"start": 0.9, "end": 2.4, "text": "短句"},
            ],
            duration=2.0,
        )

        self.assertTrue(dense.loop_detected)
        self.assertEqual(dense.hard_findings[0]["severity"], "hard")

    def test_three_identical_segments_within_45_seconds_are_detected(self) -> None:
        result = detect_high_confidence_loops(
            [
                {"start": 0.0, "end": 1.0, "text": "相同句子"},
                {"start": 4.0, "end": 5.0, "text": "其他内容甲"},
                {"start": 10.0, "end": 11.0, "text": "相同句子"},
                {"start": 14.0, "end": 15.0, "text": "其他内容乙"},
                {"start": 20.0, "end": 21.0, "text": "相同句子"},
            ]
        )

        self.assertIn("repeated_segment_within_45s", {finding["rule"] for finding in result.findings})

    def test_identical_segments_outside_45_second_window_are_not_detected_by_window_rule(self) -> None:
        result = detect_high_confidence_loops(
            [
                {"start": 0.0, "end": 1.0, "text": "相同句子"},
                {"start": 19.0, "end": 20.0, "text": "其他内容甲"},
                {"start": 30.0, "end": 31.0, "text": "相同句子"},
                {"start": 49.0, "end": 50.0, "text": "其他内容乙"},
                {"start": 60.0, "end": 61.0, "text": "相同句子"},
            ]
        )

        self.assertNotIn("repeated_segment_within_45s", {finding["rule"] for finding in result.findings})

    def test_findings_store_hashes_without_text_or_tokens(self) -> None:
        original = "秘密秘密秘密"
        result = detect_high_confidence_loops([{"start": 0.0, "end": 1.0, "text": original}])
        encoded = json.dumps(result.as_dict(), ensure_ascii=False)

        self.assertTrue(result.loop_detected)
        self.assertNotIn(original, encoded)
        self.assertNotIn("秘密", encoded)
        self.assertIn("text_sha256", result.findings[0])
        self.assertEqual(result.findings[0]["text_length"], len(original))
        self.assertNotIn("text", result.findings[0])
        self.assertNotIn("tokens", result.findings[0])


class RuntimeDiagnosticsTests(unittest.TestCase):
    def test_fallback_exhaustion_can_be_reported_at_top_level_or_window_level(self) -> None:
        self.assertTrue(runtime_fallback_exhausted({"fallback_exhausted": True}))
        self.assertTrue(runtime_fallback_exhausted({"windows": [{"fallback_exhausted": True}]}))
        self.assertFalse(runtime_fallback_exhausted({"windows": [{"fallback_exhausted": False}]}))

    def test_anomaly_reason_reports_both_signals(self) -> None:
        anomaly = analyze_runtime_result(
            {
                "runtime_diagnostics": {"fallback_exhausted": True},
                "segments": [{"start": 0.0, "end": 1.0, "text": "还算还算还算"}],
            }
        )

        self.assertTrue(anomaly.anomaly_detected)
        self.assertEqual(anomaly.anomaly_reason, "both")


class CandidateQualityTests(unittest.TestCase):
    @staticmethod
    def _result(
        *,
        text: str = "正常",
        fallback_exhausted: bool = False,
        end: float = 1.5,
    ) -> dict[str, Any]:
        return {
            "segments": [{"start": 0.25, "end": end, "text": text}],
            "runtime_diagnostics": {
                "schema_version": 2,
                "fallback_exhausted": fallback_exhausted,
                "timestamp_violation_count": int(end > 2.0),
                "timestamp_overrun_seconds": max(0.0, end - 2.0),
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
                        "final_failure_reasons": ["compression_ratio"] if fallback_exhausted else [],
                        "timestamp_violation_count": int(end > 2.0),
                        "timestamp_overrun_seconds": max(0.0, end - 2.0),
                    }
                ],
            },
        }

    def test_quality_vector_prefers_non_exhausted_candidate(self) -> None:
        exhausted = evaluate_candidate_quality(
            self._result(fallback_exhausted=True),
            sample_count=32_000,
        )
        accepted = evaluate_candidate_quality(self._result(), sample_count=32_000)

        self.assertLess(accepted.comparison_key, exhausted.comparison_key)
        self.assertEqual(exhausted.fallback_exhausted_window_count, 1)
        self.assertEqual(exhausted.terminal_failure_reason_count, 1)

    def test_hard_loop_outweighs_later_vector_improvements(self) -> None:
        initial = evaluate_candidate_quality(
            self._result(fallback_exhausted=True),
            sample_count=32_000,
        )
        hard_repair = evaluate_candidate_quality(
            self._result(text="晚安晚安晚安"),
            sample_count=32_000,
        )

        self.assertLess(initial.comparison_key, hard_repair.comparison_key)

    def test_hard_loop_severity_distinguishes_more_extreme_repetition(self) -> None:
        lighter = evaluate_candidate_quality(
            self._result(text="晚安晚安晚安"),
            sample_count=32_000,
        )
        heavier = evaluate_candidate_quality(
            self._result(text="晚安晚安晚安晚安晚安"),
            sample_count=32_000,
        )

        self.assertLess(lighter.hard_loop_severity, heavier.hard_loop_severity)
        self.assertLess(lighter.comparison_key, heavier.comparison_key)

    def test_timestamp_overrun_is_counted_in_samples(self) -> None:
        quality = evaluate_candidate_quality(self._result(end=2.25), sample_count=32_000)

        self.assertEqual(quality.invalid_timestamp_count, 1)
        self.assertEqual(quality.timestamp_overrun_samples, 4_000)
        self.assertEqual(quality.as_dict()["comparison_key"], list(quality.comparison_key))

    def test_quality_vector_serializes_seek_empty_and_evidence_defects(self) -> None:
        result = self._result(text="")
        result["segments"] = []
        result["runtime_diagnostics"]["schema_version"] = 1
        result["runtime_diagnostics"]["windows"][0]["seek_after"] = 0

        quality = evaluate_candidate_quality(result, sample_count=32_000)
        serialized = quality.as_dict()

        self.assertEqual(quality.diagnostics_incomplete_count, 1)
        self.assertEqual(quality.seek_nonprogress_count, 1)
        self.assertEqual(quality.empty_output_count, 1)
        self.assertEqual(serialized["comparison_key"], list(quality.comparison_key))

    def test_evidence_completeness_breaks_ties_after_observed_hard_defects(self) -> None:
        complete_hard_loop = evaluate_candidate_quality(
            self._result(text="还算还算还算还算"),
            sample_count=32_000,
        )
        complete_clean = evaluate_candidate_quality(self._result(text="正常"), sample_count=32_000)
        incomplete_clean_text = self._result(text="正常")
        incomplete_clean_text["runtime_diagnostics"]["schema_version"] = 1
        incomplete = evaluate_candidate_quality(incomplete_clean_text, sample_count=32_000)

        self.assertLess(complete_clean.comparison_key, incomplete.comparison_key)
        self.assertLess(incomplete.comparison_key, complete_hard_loop.comparison_key)


class CT2RescueDecisionTests(unittest.TestCase):
    @staticmethod
    def _record(
        *,
        fallback_exhausted: bool = False,
        diagnostics_incomplete: bool = False,
        empty_output: bool = False,
        output_truncated: bool = False,
        invalid_segment_count: int = 0,
        timestamp_violation_count: int = 0,
        timestamp_overrun_samples: int = 0,
        seek_nonprogress_count: int = 0,
        repeated_frame_range_count: int = 0,
        hard_loop_severity: int = 0,
        soft_review_required: bool = False,
    ) -> dict[str, Any]:
        hard_finding_count = int(hard_loop_severity > 0)
        soft_finding_count = int(soft_review_required)
        repair_required = bool(
            fallback_exhausted
            or empty_output
            or output_truncated
            or invalid_segment_count
            or timestamp_violation_count
            or timestamp_overrun_samples
            or seek_nonprogress_count
            or repeated_frame_range_count
            or hard_finding_count
        )
        return {
            "anomaly_detected": repair_required or diagnostics_incomplete,
            "repair_required": repair_required,
            "fallback_exhausted": fallback_exhausted,
            "diagnostics_incomplete": diagnostics_incomplete,
            "empty_output": empty_output,
            "output_truncated": output_truncated,
            "invalid_segment_count": invalid_segment_count,
            "timestamp_violation_count": timestamp_violation_count,
            "timestamp_overrun_samples": timestamp_overrun_samples,
            "seek_nonprogress_count": seek_nonprogress_count,
            "repeated_frame_range_count": repeated_frame_range_count,
            "loop_detected": bool(hard_finding_count),
            "soft_review_required": soft_review_required,
            "hard_finding_count": hard_finding_count,
            "soft_finding_count": soft_finding_count,
            "quality": {
                "diagnostics_incomplete_count": int(diagnostics_incomplete),
                "hard_loop_severity": hard_loop_severity,
                "hard_finding_count": hard_finding_count,
            },
        }

    @classmethod
    def _result(
        cls,
        *,
        selected_result: str = "initial",
        initial_overrides: dict[str, Any] | None = None,
        repair_overrides: dict[str, Any] | None = None,
        segments: list[BackendSegment] | None = None,
        repair_ran: bool = True,
    ) -> BackendResult:
        initial_options: dict[str, Any] = {"fallback_exhausted": True}
        repair_options: dict[str, Any] = {"fallback_exhausted": True}
        initial_options.update(initial_overrides or {})
        repair_options.update(repair_overrides or {})
        initial = cls._record(**initial_options)
        repair = cls._record(**repair_options) if repair_ran else None
        selected = repair if selected_result == "repair" and repair is not None else initial
        final_segments = segments if segments is not None else [BackendSegment(0.1, 0.9, "正常")]
        diagnostics: dict[str, Any] = {
            "schema_version": 2,
            "sample_count": 16_000,
            "quality_sample_count": 16_000,
            "initial": initial,
            "repair": repair,
            "repair_reason": "fallback_exhausted" if repair_ran else None,
            "repair_mode": "full_t0" if repair_ran else None,
            "repair_rejected": repair_ran and selected_result == "initial",
            "safe_retry_count": int(repair_ran),
            "selected_result": selected_result,
            "selection_reason": (
                "repair_strictly_better"
                if selected_result == "repair"
                else "repair_fallback_without_hard_loop_improvement"
                if repair_ran
                else "initial_acceptable"
            ),
            "selected_quality": dict(selected["quality"]),
            "selected_not_worse": True,
            "unresolved_anomaly": selected["anomaly_detected"],
            "soft_review_required": selected["soft_review_required"],
            "result_status": "degraded_unresolved" if selected["anomaly_detected"] else "clean",
            "adapter": {
                "raw_segment_count": len(final_segments),
                "adapter_timestamp_clamp_count": 0,
                "adapter_invalid_timestamp_count": 0,
                "adapter_segment_drop_count": 0,
            },
        }
        return BackendResult(
            segments=final_segments,
            duration=1.0,
            duration_after_vad=None,
            language="zh",
            backend="mlx",
            metrics={
                "runtime_call_count": 2.0 if repair_ran else 1.0,
                "safe_retry_count": 1.0 if repair_ran else 0.0,
            },
            diagnostics=diagnostics,
        )

    @staticmethod
    def _decide(
        result: BackendResult,
        *,
        outer_vad_has_speech: bool = False,
        ct2_rescue_already_attempted: bool = False,
    ) -> CT2RescueDecision:
        return decide_ct2_rescue(
            result,
            result.diagnostics,
            outer_vad_has_speech=outer_vad_has_speech,
            ct2_rescue_already_attempted=ct2_rescue_already_attempted,
        )

    def test_fallback_exhausted_alone_does_not_trigger(self) -> None:
        result = self._result(
            selected_result="repair",
            initial_overrides={"hard_loop_severity": 1},
        )

        self.assertEqual(self._decide(result), CT2RescueDecision(False, "no_objective_high_risk_anomaly"))

    def test_soft_loop_alone_does_not_trigger(self) -> None:
        result = self._result(
            selected_result="repair",
            repair_overrides={"fallback_exhausted": False, "soft_review_required": True},
        )

        self.assertEqual(self._decide(result), CT2RescueDecision(False, "repair_reliable"))

    def test_rejected_repair_without_structural_risk_does_not_trigger(self) -> None:
        result = self._result()

        self.assertTrue(result.diagnostics["repair_rejected"])
        self.assertEqual(self._decide(result), CT2RescueDecision(False, "no_objective_high_risk_anomaly"))

    def test_outer_vad_speech_with_empty_final_result_triggers(self) -> None:
        result = self._result(
            initial_overrides={"empty_output": True},
            repair_overrides={"empty_output": True},
            segments=[],
        )

        self.assertEqual(
            self._decide(result, outer_vad_has_speech=True),
            CT2RescueDecision(True, "voiced_empty_output"),
        )

    def test_truncated_output_triggers(self) -> None:
        result = self._result(
            initial_overrides={"output_truncated": True},
            repair_overrides={"output_truncated": True},
        )

        self.assertEqual(self._decide(result), CT2RescueDecision(True, "output_truncated"))

    def test_invalid_segment_timestamp_or_seek_nonprogress_triggers(self) -> None:
        cases = (
            ({"invalid_segment_count": 1}, "invalid_segment_structure"),
            ({"timestamp_violation_count": 1}, "invalid_timestamp_structure"),
            ({"seek_nonprogress_count": 1}, "seek_nonprogress"),
        )
        for overrides, reason in cases:
            with self.subTest(reason=reason):
                result = self._result(initial_overrides=overrides, repair_overrides=overrides)

                self.assertEqual(self._decide(result), CT2RescueDecision(True, reason))

    def test_unresolved_hard_loop_with_fallback_exhaustion_triggers(self) -> None:
        unresolved = {"fallback_exhausted": True, "hard_loop_severity": 1}
        result = self._result(initial_overrides=unresolved, repair_overrides=unresolved)

        self.assertEqual(
            self._decide(result),
            CT2RescueDecision(True, "hard_loop_with_fallback_exhausted"),
        )

    def test_production_without_fixed_t0_repair_does_not_trigger(self) -> None:
        result = self._result(repair_ran=False)

        self.assertEqual(self._decide(result), CT2RescueDecision(False, "fixed_t0_repair_not_run"))

    def test_incomplete_diagnostics_conservatively_do_not_trigger(self) -> None:
        flagged_incomplete = self._result(repair_overrides={"diagnostics_incomplete": True})
        missing_selection_evidence = self._result()
        missing_selection_evidence.diagnostics.pop("selection_reason")

        for result in (flagged_incomplete, missing_selection_evidence):
            with self.subTest(diagnostics=result.diagnostics):
                self.assertEqual(
                    self._decide(result),
                    CT2RescueDecision(False, "diagnostics_incomplete"),
                )

    def test_repeated_frame_range_and_invalid_final_timeline_trigger(self) -> None:
        repeated_range = self._result(
            initial_overrides={"repeated_frame_range_count": 1},
            repair_overrides={"repeated_frame_range_count": 1},
        )
        invalid_timeline = self._result(
            segments=[
                BackendSegment(0.5, 0.9, "第一段"),
                BackendSegment(0.2, 0.4, "第二段"),
            ]
        )

        self.assertEqual(self._decide(repeated_range), CT2RescueDecision(True, "repeated_frame_range"))
        self.assertEqual(self._decide(invalid_timeline), CT2RescueDecision(True, "invalid_final_timeline"))

    def test_hard_loop_without_fallback_exhaustion_does_not_trigger(self) -> None:
        hard_loop_only = {"fallback_exhausted": False, "hard_loop_severity": 1}
        result = self._result(initial_overrides=hard_loop_only, repair_overrides=hard_loop_only)

        self.assertEqual(self._decide(result), CT2RescueDecision(False, "no_objective_high_risk_anomaly"))

    def test_call_budget_and_single_rescue_contract_are_enforced(self) -> None:
        wrong_call_count = self._result()
        wrong_call_count.metrics["runtime_call_count"] = 1.0
        already_attempted = self._result()

        self.assertEqual(
            self._decide(wrong_call_count),
            CT2RescueDecision(False, "mlx_runtime_call_count_not_two"),
        )
        self.assertEqual(
            self._decide(already_attempted, ct2_rescue_already_attempted=True),
            CT2RescueDecision(False, "ct2_rescue_already_attempted"),
        )

    def test_selected_result_value_domain_is_enforced(self) -> None:
        result = self._result()
        result.diagnostics["selected_result"] = "ct2"

        self.assertEqual(self._decide(result), CT2RescueDecision(False, "invalid_selected_result"))


class StructuredDiagnosticEventTests(unittest.TestCase):
    @staticmethod
    def _attempt(index: int, *, accepted: bool) -> dict[str, Any]:
        return {
            "temperature": index / 10,
            "compression_ratio": 1.1 + index / 10,
            "avg_logprob": -0.2 - index / 10,
            "no_speech_prob": 0.01,
            "failure_reasons": [] if accepted else ["compression_ratio"],
            "accepted": accepted,
            "text_sha256": "a" * 64,
            "text_length": 4,
            "token_sha256": "b" * 64,
            "token_count": 200,
            "timestamp_token_sha256": "c" * 64,
            "timestamp_token_count": 80,
            "text_excerpt": "受限片段",
            "text_truncated": False,
            "token_ids": list(range(200)),
            "tokens_truncated": True,
            "timestamp_token_ids": list(range(80)),
            "timestamp_tokens_truncated": True,
        }

    @classmethod
    def _window(
        cls,
        index: int,
        *,
        attempt_count: int = 1,
        exhausted: bool = False,
    ) -> dict[str, Any]:
        return {
            "seek": index * 3_000,
            "segment_size": 3_000,
            "segment_duration": 30.0,
            "attempts": [
                cls._attempt(attempt_index, accepted=not exhausted and attempt_index == attempt_count - 1)
                for attempt_index in range(attempt_count)
            ],
            "fallback_exhausted": exhausted,
            "final_failure_reasons": ["compression_ratio"] if exhausted else [],
        }

    @classmethod
    def _diagnostics(
        cls,
        *,
        windows: list[dict[str, Any]] | None = None,
        fallback_exhausted: bool = False,
        loop_detected: bool = False,
    ) -> dict[str, Any]:
        anomaly_detected = fallback_exhausted or loop_detected
        if fallback_exhausted and loop_detected:
            anomaly_reason = "both"
        elif fallback_exhausted:
            anomaly_reason = "fallback_exhausted"
        elif loop_detected:
            anomaly_reason = "high_confidence_loop"
        else:
            anomaly_reason = None
        return {
            "sample_count": 32_000,
            "initial_clip_mode": "full_chunk",
            "initial": {
                "anomaly_detected": anomaly_detected,
                "anomaly_reason": anomaly_reason,
                "fallback_exhausted": fallback_exhausted,
                "loop_detected": loop_detected,
                "findings": (
                    [
                        {
                            "rule": "consecutive_unit",
                            "segment_index": 0,
                            "unit_length": 2,
                            "repeat_count": 3,
                            "text_length": 6,
                            "text_sha256": "a" * 64,
                        }
                    ]
                    if loop_detected
                    else []
                ),
                "runtime_diagnostics": {
                    "schema_version": 2,
                    "fallback_exhausted": fallback_exhausted,
                    "windows": windows if windows is not None else [cls._window(0)],
                },
            },
            "retry": None,
            "retry_reason": None,
            "safe_retry_count": 0,
            "selected_result": "initial",
            "unresolved_anomaly": anomaly_detected,
        }

    @staticmethod
    def _events(diagnostics: dict[str, Any], *, debug: bool) -> tuple[dict[str, Any], ...]:
        return build_mlx_diagnostic_events(
            diagnostics=diagnostics,
            metrics={"inference_seconds": 1.25, "peak_memory_bytes": 4_096.0},
            duration=2.0,
            segment_count=1,
            debug=debug,
        )

    def test_normal_mode_emits_stable_request_and_result_summaries_only(self) -> None:
        events = self._events(self._diagnostics(), debug=False)

        self.assertEqual([event["event"] for event in events], ["mlx_request_summary", "mlx_result_summary"])
        for event in events:
            self.assertTrue(set(MLX_EVENT_REQUIRED_FIELDS).issubset(event))
        self.assertEqual(events[0]["attempt_count"], 1)
        self.assertNotIn("mlx_debug_windows", {event["event"] for event in events})

    def test_normal_mode_includes_only_exhausted_window_details_for_anomaly(self) -> None:
        diagnostics = self._diagnostics(
            windows=[self._window(0), self._window(1, exhausted=True)],
            fallback_exhausted=True,
        )
        events = self._events(diagnostics, debug=False)
        anomaly = next(event for event in events if event["event"] == "mlx_runtime_anomaly")

        self.assertEqual(len(anomaly["anomalous_windows"]), 1)
        self.assertTrue(anomaly["anomalous_windows"][0]["fallback_exhausted"])
        self.assertNotIn("mlx_debug_windows", {event["event"] for event in events})

    def test_debug_mode_caps_windows_and_attempts(self) -> None:
        windows = [
            self._window(index, attempt_count=MLX_DEBUG_MAX_ATTEMPTS + 2) for index in range(MLX_DEBUG_MAX_WINDOWS + 2)
        ]
        events = self._events(self._diagnostics(windows=windows), debug=True)
        debug_event = next(event for event in events if event["event"] == "mlx_debug_windows")

        self.assertEqual(len(debug_event["windows"]), MLX_DEBUG_MAX_WINDOWS)
        self.assertTrue(debug_event["windows_truncated"])
        self.assertTrue(all(len(window["attempts"]) == MLX_DEBUG_MAX_ATTEMPTS for window in debug_event["windows"]))
        self.assertTrue(all(window["attempts_truncated"] for window in debug_event["windows"]))
        first_attempt = debug_event["windows"][0]["attempts"][0]
        self.assertEqual(len(first_attempt["token_ids"]), 128)
        self.assertEqual(len(first_attempt["timestamp_token_ids"]), 64)
        self.assertEqual(first_attempt["text_excerpt"], "受限片段")
        self.assertLessEqual(mlx_diagnostic_events_size(events), MLX_DEBUG_MAX_BYTES)

    def test_debug_sanitization_never_serializes_audio_tokens_mel_or_full_text(self) -> None:
        secret = "不应进入结构化日志的完整文本"
        window = self._window(0)
        window.update({"audio": [0.0], "mel": [[0.0]], "tokens": [1, 2], "text": secret})
        window["attempts"][0].update({"tokens": [3], "text": secret})
        diagnostics = self._diagnostics(windows=[window])
        diagnostics["initial"]["runtime_diagnostics"].update(
            {"audio": [0.0], "mel": [[0.0]], "tokens": [4], "text": secret}
        )

        serialized = "\n".join(serialize_mlx_diagnostic_event(event) for event in self._events(diagnostics, debug=True))

        self.assertNotIn(secret, serialized)
        for forbidden_key in ('"audio"', '"mel"', '"tokens"', '"text"'):
            self.assertNotIn(forbidden_key, serialized)

    def test_oversized_event_set_is_replaced_by_bounded_truncated_summary(self) -> None:
        base: dict[str, Any] = {name: 0 for name in MLX_EVENT_REQUIRED_FIELDS}
        base.update(
            {
                "schema_version": 2,
                "backend": "mlx",
                "clip_mode": "full_chunk",
                "fallback_exhausted": False,
                "loop_detected": False,
                "unresolved_anomaly": False,
            }
        )
        events = [{**base, "event": f"synthetic_{index}", "payload": "x" * 8_192} for index in range(3)]

        limited = limit_mlx_diagnostic_events(events, max_bytes=2_048)

        self.assertLessEqual(mlx_diagnostic_events_size(limited), 2_048)
        self.assertTrue(limited[-1]["truncated"])
        self.assertEqual(limited[-1]["event"], "mlx_diagnostics_truncated")
        self.assertEqual(limited[-1]["dropped_event_count"], 3)


if __name__ == "__main__":
    unittest.main()
