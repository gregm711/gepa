"""Tests for LLM judge integration."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from turbo_gepa.cache import DiskCache
from turbo_gepa.evaluator import AsyncEvaluator, JudgeFn
from turbo_gepa.interfaces import Candidate, EvalResult


# ============================================================================
# Test EvalResult diagnostic handling
# ============================================================================


def test_eval_result_diagnostic_field():
    """EvalResult should accept and store diagnostic field."""
    result = EvalResult(
        objectives={"quality": 0.8},
        traces=[],
        n_examples=1,
        diagnostic={"failure_stage": "reasoning", "suggestions": ["Be more specific"]},
    )
    assert result.diagnostic is not None
    assert result.diagnostic["failure_stage"] == "reasoning"
    assert "Be more specific" in result.diagnostic["suggestions"]


def test_eval_result_merge_combines_diagnostics():
    """Merging EvalResults should combine diagnostics."""
    result1 = EvalResult(
        objectives={"quality": 0.8},
        traces=[],
        n_examples=1,
        diagnostic={"failure_stage": "reasoning", "suggestions": ["Suggestion A"]},
    )
    result2 = EvalResult(
        objectives={"quality": 0.6},
        traces=[],
        n_examples=1,
        diagnostic={"failure_stage": "output", "suggestions": ["Suggestion B"]},
    )
    merged = result1.merge(result2)

    # Check that diagnostics are merged
    assert merged.diagnostic is not None
    # Lists should be concatenated
    assert "Suggestion A" in merged.diagnostic["suggestions"]
    assert "Suggestion B" in merged.diagnostic["suggestions"]
    # Scalars should prefer second result
    assert merged.diagnostic["failure_stage"] == "output"


def test_eval_result_merge_handles_none_diagnostics():
    """Merging should handle None diagnostics gracefully."""
    result1 = EvalResult(
        objectives={"quality": 0.8},
        traces=[],
        n_examples=1,
        diagnostic={"suggestions": ["Test"]},
    )
    result2 = EvalResult(
        objectives={"quality": 0.6},
        traces=[],
        n_examples=1,
        diagnostic=None,
    )
    merged = result1.merge(result2)
    assert merged.diagnostic is not None
    assert merged.diagnostic["suggestions"] == ["Test"]


# ============================================================================
# Test AsyncEvaluator judge integration
# ============================================================================


@pytest.fixture
def mock_cache():
    """Create a mock cache that returns no cached results."""
    cache = MagicMock(spec=DiskCache)
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    return cache


@pytest.fixture
def simple_task_runner():
    """Task runner that returns quality based on example_id."""
    async def runner(candidate: Candidate, example_id: str) -> dict[str, Any]:
        # Simulate some examples passing, some failing
        quality = 0.9 if example_id.startswith("pass") else 0.3
        return {
            "quality": quality,
            "output": f"Output for {example_id}",
            "expected_answer": "expected",
            "input": f"Input for {example_id}",
        }
    return runner


@pytest.fixture
def simple_judge_fn():
    """Judge function that returns diagnostic based on quality."""
    async def judge(
        output: str,
        expected: str | None,
        example: dict[str, Any],
        candidate: Candidate,
    ) -> dict[str, Any] | None:
        # Simple judge: detect failure based on output
        if "pass" in (example.get("example_id") or ""):
            return {
                "quality": 0.9,
                "failure_stage": "none",
                "failure_explanation": None,
                "suggestions": [],
            }
        else:
            return {
                "quality": 0.3,
                "failure_stage": "reasoning",
                "failure_explanation": "Failed to reason correctly",
                "suggestions": ["Add step-by-step reasoning"],
            }
    return judge


@pytest.mark.asyncio
async def test_evaluator_without_judge(mock_cache, simple_task_runner):
    """Evaluator should work without judge configured (backward compatible)."""
    evaluator = AsyncEvaluator(
        cache=mock_cache,
        task_runner=simple_task_runner,
        judge_fn=None,  # No judge
    )

    candidate = Candidate(text="Test prompt")
    result = await evaluator.eval_on_shard(
        candidate=candidate,
        example_ids=["pass_1", "fail_1"],
        concurrency=2,
    )

    # Should return result without diagnostics
    assert result.n_examples == 2
    assert result.diagnostic is None
    # Traces should not have diagnostic field
    for trace in result.traces:
        assert "diagnostic" not in trace


@pytest.mark.asyncio
async def test_evaluator_with_judge(mock_cache, simple_task_runner, simple_judge_fn):
    """Evaluator should run judge and attach diagnostics."""
    evaluator = AsyncEvaluator(
        cache=mock_cache,
        task_runner=simple_task_runner,
        judge_fn=simple_judge_fn,
        judge_sample_rate=1.0,  # Judge all traces
        judge_on_fail_only=False,
    )

    candidate = Candidate(text="Test prompt")
    result = await evaluator.eval_on_shard(
        candidate=candidate,
        example_ids=["pass_1", "fail_1"],
        concurrency=2,
    )

    # Should have diagnostics
    assert result.n_examples == 2
    assert result.diagnostic is not None
    assert result.diagnostic["judged_count"] == 2

    # At least one trace should have diagnostic attached
    diagnostics_found = [t.get("diagnostic") for t in result.traces if t.get("diagnostic")]
    assert len(diagnostics_found) == 2


@pytest.mark.asyncio
async def test_evaluator_judge_on_fail_only(mock_cache, simple_task_runner, simple_judge_fn):
    """Evaluator should only judge failures when judge_on_fail_only=True."""
    evaluator = AsyncEvaluator(
        cache=mock_cache,
        task_runner=simple_task_runner,
        judge_fn=simple_judge_fn,
        judge_sample_rate=1.0,
        judge_on_fail_only=True,
        judge_fail_threshold=0.5,  # Quality below 0.5 = failure
    )

    candidate = Candidate(text="Test prompt")
    result = await evaluator.eval_on_shard(
        candidate=candidate,
        example_ids=["pass_1", "pass_2", "fail_1"],
        concurrency=3,
    )

    # Should have judged only the failing example
    assert result.diagnostic is not None
    assert result.diagnostic["judged_count"] == 1
    assert result.diagnostic["total_traces"] == 3


@pytest.mark.asyncio
async def test_evaluator_judge_sample_rate(mock_cache, simple_task_runner, simple_judge_fn):
    """Evaluator should respect sample rate."""
    evaluator = AsyncEvaluator(
        cache=mock_cache,
        task_runner=simple_task_runner,
        judge_fn=simple_judge_fn,
        judge_sample_rate=0.0,  # Judge nothing
        judge_on_fail_only=False,
    )

    candidate = Candidate(text="Test prompt")
    result = await evaluator.eval_on_shard(
        candidate=candidate,
        example_ids=["pass_1", "fail_1"],
        concurrency=2,
    )

    # Should have no diagnostics with 0% sample rate
    assert result.diagnostic is None


@pytest.mark.asyncio
async def test_evaluator_judge_failure_aggregation(mock_cache, simple_task_runner, simple_judge_fn):
    """Evaluator should aggregate failure stages from judge."""
    evaluator = AsyncEvaluator(
        cache=mock_cache,
        task_runner=simple_task_runner,
        judge_fn=simple_judge_fn,
        judge_sample_rate=1.0,
        judge_on_fail_only=False,
    )

    candidate = Candidate(text="Test prompt")
    # Multiple failures
    result = await evaluator.eval_on_shard(
        candidate=candidate,
        example_ids=["fail_1", "fail_2", "pass_1"],
        concurrency=3,
    )

    assert result.diagnostic is not None
    # Should have aggregated failure stages
    assert "failure_stages" in result.diagnostic
    # "reasoning" failures from our mock judge
    if result.diagnostic["failure_stages"]:
        assert "reasoning" in result.diagnostic["failure_stages"]


# ============================================================================
# Test LLMJudgeEvaluator
# ============================================================================


def test_llm_judge_config():
    """LLMJudgeConfig should store configuration."""
    from turbo_gepa.evaluators.llm_judge import LLMJudgeConfig

    config = LLMJudgeConfig(
        model="gpt-4o-mini",
        task_type="qa",
        temperature=0.0,
    )
    assert config.model == "gpt-4o-mini"
    assert config.task_type == "qa"


def test_llm_judge_template_resolution():
    """LLMJudgeEvaluator should resolve templates by task_type."""
    from turbo_gepa.evaluators.llm_judge import (
        LLMJudgeConfig,
        LLMJudgeEvaluator,
        QA_EVAL_TEMPLATE,
        RAG_EVAL_TEMPLATE,
    )

    qa_config = LLMJudgeConfig(model="gpt-4o-mini", task_type="qa")
    qa_evaluator = LLMJudgeEvaluator(qa_config)
    assert qa_evaluator._template == QA_EVAL_TEMPLATE

    rag_config = LLMJudgeConfig(model="gpt-4o-mini", task_type="rag")
    rag_evaluator = LLMJudgeEvaluator(rag_config)
    assert rag_evaluator._template == RAG_EVAL_TEMPLATE


def test_llm_judge_custom_template():
    """LLMJudgeEvaluator should use custom template if provided."""
    from turbo_gepa.evaluators.llm_judge import LLMJudgeConfig, LLMJudgeEvaluator

    custom = "My custom template {input} {expected} {output} {system_prompt}"
    config = LLMJudgeConfig(model="gpt-4o-mini", prompt_template=custom)
    evaluator = LLMJudgeEvaluator(config)
    assert evaluator._template == custom


def test_llm_judge_parse_response():
    """LLMJudgeEvaluator should parse JSON responses."""
    from turbo_gepa.evaluators.llm_judge import LLMJudgeConfig, LLMJudgeEvaluator

    config = LLMJudgeConfig(model="gpt-4o-mini")
    evaluator = LLMJudgeEvaluator(config)

    # Valid JSON
    result = evaluator._parse_response('{"quality": 0.8, "failure_stage": "none"}')
    assert result is not None
    assert result["quality"] == 0.8

    # JSON in markdown code block (common from LLMs)
    result = evaluator._parse_response('```json\n{"quality": 0.5}\n```')
    assert result is not None
    assert result["quality"] == 0.5

    # JSON in plain markdown fence
    result = evaluator._parse_response('```\n{"quality": 0.6}\n```')
    assert result is not None
    assert result["quality"] == 0.6

    # Nested objects (suggestions array)
    nested = '{"quality": 0.7, "suggestions": ["Add X", "Remove Y"], "nested": {"key": "val"}}'
    result = evaluator._parse_response(nested)
    assert result is not None
    assert result["quality"] == 0.7
    assert result["suggestions"] == ["Add X", "Remove Y"]
    assert result["nested"]["key"] == "val"

    # JSON with preamble text
    result = evaluator._parse_response('Here is my evaluation:\n{"quality": 0.4}')
    assert result is not None
    assert result["quality"] == 0.4

    # Invalid JSON
    result = evaluator._parse_response("not json at all")
    assert result is None

    # Empty
    result = evaluator._parse_response("")
    assert result is None

    # Unclosed brace
    result = evaluator._parse_response('{"quality": 0.5')
    assert result is None


# ============================================================================
# Test evaluator_feedback_reflection strategy
# ============================================================================


def test_evaluator_feedback_strategy_registered():
    """evaluator_feedback_reflection should be available (opt-in, not default)."""
    from turbo_gepa.strategies import (
        available_reflection_strategy_names,
        default_reflection_strategies,
        resolve_reflection_strategy_names,
        get_evaluator_feedback_strategy,
    )

    # Should be in available names
    names = available_reflection_strategy_names()
    assert "evaluator_feedback_reflection" in names

    # Should NOT be in defaults (opt-in only)
    defaults = default_reflection_strategies()
    default_names = [s.name for s in defaults]
    assert "evaluator_feedback_reflection" not in default_names

    # Should be resolvable when explicitly requested
    strategies = resolve_reflection_strategy_names(["evaluator_feedback_reflection"])
    assert len(strategies) == 1
    assert strategies[0].name == "evaluator_feedback_reflection"

    # Should be accessible via get_evaluator_feedback_strategy()
    strategy = get_evaluator_feedback_strategy()
    assert strategy.name == "evaluator_feedback_reflection"


def test_evaluator_feedback_prompt_with_diagnostics():
    """evaluator_feedback_reflection should format diagnostics in prompt."""
    from turbo_gepa.strategies import build_evaluator_feedback_prompt

    parent_contexts = [
        {
            "candidate": Candidate(text="Test prompt"),
            "diagnostics": [
                {
                    "failure_stage": "reasoning",
                    "failure_explanation": "Logic was flawed",
                    "suggestions": ["Add step-by-step reasoning"],
                },
                {
                    "failure_stage": "reasoning",
                    "suggestions": ["Be more explicit"],
                },
            ],
            "traces": [],
        }
    ]

    prompt = build_evaluator_feedback_prompt(
        parent_contexts=parent_contexts,
        reflection_examples=[],
        _task_examples=[],
        num_mutations=3,
    )

    # Should include failure analysis
    assert "reasoning" in prompt.lower()
    assert "failure" in prompt.lower()
    # Should include suggestions
    assert "step-by-step" in prompt.lower() or "explicit" in prompt.lower()


def test_evaluator_feedback_prompt_without_diagnostics():
    """evaluator_feedback_reflection should handle empty diagnostics gracefully."""
    from turbo_gepa.strategies import build_evaluator_feedback_prompt

    parent_contexts = [
        {
            "candidate": Candidate(text="Test prompt"),
            "diagnostics": [],
            "traces": [],
        }
    ]

    prompt = build_evaluator_feedback_prompt(
        parent_contexts=parent_contexts,
        reflection_examples=[],
        _task_examples=[],
        num_mutations=3,
    )

    # Should still produce a valid prompt
    assert "Generate improved prompts" in prompt or "improved prompts" in prompt.lower()
    assert "No specific failure stages" in prompt or "failure" in prompt.lower()


def test_evaluator_feedback_extracts_from_traces():
    """evaluator_feedback_reflection should extract diagnostics from traces."""
    from turbo_gepa.strategies import _extract_diagnostics_from_context

    parent_contexts = [
        {
            "traces": [
                {"diagnostic": {"failure_stage": "understanding", "suggestions": ["S1"]}},
                {"diagnostic": {"failure_stage": "reasoning", "suggestions": ["S2"]}},
                {"no_diagnostic": True},
            ],
        }
    ]

    failure_stages, suggestions, sample_diagnostics = _extract_diagnostics_from_context(
        parent_contexts
    )

    assert "understanding" in failure_stages
    assert "reasoning" in failure_stages
    assert "S1" in suggestions
    assert "S2" in suggestions
    assert len(sample_diagnostics) == 2
