"""
Strategy definitions for reflection/spec-induction prompts.

Each strategy controls:
    - The system prompt instructions sent to the reflection LLM
    - How we build the user message (using parent contexts/examples)
    - How to parse the raw response into candidate prompt strings
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

StrategyPromptBuilder = Callable[
    [Sequence[dict[str, Any]], Sequence[dict[str, Any]], Sequence[dict[str, Any]], int],
    str,
]
StrategyResponseParser = Callable[[str], list[str]]


@dataclass(slots=True)
class ReflectionStrategy:
    """Configuration for a reflection/spec-induction strategy."""

    name: str
    system_prompt: str
    user_prompt_builder: StrategyPromptBuilder
    response_parser: StrategyResponseParser
    requires_examples: bool = False  # True if builder expects task examples


def _format_parent_summaries(parent_contexts: Sequence[dict[str, Any]]) -> str:
    """Render parent prompt summaries similar to the original reflection runner."""

    summaries: list[str] = []
    for i, ctx in enumerate(parent_contexts[:5]):  # Limit for token safety
        prompt_text = ctx.get("prompt", "")
        meta = ctx.get("meta", {}) or {}
        traces = ctx.get("traces", []) or []
        diagnostics = ctx.get("diagnostics", []) or []

        objective_key = meta.get("objective_key", "quality")
        parent_objectives = meta.get("parent_objectives")
        if isinstance(parent_objectives, dict):
            quality = parent_objectives.get(objective_key)
            if quality is None:
                quality = parent_objectives.get("quality", 0.0)
        else:
            quality = meta.get(objective_key, meta.get("quality", 0.0))

        shard_fraction = meta.get("quality_shard_fraction", 0.0)
        shard_info = f", shard={shard_fraction * 100:.0f}%" if shard_fraction else ""
        avg_quality = 0.0
        if traces:
            values = [t.get(objective_key, t.get("quality", 0.0)) for t in traces[:3]]
            if values:
                avg_quality = sum(values) / len(values)

        label = "Quality"
        if objective_key != "quality":
            label = objective_key
        perf_summary = f"Recent avg: {avg_quality:.1%}" if traces else f"{label}: {quality:.1%}"

        diag_entries: list[dict[str, Any]] = []
        if isinstance(diagnostics, list):
            diag_entries.extend([d for d in diagnostics if isinstance(d, dict)])
        for trace in traces[:5]:
            diag = trace.get("diagnostic") if isinstance(trace, dict) else None
            if isinstance(diag, dict):
                diag_entries.append(diag)

        diag_lines: list[str] = []
        if diag_entries:
            stage_counts: dict[str, int] = {}
            suggestions: list[str] = []
            for diag in diag_entries:
                stage = diag.get("failure_stage")
                if isinstance(stage, str) and stage and stage != "none":
                    stage_counts[stage] = stage_counts.get(stage, 0) + 1
                sugg = diag.get("suggestions")
                if isinstance(sugg, list):
                    for s in sugg:
                        if isinstance(s, str) and s.strip():
                            suggestions.append(s.strip())
            if stage_counts:
                stage_summary = ", ".join(f"{k}:{v}" for k, v in sorted(stage_counts.items(), key=lambda x: -x[1]))
                diag_lines.append(f"Diagnostics: stages={stage_summary}")
            if suggestions:
                unique_suggestions: list[str] = []
                seen: set[str] = set()
                for s in suggestions:
                    key = s.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    unique_suggestions.append(s)
                diag_lines.append(f"Suggestions: {'; '.join(unique_suggestions[:3])}")

        block = f"""PROMPT {chr(65 + i)} ({perf_summary}{shard_info}):
"{prompt_text.strip()}"
"""
        if diag_lines:
            block += "\n".join(diag_lines) + "\n"
        summaries.append(block)

    return "\n".join(summaries)


def _format_reflection_examples(reflection_examples: Sequence[dict[str, Any]]) -> str:
    """Summarize previous reflections/examples for few-shot guidance."""

    if not reflection_examples:
        return ""
    summaries: list[str] = []
    for idx, ex in enumerate(reflection_examples[:5]):
        example_input = ex.get("input", "").strip()
        answer = (ex.get("expected_answer") or ex.get("answer") or "").strip()
        assistant_output = ex.get("assistant_output", "").strip()
        feedback = ex.get("feedback", "").strip()
        additional = ex.get("additional_context") or {}
        solution = ""
        if isinstance(additional, dict):
            solution = (additional.get("solution") or additional.get("Solution") or "").strip()

        block = [f"Example {idx + 1} Input: {example_input}"]
        if answer:
            block.append(f"Expected answer: {answer}")
        if assistant_output:
            block.append(f"Assistant output: {assistant_output}")
        if feedback:
            block.append(f"Feedback: {feedback}")
        if solution:
            block.append(f"Solution: {solution}")
        summaries.append("\n".join(block))

    return "\n\n".join(summaries)


def build_incremental_reflection_prompt(
    parent_contexts: Sequence[dict[str, object]],
    reflection_examples: Sequence[dict[str, object]],
    _task_examples: Sequence[dict[str, object]],
    num_mutations: int,
) -> str:
    """Default user message for incremental reflection mutations."""

    parent_section = _format_parent_summaries(parent_contexts)
    examples_section = _format_reflection_examples(reflection_examples)
    instruction = f"""You are TurboGEPA's reflection model. Generate {num_mutations} high-quality prompt mutations.
- Analyze the successful parent prompts and their quality metrics.
- Blend the strongest instructions while keeping structured formatting.
- Each mutation MUST be wrapped inside <PROMPT>...</PROMPT> tags.
- Avoid copying answers (e.g., "### 242") or giving final answers yourself.
- Ensure each prompt ends with guidance to answer using '### <final answer>'.\n"""

    sections = [instruction]
    if parent_section:
        sections.append("=== PARENT PROMPT SUMMARIES ===")
        sections.append(parent_section)
    if examples_section:
        sections.append("=== TASK EXAMPLES & SOLUTIONS ===")
        sections.append(examples_section)

    sections.append("Generate improved prompts now.")
    return "\n".join(sections)


def _format_task_examples(task_examples: Sequence[dict[str, object]]) -> str:
    formatted: list[str] = []
    for i, ex in enumerate(task_examples, 1):
        input_text = ex.get("input", "")
        answer = ex.get("answer", "")
        additional = ex.get("additional_context") or {}
        block = [f"Example {i}:", f"Input: {input_text}", f"Expected Output: {answer}"]
        if isinstance(additional, dict):
            for key, value in additional.items():
                block.append(f"{key.title()}: {value}")
        formatted.append("\n".join(block))
    return "\n\n".join(formatted)


def build_spec_induction_prompt(
    parent_contexts: Sequence[dict[str, object]],
    _reflection_examples: Sequence[dict[str, object]],
    task_examples: Sequence[dict[str, object]],
    num_mutations: int,
) -> str:
    """PROMPT-MII style spec induction prompt."""

    examples_section = _format_task_examples(task_examples[:3])
    parent_section = _format_parent_summaries(parent_contexts)
    instruction = f"""You are designing new instruction prompts for the same task shown below.
Generate {num_mutations} distinct instruction variants that would help an AI assistant solve NEW problems in this domain.
Each instruction MUST be wrapped in <PROMPT>...</PROMPT> tags.
Focus on clarity, domain-specific guidance, and enforcing the '### <final answer>' output format.
"""
    sections = [instruction]
    sections.append("=== TASK EXAMPLES ===")
    sections.append(examples_section)
    if parent_section:
        sections.append("\n=== CURRENT HIGH-PERFORMING PROMPTS ===")
        sections.append(parent_section)
    sections.append("\nWrite the new instructions now.")
    return "\n".join(sections)


def parse_prompts_from_tags(content: str) -> list[str]:
    """Extract prompt blocks from LLM output."""

    matches = re.findall(r"<PROMPT>\s*(.*?)\s*</PROMPT>", content, re.DOTALL | re.IGNORECASE)
    if matches:
        return [m.strip() for m in matches if m.strip()]

    # Fallback: split on line breaks or --- separators
    fallback = [segment.strip() for segment in content.split("---") if segment.strip()]
    if fallback:
        return fallback

    return [content.strip()] if content.strip() else []


BASE_REFLECTION_SYSTEM_PROMPT = (
    "You are a prompt-evolution strategist. Given successful parent prompts and evaluation traces, "
    "generate improved instructions that stay faithful to the task requirements."
)

SPEC_INDUCTION_SYSTEM_PROMPT = (
    "You are a specification engineer. Study the provided task examples and craft new instructions "
    "that would let a model solve similar problems."
)

INTERLEAVED_THINKING_SYSTEM_PROMPT = (
    "You are a cognitive coach specializing in interleaved reasoning. "
    "Rewrite or enhance task instructions so the student alternates between private <think> steps and public "
    "<answer> outputs, each covering one verifiable chunk of reasoning. Preserve the original task intent and "
    "constraints while guiding the student to finish with a single final <answer> that contains only the final solution."
)


def default_reflection_strategies() -> tuple[ReflectionStrategy, ...]:
    """Return the built-in strategy list (incremental reflection + spec induction + interleaved thinking).

    Note: evaluator_feedback_reflection is NOT included by default. Enable it explicitly
    via config.reflection_strategy_names when using judge_fn for diagnostic feedback.
    """

    return (
        ReflectionStrategy(
            name="incremental_reflection",
            system_prompt=BASE_REFLECTION_SYSTEM_PROMPT,
            user_prompt_builder=build_incremental_reflection_prompt,
            response_parser=parse_prompts_from_tags,
            requires_examples=False,
        ),
        ReflectionStrategy(
            name="spec_induction",
            system_prompt=SPEC_INDUCTION_SYSTEM_PROMPT,
            user_prompt_builder=build_spec_induction_prompt,
            response_parser=parse_prompts_from_tags,
            requires_examples=True,
        ),
        ReflectionStrategy(
            name="interleaved_thinking",
            system_prompt=INTERLEAVED_THINKING_SYSTEM_PROMPT,
            user_prompt_builder=build_interleaved_prompt,
            response_parser=parse_prompts_from_tags,
            requires_examples=False,
        ),
    )


def get_evaluator_feedback_strategy() -> ReflectionStrategy:
    """Return the evaluator_feedback_reflection strategy (opt-in, not in defaults).

    Use this when judge_fn is enabled to leverage diagnostic feedback for reflection.
    Add to config.reflection_strategies or use resolve_reflection_strategy_names().
    """
    return ReflectionStrategy(
        name="evaluator_feedback_reflection",
        system_prompt=EVALUATOR_FEEDBACK_SYSTEM_PROMPT,
        user_prompt_builder=build_evaluator_feedback_prompt,
        response_parser=parse_prompts_from_tags,
        requires_examples=False,
    )


def available_reflection_strategy_names() -> tuple[str, ...]:
    """Return all available strategy names (defaults + opt-in strategies)."""
    defaults = tuple(strategy.name for strategy in default_reflection_strategies())
    # Include opt-in strategies
    return defaults + ("evaluator_feedback_reflection",)


def resolve_reflection_strategy_names(names: Sequence[str] | None) -> tuple[ReflectionStrategy, ...]:
    """
    Resolve a list of strategy names to ReflectionStrategy objects.

    Args:
        names: Iterable of strategy names (order preserved). If None, returns all defaults.
               Pass an empty tuple/list to disable built-ins (use custom strategies only).
               Include "evaluator_feedback_reflection" to enable diagnostic-aware reflection.
    """

    if names is None:
        return default_reflection_strategies()

    # Build registry including opt-in strategies
    registry = {strategy.name: strategy for strategy in default_reflection_strategies()}
    registry["evaluator_feedback_reflection"] = get_evaluator_feedback_strategy()

    resolved: list[ReflectionStrategy] = []
    for name in names:
        strategy = registry.get(name)
        if strategy is None:
            available = ", ".join(registry)
            raise ValueError(f"Unknown reflection strategy '{name}'. Available strategies: {available}")
        resolved.append(strategy)
    return tuple(resolved)


def build_interleaved_prompt(
    parent_contexts: Sequence[dict[str, object]],
    reflection_examples: Sequence[dict[str, object]],
    _task_examples: Sequence[dict[str, object]],
    num_mutations: int,
) -> str:
    """Prompt that enforces interleaved <think>/<answer> reasoning."""

    parent_section = _format_parent_summaries(parent_contexts)
    examples_section = _format_reflection_examples(reflection_examples)
    instruction = f"""You are improving prompts to enforce interleaved reasoning.
Generate {num_mutations} new prompt variants that explicitly teach the student to alternate between <think> (private) and <answer> (public) blocks.
Rules:
- Each <think> block must handle one short reasoning step and stay hidden.
- Each <answer> block must summarize only that step's conclusion for the user.
- The process continues step-by-step until a final <answer> presents ONLY the final solution.
- Keep the original task intent, formatting requirements, and answer style intact.
- Ensure the rewritten prompt clearly explains this alternating pattern and how to end with the final answer.
Wrap every candidate prompt in <PROMPT>...</PROMPT> tags."""

    sections = [instruction]
    if parent_section:
        sections.append("\n=== CURRENT PROMPT SUMMARIES ===")
        sections.append(parent_section)
    if examples_section:
        sections.append("\n=== RECENT TASK TRACES ===")
        sections.append(examples_section)
    sections.append("\nGenerate the improved prompts now.")
    return "\n".join(sections)


# ============================================================================
# EVALUATOR FEEDBACK REFLECTION STRATEGY
# ============================================================================

EVALUATOR_FEEDBACK_SYSTEM_PROMPT = (
    "You are an expert prompt engineer analyzing structured evaluation feedback from an LLM judge. "
    "Your goal is to generate improved prompts that specifically address the identified failure modes. "
    "Focus on the failure stages and concrete suggestions provided by the evaluator."
)


def _extract_diagnostics_from_context(
    parent_contexts: Sequence[dict[str, object]],
) -> tuple[dict[str, int], list[str], list[dict[str, object]]]:
    """
    Extract diagnostic information from parent contexts.

    Returns:
        (failure_stage_counts, all_suggestions, sample_diagnostics)
    """
    failure_stages: dict[str, int] = {}
    all_suggestions: list[str] = []
    sample_diagnostics: list[dict[str, object]] = []

    for ctx in parent_contexts:
        # Check for diagnostics in the context itself
        diagnostics_list = ctx.get("diagnostics", [])
        if isinstance(diagnostics_list, list):
            for diag in diagnostics_list[:5]:  # Limit per parent
                if not isinstance(diag, dict):
                    continue
                sample_diagnostics.append(diag)
                stage = diag.get("failure_stage")
                if stage and stage != "none":
                    failure_stages[stage] = failure_stages.get(stage, 0) + 1
                suggestions = diag.get("suggestions")
                if isinstance(suggestions, list):
                    all_suggestions.extend(s for s in suggestions if isinstance(s, str))

        # Also check traces for per-example diagnostics
        traces = ctx.get("traces", [])
        if isinstance(traces, list):
            for trace in traces[:3]:  # Limit per parent
                if not isinstance(trace, dict):
                    continue
                diag = trace.get("diagnostic")
                if isinstance(diag, dict):
                    sample_diagnostics.append(diag)
                    stage = diag.get("failure_stage")
                    if stage and stage != "none":
                        failure_stages[stage] = failure_stages.get(stage, 0) + 1
                    suggestions = diag.get("suggestions")
                    if isinstance(suggestions, list):
                        all_suggestions.extend(s for s in suggestions if isinstance(s, str))

    # Deduplicate suggestions while preserving order
    seen: set[str] = set()
    unique_suggestions: list[str] = []
    for s in all_suggestions:
        s_lower = s.lower().strip()
        if s_lower not in seen:
            seen.add(s_lower)
            unique_suggestions.append(s)

    return failure_stages, unique_suggestions[:15], sample_diagnostics[:10]


def _format_failure_analysis(failure_stages: dict[str, int]) -> str:
    """Format failure stage analysis as readable text."""
    if not failure_stages:
        return "No specific failure stages identified."

    total = sum(failure_stages.values())
    lines = ["Failure distribution across evaluated traces:"]
    for stage, count in sorted(failure_stages.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total else 0
        lines.append(f"  - {stage}: {count} occurrences ({pct:.0f}%)")

    most_common = max(failure_stages, key=failure_stages.get)  # type: ignore
    lines.append(f"\nPrimary failure mode: {most_common}")
    return "\n".join(lines)


def _format_evaluator_suggestions(suggestions: list[str]) -> str:
    """Format evaluator suggestions as a bulleted list."""
    if not suggestions:
        return "No specific suggestions from evaluator."

    lines = ["Improvement suggestions from evaluator:"]
    for i, suggestion in enumerate(suggestions[:10], 1):
        lines.append(f"  {i}. {suggestion}")
    return "\n".join(lines)


def _format_sample_diagnostics(diagnostics: list[dict[str, object]]) -> str:
    """Format sample diagnostic entries for context."""
    if not diagnostics:
        return ""

    lines = ["Sample diagnostic details:"]
    for i, diag in enumerate(diagnostics[:5], 1):
        stage = diag.get("failure_stage", "unknown")
        explanation = diag.get("failure_explanation", "")
        quality = diag.get("quality")
        quality_str = f" (quality={quality:.2f})" if isinstance(quality, (int, float)) else ""
        lines.append(f"  {i}. Stage: {stage}{quality_str}")
        if explanation:
            # Truncate long explanations
            exp_text = str(explanation)[:200]
            if len(str(explanation)) > 200:
                exp_text += "..."
            lines.append(f"     Explanation: {exp_text}")
    return "\n".join(lines)


def build_evaluator_feedback_prompt(
    parent_contexts: Sequence[dict[str, object]],
    reflection_examples: Sequence[dict[str, object]],
    _task_examples: Sequence[dict[str, object]],
    num_mutations: int,
) -> str:
    """
    Build reflection prompt that leverages LLM judge diagnostic feedback.

    This strategy extracts failure stages and suggestions from diagnostic
    information attached to traces, then asks the reflection model to
    generate prompts that specifically address these failure modes.
    """
    parent_section = _format_parent_summaries(parent_contexts)

    # Extract diagnostic information
    failure_stages, suggestions, sample_diagnostics = _extract_diagnostics_from_context(parent_contexts)

    # Build specialized sections
    failure_analysis = _format_failure_analysis(failure_stages)
    suggestions_section = _format_evaluator_suggestions(suggestions)
    diagnostics_section = _format_sample_diagnostics(sample_diagnostics)

    instruction = f"""You are improving prompts based on structured feedback from an LLM evaluator.

The evaluator analyzed system outputs and identified specific failure modes.
Your task is to generate {num_mutations} improved prompts that directly address these failures.

=== DIAGNOSTIC ANALYSIS ===
{failure_analysis}

=== EVALUATOR SUGGESTIONS ===
{suggestions_section}
"""

    sections = [instruction]

    if diagnostics_section:
        sections.append(f"\n{diagnostics_section}")

    if parent_section:
        sections.append("\n=== CURRENT PROMPT SUMMARIES ===")
        sections.append(parent_section)

    sections.append("""
=== YOUR TASK ===
Generate improved prompts that:
1. Directly address the identified failure modes
2. Incorporate the evaluator's suggestions where applicable
3. Maintain the core task requirements and output format
4. Add specific guidance to prevent the most common failure stage

Wrap each improved prompt in <PROMPT>...</PROMPT> tags.
Generate the improved prompts now.""")

    return "\n".join(sections)
