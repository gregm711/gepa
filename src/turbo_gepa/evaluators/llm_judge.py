"""
LLM-as-Judge evaluator for rich diagnostic feedback.

This module provides an opt-in LLM judge that produces structured diagnostic
feedback for reflection strategies. The judge runs in batch after shard completion
and never affects the promotion scalar used by the scheduler.

Usage:
    from turbo_gepa.evaluators.llm_judge import LLMJudgeConfig, LLMJudgeEvaluator

    config = LLMJudgeConfig(model="gpt-4o-mini", task_type="qa")
    evaluator = LLMJudgeEvaluator(config)

    # Use as judge_fn in DefaultAdapter
    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm="gpt-4o",
        reflection_lm="gpt-4o",
        judge_fn=evaluator.evaluate,
        judge_sample_rate=0.2,  # Only judge 20% of traces
        judge_on_fail_only=True,  # Only judge failures
    )
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from turbo_gepa.interfaces import Candidate

# ============================================================================
# EVALUATION TEMPLATES
# ============================================================================

QA_EVAL_TEMPLATE = """You are evaluating a question-answering system.

=== INPUT QUESTION ===
{input}

=== EXPECTED ANSWER ===
{expected}

=== SYSTEM OUTPUT ===
{output}

=== EVALUATION TASK ===

Analyze the system output and provide structured feedback:

1. **Correctness** (0.0-1.0): Does the output correctly answer the question?
2. **Failure Stage**: If incorrect, which stage failed?
   - "understanding" - misunderstood the question
   - "reasoning" - faulty logic or incorrect facts
   - "output_formatting" - correct answer but wrong format
   - "none" - fully correct
3. **Failure Explanation**: Brief explanation of why this stage failed (if applicable)
4. **Suggestions**: 1-3 specific prompt improvements to address this failure

Respond ONLY with valid JSON (no markdown, no extra text):
{{"quality": 0.0, "failure_stage": "...", "failure_explanation": "...", "suggestions": ["..."]}}
"""

RAG_EVAL_TEMPLATE = """You are evaluating a multi-step reasoning/retrieval system.

=== INPUT ===
{input}

=== EXPECTED OUTPUT ===
{expected}

=== SYSTEM OUTPUT ===
{output}

=== SYSTEM PROMPT USED ===
{system_prompt}

=== EVALUATION TASK ===

Analyze the reasoning chain and provide structured feedback:

1. **Correctness** (0.0-1.0): Does the output match expected?
2. **Reasoning Quality** (0.0-1.0): Is the logic sound and well-supported?
3. **Failure Stage**: If incorrect, which stage failed?
   - "input_parsing" - misunderstood the question
   - "retrieval" - wrong or insufficient information gathered
   - "reasoning" - faulty logic in reasoning steps
   - "synthesis" - failed to combine information correctly
   - "output_formatting" - correct answer but wrong format
   - "none" - fully correct
4. **Failure Explanation**: Why did this stage fail?
5. **Suggestions**: 1-3 specific prompt improvements

Respond ONLY with valid JSON (no markdown, no extra text):
{{"quality": 0.0, "reasoning_quality": 0.0, "failure_stage": "...",
"failure_explanation": "...", "suggestions": ["..."]}}
"""

CODING_EVAL_TEMPLATE = """You are evaluating a code generation system.

=== TASK ===
{input}

=== EXPECTED OUTPUT ===
{expected}

=== SYSTEM OUTPUT ===
{output}

=== EVALUATION TASK ===

Rate the code generation:

1. **Correctness** (0.0-1.0): Does it solve the problem correctly?
2. **Failure Stage**: If incorrect, which stage failed?
   - "understanding" - misunderstood the task requirements
   - "approach" - chose wrong algorithm or data structure
   - "implementation" - bugs in the code logic
   - "edge_cases" - fails on edge cases
   - "output_formatting" - correct solution but wrong format
   - "none" - fully correct
3. **Failure Explanation**: What specifically went wrong?
4. **Suggestions**: 1-3 specific prompt improvements

Respond ONLY with valid JSON (no markdown, no extra text):
{{"quality": 0.0, "failure_stage": "...", "failure_explanation": "...", "suggestions": ["..."]}}
"""

GENERIC_EVAL_TEMPLATE = """You are evaluating an AI system's output.

=== INPUT ===
{input}

=== EXPECTED OUTPUT ===
{expected}

=== SYSTEM OUTPUT ===
{output}

=== EVALUATION TASK ===

Provide structured feedback:

1. **Quality** (0.0-1.0): How well does the output meet expectations?
2. **Failure Stage**: If suboptimal, what went wrong?
   - "understanding" - misunderstood the task
   - "reasoning" - flawed logic
   - "execution" - correct approach but poor execution
   - "formatting" - correct content but wrong format
   - "none" - fully correct
3. **Failure Explanation**: Brief explanation
4. **Suggestions**: 1-3 specific improvements

Respond ONLY with valid JSON (no markdown, no extra text):
{{"quality": 0.0, "failure_stage": "...", "failure_explanation": "...", "suggestions": ["..."]}}
"""

# Template registry
_TEMPLATES = {
    "qa": QA_EVAL_TEMPLATE,
    "rag": RAG_EVAL_TEMPLATE,
    "multi_hop": RAG_EVAL_TEMPLATE,  # Alias
    "coding": CODING_EVAL_TEMPLATE,
    "code": CODING_EVAL_TEMPLATE,  # Alias
    "generic": GENERIC_EVAL_TEMPLATE,
}


@dataclass
class LLMJudgeConfig:
    """Configuration for the LLM judge evaluator."""

    model: str  # e.g., "gpt-4o-mini", "claude-3-haiku-20240307"
    prompt_template: str | None = None  # Custom template (overrides task_type)
    task_type: str | None = None  # "qa", "rag", "coding", "generic"
    temperature: float = 0.0  # Deterministic by default
    max_tokens: int = 1024
    max_output_chars: int = 2000  # Truncate long outputs to save tokens
    max_input_chars: int = 2000  # Truncate long inputs


class LLMJudgeEvaluator:
    """LLM-as-judge evaluator that produces structured diagnostic feedback."""

    def __init__(self, config: LLMJudgeConfig) -> None:
        self.config = config
        self._template = self._resolve_template()

    def _resolve_template(self) -> str:
        """Get the prompt template to use."""
        if self.config.prompt_template:
            return self.config.prompt_template
        task_type = self.config.task_type or "generic"
        return _TEMPLATES.get(task_type.lower(), GENERIC_EVAL_TEMPLATE)

    def _truncate(self, text: str, max_chars: int) -> str:
        """Truncate text to max_chars, adding ellipsis if truncated."""
        if not text or len(text) <= max_chars:
            return text or ""
        return text[:max_chars] + "..."

    def _build_prompt(
        self,
        output: str,
        expected: str | None,
        example: dict[str, Any],
        candidate: Candidate,
    ) -> str:
        """Build the judge prompt from template."""
        return self._template.format(
            input=self._truncate(example.get("input", ""), self.config.max_input_chars),
            expected=self._truncate(expected or "", self.config.max_input_chars),
            output=self._truncate(output, self.config.max_output_chars),
            system_prompt=self._truncate(candidate.text, self.config.max_input_chars),
        )

    def _parse_response(self, content: str) -> dict[str, Any] | None:
        """Parse JSON response from judge, returning None on failure.

        Handles:
        - Plain JSON
        - JSON in markdown code fences (```json ... ```)
        - Nested objects/arrays
        """
        if not content:
            return None

        # Strip markdown code fences first
        # Match ```json ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL | re.IGNORECASE)
        if fence_match:
            content = fence_match.group(1).strip()

        # Find the outermost JSON object by matching balanced braces
        def find_json_object(text: str) -> str | None:
            start = text.find("{")
            if start == -1:
                return None

            depth = 0
            in_string = False
            escape_next = False

            for i, char in enumerate(text[start:], start):
                if escape_next:
                    escape_next = False
                    continue
                if char == "\\":
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
            return None

        json_str = find_json_object(content)
        if not json_str:
            return None

        try:
            parsed = json.loads(json_str)
            # Validate required fields
            if not isinstance(parsed, dict):
                return None
            # Ensure quality is a float
            if "quality" in parsed:
                parsed["quality"] = float(parsed["quality"])
            if "reasoning_quality" in parsed:
                parsed["reasoning_quality"] = float(parsed["reasoning_quality"])
            return parsed
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    async def evaluate(
        self,
        output: str,
        expected: str | None,
        example: dict[str, Any],
        candidate: Candidate,
    ) -> dict[str, Any] | None:
        """
        Evaluate a single trace and return structured diagnostic feedback.

        Args:
            output: The model's output text
            expected: Expected answer (may be None)
            example: Full example dict with input, context, etc.
            candidate: The candidate being evaluated

        Returns:
            Diagnostic dict with quality, failure_stage, failure_explanation, suggestions
            or None if evaluation fails
        """
        try:
            from litellm import acompletion
        except ImportError:
            return None

        prompt = self._build_prompt(output, expected, example, candidate)

        try:
            response = await acompletion(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            content = response.choices[0].message.content
            return self._parse_response(content or "")
        except Exception:
            # Judge failures are non-fatal; return None and let caller handle
            return None
