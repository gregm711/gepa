"""
Evaluator modules for TurboGEPA.

This package contains optional LLM-as-judge evaluators for rich diagnostic feedback.
"""

from turbo_gepa.evaluators.llm_judge import (
    CODING_EVAL_TEMPLATE,
    GENERIC_EVAL_TEMPLATE,
    QA_EVAL_TEMPLATE,
    RAG_EVAL_TEMPLATE,
    LLMJudgeConfig,
    LLMJudgeEvaluator,
)

__all__ = [
    "LLMJudgeConfig",
    "LLMJudgeEvaluator",
    "QA_EVAL_TEMPLATE",
    "RAG_EVAL_TEMPLATE",
    "CODING_EVAL_TEMPLATE",
    "GENERIC_EVAL_TEMPLATE",
]
