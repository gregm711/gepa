"""
Post-run report generator for TurboGEPA.
Analyzes the evolution history and produces a human-readable summary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from turbo_gepa.orchestrator import Orchestrator


def generate_markdown_report(orchestrator: Orchestrator) -> str:
    """Generate a markdown report summarizing the optimization run."""
    run_id = orchestrator.run_id
    metrics = orchestrator.metrics_snapshot()
    
    # Executive Summary
    lines = [
        f"# TurboGEPA Run Report: {run_id}",
        "",
        "## 📊 Executive Summary",
        "",
        f"- **Status**: {orchestrator.stop_reason or 'Unknown'}",
        f"- **Total Evaluations**: {metrics.get('evaluations_total', 0)}",
        f"- **Best Quality**: {metrics.get('best_quality', 0.0):.2%}",
        f"- **Best Shard**: {metrics.get('best_shard_fraction', 0.0):.0%}",
        f"- **Time to Target**: {metrics.get('time_to_target_seconds', 'N/A')}s",
        "",
    ]

    # Best Candidate Analysis
    north_star = getattr(orchestrator, "_north_star_prompt", None)
    if north_star:
        lines.extend([
            "## 🏆 Winning Prompt",
            "",
            "```text",
            north_star,
            "```",
            "",
        ])
    
    # Lineage Analysis (Simplified)
    lineage_data = orchestrator.get_candidate_lineage_data()
    if lineage_data:
        lines.extend([
            "## 🧬 Lineage Analysis",
            "",
            f"- **Total Candidates**: {len(lineage_data)}",
            f"- **Generations**: {max((c.get('generation', 0) for c in lineage_data), default=0)}",
            "",
        ])

    # Operator Effectiveness
    ops = metrics.get('mutations_by_operator', {})
    if ops:
        lines.extend([
            "## 🛠️ Mutation Strategy Effectiveness",
            "",
            "| Strategy | Count |",
            "| :--- | :--- |",
        ])
        for op, count in ops.items():
            lines.append(f"| {op} | {count} |")
        lines.append("")

    # Cost Analysis
    cost = metrics.get('total_cost_usd', 0.0)
    lines.extend([
        "## 💰 Cost Analysis",
        "",
        f"- **Estimated Total Cost**: ${cost:.4f}",
        "",
    ])

    return "\n".join(lines)