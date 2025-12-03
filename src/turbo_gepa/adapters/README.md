# TurboGEPA Adapters

TurboGEPA adapters provide integration points between TurboGEPA's high-throughput optimization engine and various AI frameworks.

## Available Adapters

### DefaultAdapter

The `DefaultAdapter` is for optimizing single-component text prompts (e.g., system prompts). It provides automatic configuration based on dataset size and full integration with TurboGEPA's features.

**Location**: `src/turbo_gepa/adapters/default_adapter.py`

**⚠️ IMPORTANT**: By default, `DefaultAdapter` uses **AIME-style numeric answer matching**
(looking for `### <number>` or `\boxed{...}`). This returns `quality=0.0` for non-numeric
tasks! For text generation, reviews, code, etc., you MUST provide a custom `eval_fn`.

**Basic Example (numeric tasks)**:
```python
from turbo_gepa.adapters import DefaultAdapter

adapter = DefaultAdapter(
    dataset=trainset,
    task_lm="openrouter/google/gemini-flash-1.5",
    reflection_lm="openrouter/google/gemini-flash-1.5"
)

result = adapter.optimize(seeds=["You are a helpful assistant."])
```

**Custom eval_fn Example (text generation tasks)**:
```python
from turbo_gepa.adapters import DefaultAdapter
from typing import Any

def evaluate_review_quality(
    model_output: str,
    expected_answer: str,
    example: dict[str, Any]
) -> dict[str, float]:
    """Custom evaluation for text generation tasks.

    Args:
        model_output: The LLM's response text
        expected_answer: The expected answer from the dataset (may be unused)
        example: The full example dict with 'input', 'answer', 'additional_context'

    Returns:
        Dict with at least 'quality' key (0.0 to 1.0). Can include other metrics
        like 'tokens', 'latency', etc.
    """
    score = 0.0
    output_lower = model_output.lower()

    # Example: heuristic scoring for review sections
    if "summary" in output_lower:
        score += 0.25
    if "strength" in output_lower:
        score += 0.25
    if "weakness" in output_lower:
        score += 0.25
    if "recommend" in output_lower:
        score += 0.25

    return {"quality": score}

adapter = DefaultAdapter(
    dataset=trainset,
    task_lm="openrouter/openai/gpt-oss-120b:nitro",
    reflection_lm="openrouter/x-ai/grok-4-fast",
    eval_fn=evaluate_review_quality,  # Custom evaluation!
)
```

**Key Parameters**:
- `eval_fn`: Custom function to compute quality from LLM output. Required for non-numeric tasks.
- `scoring_fn`: Function to combine metrics into a single score for candidate selection.
  Operates on already-computed objectives (after `eval_fn` runs).
- `auto_config`: Enable automatic configuration based on dataset size (default: True)

### DSpyAdapter

The `DSpyAdapter` allows TurboGEPA to optimize DSPy programs by evolving their instruction text. It provides:

- ✅ Async evaluation with trace capture
- ✅ Integration with DSPy's `bootstrap_trace_data` for reflection
- ✅ Optional feedback functions for per-predictor improvements
- ✅ LLM-based reflection via `InstructionProposalPrompt` (requires feedback + LLM)
- ✅ Pareto frontier optimization

**Location**: `src/turbo_gepa/adapters/dspy_adapter.py`

**Example**:
```python
from turbo_gepa.adapters.dspy_adapter import DSpyAdapter
import dspy

# Create your DSPy module
class MyModule(dspy.Module):
    def __init__(self):
        self.predictor = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.predictor(question=question)

# Define metric
def metric(example, prediction, trace=None):
    return example.answer.lower() == prediction.answer.lower()

# Create adapter
adapter = DSpyAdapter(
    student_module=MyModule(),
    metric_fn=metric,
    trainset=trainset,
)

# Optimize
result = await adapter.optimize_async(
    seed_instructions={"predictor": "Solve the problem step by step."},
    max_rounds=10,
)

best_program = result['best_program']
```

**Advanced Usage with Feedback**:
```python
from turbo_gepa.adapters.dspy_adapter import DSpyAdapter, ScoreWithFeedback

# Define feedback function for better reflection
def predictor_feedback(predictor_output, predictor_inputs,
                      module_inputs, module_outputs, captured_trace):
    # Analyze the predictor's performance
    is_correct = module_inputs.answer in str(predictor_output)
    score = 1.0 if is_correct else 0.0

    feedback = "Correct" if is_correct else "Incorrect - be more precise"
    return ScoreWithFeedback(score=score, feedback=feedback)

# Create adapter with feedback
adapter = DSpyAdapter(
    student_module=student,
    metric_fn=metric,
    trainset=trainset,
    feedback_map={"predictor": predictor_feedback}
)

# Define async reflection LLM
async def reflection_lm(prompt: str) -> str:
    # Use your LLM to generate improved instructions
    response = await your_llm_call(prompt)
    return response

# Optimize with LLM reflection
result = await adapter.optimize_async(
    seed_instructions={"predictor": "Initial instruction"},
    max_rounds=10,
    reflection_lm=reflection_lm
)
```

**Full Example**: See the inline example above. A standalone example script will be added under `examples/` in a future update.

## Creating Custom Adapters

To create a custom adapter for your framework:

1. Implement a `_task_runner` function:
```python
async def _task_runner(self, candidate: Candidate, example_id: str) -> Dict[str, float]:
    # Execute the candidate on the example
    # Return metrics including:
    return {
        "quality": 0.8,           # Task performance (higher is better)
        "neg_cost": -0.001,       # Negative cost (e.g., -tokens/1000)
        "tokens": 1000,           # Token count
        "input": "...",           # Input for reflection
        "output": "...",          # Output for reflection
        "trace": {...},           # Any trace data for reflection
        "example_id": example_id
    }
```

2. Create a `_build_orchestrator` method:
```python
def _build_orchestrator(self, max_rounds: int = 100) -> Orchestrator:
    evaluator = AsyncEvaluator(
        cache=self.cache,
        task_runner=self._task_runner,
    )
    return Orchestrator(
        config=self.config,
        evaluator=evaluator,
        archive=self.archive,
        sampler=self.sampler,
        mutator=self.mutator,
        cache=self.cache,
    )
```

3. Implement an `optimize` or `optimize_async` method that:
   - Creates seed candidates
   - Calls `orchestrator.run()`
   - Returns results from the Pareto frontier

See `DSpyAdapter` or `DefaultAdapter` for complete examples.

## Adapter Comparison

| Feature | DefaultAdapter | DSpyAdapter |
|---------|----------------|-------------|
| **Target** | Text prompts | DSPy programs |
| **Evaluation** | LiteLLM calls | DSPy execution |
| **Trace Capture** | Basic metrics | Full DSPy traces |
| **Reflection** | LLM-based | LLM + feedback functions |
| **Multi-component** | No | Yes (multiple predictors) |
| **Auto-config** | Yes | Manual |
| **Islands** | Yes | No (single island) |

## Dependencies

- **DefaultAdapter**: litellm (required)
- **DSpyAdapter**: dspy-ai (optional, installed via `pip install turbo-gepa[dspy]`)

## Best Practices

1. **Provide feedback functions**: For complex tasks, define feedback functions to guide reflection per predictor
2. **Use LLM reflection**: Provide a `reflection_lm` function with a capable model
3. **Tune configuration**: Adjust `Config` parameters based on your dataset size and compute budget
4. **Monitor cache hit rates**: Check logs to see if caching is effective
5. **Inspect Pareto frontier**: Don't just use the "best" - explore the quality-cost tradeoffs
