"""Run bounded, reproducible discrete-time and Monte Carlo simulations."""

from __future__ import annotations

import ast
import json
import math
import random
import statistics
from typing import Any, Callable

from agent.cancellation import CancellationToken


_BINARY = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b, ast.Mod: lambda a, b: a % b,
}
_UNARY = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a, ast.Not: lambda a: not a}
_COMPARE = {
    ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
}

MAX_VARIABLES = 100
MAX_SCENARIOS = 20
MAX_EXPRESSION_CHARS = 2_000
MAX_POWER_EXPONENT = 100.0
MAX_EVALUATIONS = 500_000
MAX_SCENARIO_VARIABLE_PAIRS = 200
MAX_TRAJECTORY_STATE_VALUES = 1_000
MAX_SEED = (1 << 63) - 1


def _safe_power(base: float, exponent: float) -> float:
    """Bound exponentiation before Python allocates an enormous integer."""
    try:
        exponent_value = float(exponent)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Power exponent must be a finite number") from exc
    if not math.isfinite(exponent_value) or abs(exponent_value) > MAX_POWER_EXPONENT:
        raise ValueError(f"Power exponent must be between {-MAX_POWER_EXPONENT:g} and {MAX_POWER_EXPONENT:g}")
    try:
        value = base ** exponent
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError(f"Power operation failed: {exc}") from exc
    if isinstance(value, complex) or not math.isfinite(float(value)):
        raise ValueError("Power operation produced a non-finite number")
    return value


_BINARY[ast.Pow] = _safe_power


class _Expression:
    def __init__(self, functions: dict[str, Callable[..., float]]):
        self.functions = functions

    def compile(
        self,
        expression: str,
        variable_names: set[str],
    ) -> ast.AST:
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("Expression must be a non-empty string")
        if len(expression) > MAX_EXPRESSION_CHARS:
            raise ValueError(f"Expression exceeds the {MAX_EXPRESSION_CHARS}-character limit")
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 100:
            raise ValueError("Expression is too complex")
        self._validate_node(tree.body, variable_names)
        return tree.body

    def evaluate(self, expression: ast.AST, variables: dict[str, float]) -> float:
        value = self._node(expression, variables)
        if not isinstance(value, (int, float, bool)) or not math.isfinite(float(value)):
            raise ValueError("Expression produced a non-finite number")
        return float(value)

    def _validate_node(self, node: ast.AST, variable_names: set[str]) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
            return
        if isinstance(node, ast.Name):
            if node.id not in variable_names:
                raise ValueError(f"Unknown variable: {node.id}")
            return
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            self._validate_node(node.left, variable_names)
            self._validate_node(node.right, variable_names)
            return
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            self._validate_node(node.operand, variable_names)
            return
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            for value in node.values:
                self._validate_node(value, variable_names)
            return
        if isinstance(node, ast.Compare):
            if any(type(operator) not in _COMPARE for operator in node.ops):
                raise ValueError("Expression contains an unsupported comparison")
            self._validate_node(node.left, variable_names)
            for comparator in node.comparators:
                self._validate_node(comparator, variable_names)
            return
        if isinstance(node, ast.IfExp):
            self._validate_node(node.test, variable_names)
            self._validate_node(node.body, variable_names)
            self._validate_node(node.orelse, variable_names)
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in self.functions:
                raise ValueError(f"Unsupported function: {node.func.id}")
            if node.keywords:
                raise ValueError("Simulation functions accept positional arguments only")
            for argument in node.args:
                self._validate_node(argument, variable_names)
            return
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    def _node(self, node: ast.AST, variables: dict[str, float]) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"Unknown variable: {node.id}")
            return variables[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](self._node(node.left, variables), self._node(node.right, variables))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](self._node(node.operand, variables))
        if isinstance(node, ast.BoolOp):
            values = [bool(self._node(value, variables)) for value in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare):
            left = self._node(node.left, variables)
            for operator, comparator in zip(node.ops, node.comparators):
                right = self._node(comparator, variables)
                if type(operator) not in _COMPARE or not _COMPARE[type(operator)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self._node(node.body if self._node(node.test, variables) else node.orelse, variables)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in self.functions:
            return self.functions[node.func.id](*[self._node(arg, variables) for arg in node.args])
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _bounded_integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def run_simulation(
    variables: dict[str, float],
    equations: dict[str, str],
    steps: int = 10,
    dt: float = 1.0,
    mode: str = "recurrence",
    scenarios: list[dict] | None = None,
    trials: int = 1,
    seed: int | None = None,
    cancellation_token: CancellationToken | None = None,
) -> str:
    """Simulate state equations; equations are safely parsed, never passed to eval()."""
    if cancellation_token:
        cancellation_token.raise_if_cancelled()
    if not isinstance(variables, dict) or not isinstance(equations, dict) or not variables or not equations:
        return json.dumps({"error": "variables and equations must be non-empty objects"})
    if len(variables) > MAX_VARIABLES or len(equations) > MAX_VARIABLES:
        return json.dumps({"error": f"At most {MAX_VARIABLES} variables/equations are allowed"})
    invalid_names = sorted(
        str(name) for name in set(variables) | set(equations)
        if not isinstance(name, str) or not name.isidentifier()
    )
    reserved_names = sorted(set(variables) & {"step", "time", "dt", "pi", "e"})
    if invalid_names:
        return json.dumps({"error": "Variable and equation names must be valid identifiers", "invalid": invalid_names})
    if reserved_names:
        return json.dumps({"error": "Variable names conflict with reserved simulation values", "reserved": reserved_names})
    if set(equations) - set(variables):
        return json.dumps({"error": "Every equation target must be declared in variables", "unknown": sorted(set(equations) - set(variables))})
    try:
        step_count = _bounded_integer(steps, "steps", 1, 10_000)
        trial_count = _bounded_integer(trials, "trials", 1, 1_000)
        if isinstance(dt, bool):
            raise ValueError("dt must be numeric")
        dt_value = float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        return json.dumps({"error": str(exc)})
    if not math.isfinite(dt_value) or dt_value <= 0 or dt_value > 1_000_000:
        return json.dumps({"error": "dt must be finite and between 0 (exclusive) and 1,000,000"})
    if any(isinstance(value, bool) for value in variables.values()):
        return json.dumps({"error": "Initial variable values must be numbers, not booleans"})
    try:
        initial_state = {key: float(value) for key, value in variables.items()}
    except (TypeError, ValueError, OverflowError):
        return json.dumps({"error": "Every initial variable value must be numeric"})
    if not all(math.isfinite(value) for value in initial_state.values()):
        return json.dumps({"error": "Every initial variable value must be finite"})
    invalid_expressions = [key for key, value in equations.items() if not isinstance(value, str) or not value.strip()]
    if invalid_expressions:
        return json.dumps({"error": "Every equation must be a non-empty string", "invalid": invalid_expressions})
    mode = str(mode or "").strip().lower()
    if mode not in {"recurrence", "euler"}:
        return json.dumps({"error": "mode must be 'recurrence' or 'euler'"})

    if scenarios is not None and not isinstance(scenarios, list):
        return json.dumps({"error": "scenarios must be an array"})
    if isinstance(scenarios, list) and not scenarios:
        return json.dumps({"error": "scenarios must contain at least one scenario when provided"})
    scenario_defs = scenarios if scenarios is not None else [{"name": "baseline", "overrides": {}}]
    if len(scenario_defs) > MAX_SCENARIOS:
        return json.dumps({"error": f"At most {MAX_SCENARIOS} scenarios are allowed"})
    evaluation_count = step_count * trial_count * len(equations) * len(scenario_defs)
    if evaluation_count > MAX_EVALUATIONS:
        return json.dumps({
            "error": f"Simulation exceeds the {MAX_EVALUATIONS:,} evaluation limit across all scenarios"
        })
    if len(variables) * len(scenario_defs) > MAX_SCENARIO_VARIABLE_PAIRS:
        return json.dumps({
            "error": (
                "Simulation result would be too large: scenarios multiplied by variables "
                f"must not exceed {MAX_SCENARIO_VARIABLE_PAIRS}"
            )
        })
    if (
        seed is not None
        and (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or abs(seed) > MAX_SEED
        )
    ):
        return json.dumps({
            "error": f"seed must be an integer between {-MAX_SEED} and {MAX_SEED}, or null"
        })
    try:
        randomizer = random.Random(seed)
    except (TypeError, ValueError):
        return json.dumps({"error": "seed must be an integer or null"})
    evaluator = _Expression({
        "min": min, "max": max, "abs": abs, "sqrt": math.sqrt, "log": math.log,
        "exp": math.exp, "sin": math.sin, "cos": math.cos, "floor": math.floor,
        "ceil": math.ceil, "normal": randomizer.gauss, "uniform": randomizer.uniform,
    })
    expression_names = set(variables) | {"step", "time", "dt", "pi", "e"}
    try:
        compiled_equations = {
            key: evaluator.compile(expression, expression_names)
            for key, expression in equations.items()
        }
    except (SyntaxError, TypeError, ValueError) as exc:
        return json.dumps({"error": f"Invalid simulation expression: {exc}"})

    validated_scenarios: list[tuple[str, dict[str, Any], dict[str, float]]] = []
    scenario_names: set[str] = set()
    for scenario_index, scenario in enumerate(scenario_defs):
        if not isinstance(scenario, dict):
            return json.dumps({"error": f"scenarios[{scenario_index}] must be an object"})
        unknown_fields = sorted(set(scenario) - {"name", "overrides"})
        if unknown_fields:
            return json.dumps({
                "error": f"scenarios[{scenario_index}] contains unsupported fields",
                "unknown": unknown_fields,
            })
        raw_name = scenario.get("name")
        if raw_name is None:
            name = f"scenario-{scenario_index + 1}"
        elif not isinstance(raw_name, str):
            return json.dumps({"error": f"scenarios[{scenario_index}].name must be a string"})
        else:
            name = raw_name.strip()
        if not name or len(name) > 200 or any(ord(char) < 32 for char in name):
            return json.dumps({"error": f"scenarios[{scenario_index}].name is invalid or exceeds 200 characters"})
        normalized_name = name.casefold()
        if normalized_name in scenario_names:
            return json.dumps({"error": f"Duplicate scenario name: {name}"})
        scenario_names.add(normalized_name)
        overrides = scenario.get("overrides", {})
        if not isinstance(overrides, dict):
            return json.dumps({"error": f"Scenario '{name}' overrides must be an object"})
        unknown_overrides = sorted(str(value) for value in set(overrides) - set(variables))
        if unknown_overrides:
            return json.dumps({"error": f"Scenario '{name}' overrides unknown variables", "unknown": unknown_overrides})
        if any(isinstance(value, bool) for value in overrides.values()):
            return json.dumps({"error": f"Scenario '{name}' override values must be numbers, not booleans"})
        try:
            scenario_state = {**initial_state, **{key: float(value) for key, value in overrides.items()}}
        except (TypeError, ValueError, OverflowError):
            return json.dumps({"error": f"Scenario '{name}' override values must be numeric"})
        if not all(math.isfinite(value) for value in scenario_state.values()):
            return json.dumps({"error": f"Scenario '{name}' override values must be finite"})
        validated_scenarios.append((name, overrides, scenario_state))

    state_values_per_point = len(variables) * len(validated_scenarios)
    points_per_scenario = max(
        2,
        min(
            step_count + 1,
            MAX_TRAJECTORY_STATE_VALUES // max(1, state_values_per_point),
        ),
    )
    update_samples = points_per_scenario - 1
    sampled_steps = {
        round(index * step_count / update_samples)
        for index in range(1, update_samples + 1)
    }

    results = []
    for name, overrides, scenario_state in validated_scenarios:
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        trial_finals: dict[str, list[float]] = {key: [] for key in variables}
        sample_series = []
        for trial in range(trial_count):
            if cancellation_token:
                cancellation_token.raise_if_cancelled()
            state = dict(scenario_state)
            series = [{"step": 0, "time": 0.0, **state}]
            for step in range(1, step_count + 1):
                if cancellation_token and (step == 1 or step % 32 == 0):
                    cancellation_token.raise_if_cancelled()
                context = {**state, "step": float(step), "time": step * dt_value, "dt": dt_value, "pi": math.pi, "e": math.e}
                try:
                    calculated = {
                        key: evaluator.evaluate(expression, context)
                        for key, expression in compiled_equations.items()
                    }
                except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
                    return json.dumps({"error": f"Invalid simulation expression at scenario '{name}', trial {trial + 1}, step {step}: {exc}"})
                if mode == "euler":
                    state = {**state, **{key: state[key] + dt_value * value for key, value in calculated.items()}}
                else:
                    state = {**state, **calculated}
                if not all(math.isfinite(value) for value in state.values()):
                    return json.dumps({"error": f"Simulation produced a non-finite state at scenario '{name}', trial {trial + 1}, step {step}"})
                if trial == 0 and step in sampled_steps:
                    series.append({"step": step, "time": step * dt_value, **state})
            if trial == 0:
                sample_series = series
            for key, value in state.items():
                trial_finals[key].append(value)

        summary = {
            key: {
                "mean": statistics.fmean(values),
                "min": min(values), "max": max(values),
                "p05": _percentile(values, 0.05), "p50": _percentile(values, 0.5), "p95": _percentile(values, 0.95),
            }
            for key, values in trial_finals.items()
        }
        results.append({
            "name": name,
            "overrides": overrides,
            "sample_trajectory": sample_series,
            "trajectory_points": len(sample_series),
            "trajectory_truncated": len(sample_series) < step_count + 1,
            "final_distribution": summary,
        })

    return json.dumps({
        "mode": mode, "steps": step_count, "dt": dt_value, "trials": trial_count, "seed": seed,
        "evaluations": evaluation_count,
        "scenarios": results,
        "interpretation_note": "Outputs are conditional on the supplied equations and assumptions; they are not independently validated forecasts.",
    }, ensure_ascii=False)
