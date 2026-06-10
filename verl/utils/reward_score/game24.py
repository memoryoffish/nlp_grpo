"""
Reward function for the 24-point game.

Scoring:
  1.0  - correct: expression uses exactly the 4 given numbers and evaluates to 24
  0.1  - format ok but wrong: has <answer> tag, numbers valid, result ≠ 24
         OR has <answer> tag but numbers are wrong
  0.0  - no <answer> tag at all
"""

import os
import re
import random

TARGET = 24


def extract_answer(solution_str: str):
    """Extract the expression from the last <answer>...</answer> tag."""
    for sep in ["<|im_start|>assistant", "Assistant:"]:
        if sep in solution_str:
            solution_str = solution_str.split(sep, 1)[1]
            break
    matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    return matches[-1].strip() if matches else None


def validate_numbers(expr: str, required_numbers: list) -> bool:
    """Check that expr uses exactly the required integers, each once."""
    try:
        nums_in_expr = [int(n) for n in re.findall(r"\d+", expr)]
        return sorted(nums_in_expr) == sorted(int(n) for n in required_numbers)
    except Exception:
        return False


def safe_eval(expr: str):
    """Evaluate an arithmetic expression; return float or None on error.

    Only the four legal Game-24 operations (+ - * /) and parentheses are allowed.
    The character whitelist alone would let `**` (power) and `//` (floor div) slip
    through (two allowed chars in a row), which are NOT legal Game-24 ops and could
    otherwise be a verifier false positive, so reject them explicitly.
    """
    if not re.match(r"^[\d\s\+\-\*\/\(\)\.]+$", expr):
        return None
    if "**" in expr or "//" in expr:
        return None
    try:
        result = eval(expr, {"__builtins__": None}, {})  # noqa: S307
        return float(result)
    except Exception:
        return None


def _reward_mode() -> str:
    """Reward variant, selected by env so the sweep arms differ by one knob only.
      sparse (default): 1.0 / 0.1 / 0.0   (the clean baseline / comparison anchor)
      shaped: adds proximity-to-24 partial credit to VALID-number wrong answers.
              This varies PER-SAMPLE within a GRPO group (unlike a constant per-puzzle
              weight, which cancels under group-relative advantage normalization), so it
              breaks the all-0.1 zero-variance trap. Greedy eval is immune (only ==24 counts).
    """
    return os.environ.get('GAME24_REWARD', 'sparse').strip().lower()


def _shaping_coef() -> float:
    try:
        return float(os.environ.get('GAME24_SHAPING_COEF', '0.5'))
    except ValueError:
        return 0.5


def compute_score(solution_str: str, ground_truth: dict,
                  format_score: float = 0.1, score: float = 1.0) -> float:
    """
    Args:
        solution_str  : full decoded model output (prompt + response)
        ground_truth  : {"numbers": list[int], "target": int, "solved_rate": float}
        format_score  : reward for correct format but wrong answer
        score         : reward for a fully correct answer
    Returns:
        float reward. sparse: {0.0, format_score, score}. shaped: wrong-but-valid gets
        format_score + coef*proximity (capped at format_score+coef).
    """
    numbers = ground_truth["numbers"]
    mode = _reward_mode()
    do_print = random.randint(1, 64) == 1

    expr = extract_answer(solution_str)

    if do_print:
        print("--------------------------------")
        print(f"Numbers: {numbers} | Target: {TARGET} | mode={mode}")
        print(f"Extracted expr: {expr}")

    if expr is None:
        if do_print:
            print("No <answer> tag found → 0.0")
        return 0.0

    # Gate: shaping is only ever applied to expressions that use the right numbers,
    # so it cannot be hacked by emitting arbitrary close-to-24 numbers.
    if not validate_numbers(expr, numbers):
        if do_print:
            print(f"Wrong numbers in expr → {format_score}")
        return format_score

    result = safe_eval(expr)
    if result is None:
        if do_print:
            print(f"Cannot evaluate expr → {format_score}")
        return format_score

    if abs(result - TARGET) < 1e-5:
        if do_print:
            print(f"Correct! {expr} = {result} → {score}")
        return score

    # wrong value, but valid numbers
    if mode == 'shaped':
        coef = _shaping_coef()
        prox = max(0.0, 1.0 - abs(result - TARGET) / 24.0)
        shaped = format_score + coef * prox  # in [format_score, format_score+coef]
        if do_print:
            print(f"Wrong result (shaped): {expr} = {result} prox={prox:.2f} → {shaped:.3f}")
        return shaped

    if do_print:
        print(f"Wrong result: {expr} = {result} (expected {TARGET}) → {format_score}")
    return format_score
