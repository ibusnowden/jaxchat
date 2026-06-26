"""Tests for the GSM8K shaped GRPO reward and the SFT formatter.

These guard the fix for the dead-gradient GRPO run (mean_reward pinned at 0):
the shaped reward must give *within-group* variance for a cold policy, while the
strict eval reward and the SFT target stay mutually consistent.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np  # noqa: E402

from tasks import gsm8k  # noqa: E402

GOLD = 72.0
CORRECT_BOXED = "We compute 48+24=72. The final answer is \\boxed{72}."
WRONG_BOXED = "I think it is \\boxed{70}."
CORRECT_NOBOX = "So the total is 72."
WRONG_NOBOX = "Maybe around 50 or so."
EMPTY = "Hmm let me think..."


def test_strict_reward_unchanged():
    # The headline eval metric: exact match, lenient last-number fallback.
    assert gsm8k.reward(CORRECT_BOXED, GOLD) == 1.0
    assert gsm8k.reward(WRONG_BOXED, GOLD) == 0.0
    assert gsm8k.reward(CORRECT_NOBOX, GOLD) == 1.0  # fallback to trailing number
    assert gsm8k.reward(EMPTY, GOLD) == 0.0


def test_shaped_components():
    c = gsm8k.reward_components(CORRECT_BOXED, GOLD, format_bonus=0.1)
    assert c["correct"] == 1.0 and c["has_format"] == 1.0
    assert c["reward"] == 1.1

    c = gsm8k.reward_components(WRONG_BOXED, GOLD, format_bonus=0.1)
    assert c["correct"] == 0.0 and c["has_format"] == 1.0
    assert c["reward"] == 0.1  # format-only partial credit

    c = gsm8k.reward_components(CORRECT_NOBOX, GOLD, format_bonus=0.1)
    assert c["correct"] == 1.0 and c["has_format"] == 0.0  # right but didn't box
    assert c["reward"] == 1.0

    c = gsm8k.reward_components(EMPTY, GOLD, format_bonus=0.1)
    assert c["reward"] == 0.0


def test_correctness_term_equals_eval_metric():
    # The shaped correctness flag must never disagree with the strict eval reward
    # (RL should optimize exactly the metric we report, plus pure shaping on top).
    for text in (CORRECT_BOXED, WRONG_BOXED, CORRECT_NOBOX, WRONG_NOBOX, EMPTY):
        comp = gsm8k.reward_components(text, GOLD, format_bonus=0.1, proximity_coef=0.3)
        assert comp["correct"] == gsm8k.reward(text, GOLD)


def test_proximity_is_monotonic_and_bounded():
    near = gsm8k.reward_components("\\boxed{71}", GOLD, format_bonus=0.0, proximity_coef=1.0)["proximity"]
    far = gsm8k.reward_components("\\boxed{5}", GOLD, format_bonus=0.0, proximity_coef=1.0)["proximity"]
    assert 0.0 < far < near < 1.0
    # Correct answers get no proximity bonus (already maxed on correctness).
    assert gsm8k.reward_components("\\boxed{72}", GOLD, proximity_coef=1.0)["proximity"] == 0.0


def test_within_group_variance_breaks_the_deadlock():
    # The whole point: a cold-policy group (all incorrect) must still vary under
    # the shaped reward, whereas the strict reward collapses it to zero variance.
    shaped = gsm8k.make_reward_fn("shaped", format_bonus=0.1)
    strict = gsm8k.make_reward_fn("strict")
    group = [WRONG_BOXED, WRONG_NOBOX, EMPTY, "\\boxed{99}"]  # none correct
    shaped_r = [shaped(t, GOLD)["reward"] for t in group]
    strict_r = [strict(t, GOLD)["reward"] for t in group]
    assert np.std(shaped_r) > 0.0, "shaped reward must give a gradient on an all-wrong group"
    assert np.std(strict_r) == 0.0, "strict reward is the dead case (no gradient)"


def test_strict_mode_ignores_shaping_knobs():
    fn = gsm8k.make_reward_fn("strict", format_bonus=0.5, proximity_coef=0.9)
    assert fn(WRONG_BOXED, GOLD)["reward"] == 0.0  # format bonus suppressed in strict mode
    assert fn(CORRECT_BOXED, GOLD)["reward"] == 1.0


def test_sft_formatter_strips_calc_and_boxes_answer():
    raw = ("Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
           "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n#### 72")
    msgs = gsm8k.build_sft_messages("Natalia sold clips...", raw)["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    target = msgs[-1]["content"]
    assert "<<" not in target and ">>" not in target  # calculator markers gone
    assert "\\boxed{72}" in target
    # The SFT target must itself score a perfect reward — train target == objective.
    assert gsm8k.reward(target, 72.0) == 1.0
    assert gsm8k.reward_components(target, 72.0, format_bonus=0.1)["reward"] == 1.1


def test_sft_formatter_without_marker_falls_back_to_gold():
    target = gsm8k.format_solution("Adding gives five, which is 5.")
    # No #### marker present; gold parsed from the body, still boxed.
    assert "\\boxed{5}" in target


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
