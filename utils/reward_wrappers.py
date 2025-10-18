"""Reward function wrappers for different extraction methods."""

from utils.custom_gsm8k_reward import compute_score


def compute_score_strict(data_source, solution_str, ground_truth, extra_info=None):
    """Wrapper for strict reward method."""
    return compute_score(data_source, solution_str, ground_truth, {"method": "strict"})


def compute_score_flexible(data_source, solution_str, ground_truth, extra_info=None):
    """Wrapper for flexible reward method."""
    return compute_score(data_source, solution_str, ground_truth, {"method": "flexible"})


def compute_score_custom_flexible(data_source, solution_str, ground_truth, extra_info=None):
    """Wrapper for custom_flexible reward method."""
    return compute_score(data_source, solution_str, ground_truth, {"method": "custom_flexible"})

