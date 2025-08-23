import re

_SOLUTION_CLIP_CHARS = 300


def extract_solution_strict(solution_str):
    """Extract solution using Verl's strict method - requires #### format."""
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
    if len(solutions) == 0:
        return None
    return solutions[-1].replace(",", "").replace("$", "")


def extract_solution_flexible(solution_str):
    """Extract solution using Verl's flexible method - finds last valid number."""
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
    final_answer = None
    if len(answer) == 0:
        return None
    invalid_str = ["", "."]
    for final_answer in reversed(answer):
        if final_answer not in invalid_str:
            break
    return final_answer


def extract_solution_custom_flexible(solution_str: str):
    if (marker_pos := solution_str.find("#### ")) != -1:
        solution_str = solution_str[: min(len(solution_str), marker_pos + 10)]
    return extract_solution_flexible(solution_str)


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    method = (
        "custom_flexible"
        if not extra_info
        else extra_info.get("method", "custom_flexible")
    )

    include_format_reward = False
    num_hashes = solution_str.count("####")

    if method == "strict":
        final_answer = extract_solution_strict(solution_str)
    elif method == "flexible":
        final_answer = extract_solution_flexible(solution_str)
    elif method == "custom_flexible":
        final_answer = extract_solution_custom_flexible(solution_str)
    else:
        raise ValueError(f"Unknown evaluation method: {method}")

    score = 1.0 if final_answer == ground_truth else 0.0
    if include_format_reward and score and num_hashes != 1:
        # want the signal to be high reward, get the answer, but slightly worst if bad format, so there's certain bias towards that.
        return score - 0.2
    return score
