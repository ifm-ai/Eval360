"""
This file defines various common metrics of interest.
"""
import random
import numpy as np
from math import comb
from typing import List


def get_accuracy(corrects: List[bool] | List[List[bool]]) -> float:
    # flatten lists of lists
    if len(corrects) and isinstance(corrects[0], list):
        corrects = [j for i in corrects for j in i]
    num_correct = sum(int(correct) for correct in corrects)
    num_total = len(corrects)
    if num_total == 0:
        return float("nan")
    else:
        return num_correct / num_total


def get_bootstrap_accuracy_std(corrects: List[bool], num_samples: int = 1000) -> float:
    if len(corrects) <=1 :
        return 0.0
    return np.std([np.mean(random.sample(corrects, len(corrects) // 2)) for _ in range(num_samples)])


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    n = total sampled solutions for a problem
    c = number of correct solutions among those n
    k = pass@k target
    """
    if k <= 0:
        return 0.0
    if k > n:
        k = n  # common choice; alternatively raise ValueError

    wrong = n - c
    if wrong < k:
        return 1.0
    return 1.0 - (comb(wrong, k) / comb(n, k))


def mean_pass_at_k(per_problem_counts, k: int) -> float:
    """
    per_problem_counts: iterable of (n, c) for each problem
    """
    vals = [pass_at_k(n, c, k) for (n, c) in per_problem_counts]
    return sum(vals) / len(vals) if vals else 0.0
