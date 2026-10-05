"""Small dependency-free agreement and correlation helpers."""

from __future__ import annotations

import math
from typing import Sequence


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for index in range(start, end):
            ranks[ordered[index][0]] = average_rank
        start = end
    return ranks


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_norm = math.sqrt(sum(value * value for value in left_centered))
    right_norm = math.sqrt(sum(value * value for value in right_centered))
    if not left_norm or not right_norm:
        return None
    return sum(a * b for a, b in zip(left_centered, right_centered)) / (left_norm * right_norm)


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return pearson_correlation(_average_ranks(left), _average_ranks(right))