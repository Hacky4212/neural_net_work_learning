from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class ClassificationMetrics:
    loss: float
    accuracy: float
    samples: int


def split_items(
    items: Sequence[T],
    validation_ratio: float,
    seed: int,
    label_fn: Callable[[T], object] | None = None,
) -> tuple[list[T], list[T]]:
    sequence = list(items)
    if len(sequence) < 2 or validation_ratio <= 0:
        return sequence, []

    ratio = min(max(validation_ratio, 0.0), 0.9)
    rng = random.Random(seed)

    if label_fn is None:
        validation_size = int(round(len(sequence) * ratio))
        validation_size = max(1, min(len(sequence) - 1, validation_size))

        indices = list(range(len(sequence)))
        rng.shuffle(indices)
        validation_indices = set(indices[:validation_size])

        train_items = [item for index, item in enumerate(sequence) if index not in validation_indices]
        validation_items = [item for index, item in enumerate(sequence) if index in validation_indices]
        return train_items, validation_items

    groups: dict[object, list[T]] = {}
    for item in sequence:
        groups.setdefault(label_fn(item), []).append(item)

    train_items: list[T] = []
    validation_items: list[T] = []
    for group_items in groups.values():
        rng.shuffle(group_items)
        if len(group_items) < 2:
            train_items.extend(group_items)
            continue

        validation_size = int(round(len(group_items) * ratio))
        validation_size = max(1, min(len(group_items) - 1, validation_size))
        validation_items.extend(group_items[:validation_size])
        train_items.extend(group_items[validation_size:])

    rng.shuffle(train_items)
    rng.shuffle(validation_items)
    return train_items, validation_items
