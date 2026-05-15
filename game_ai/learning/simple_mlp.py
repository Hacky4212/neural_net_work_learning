from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Prediction:
    label: int
    confidence: float


class SimpleMLP:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        seed: int = 42,
    ) -> None:
        rng = random.Random(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.w1 = _matrix(hidden_size, input_size, rng, scale=0.2)
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.w2 = _matrix(output_size, hidden_size, rng, scale=0.2)
        self.b2 = [0.0 for _ in range(output_size)]

    def predict(self, x: list[float]) -> Prediction:
        _, _, probs = self._forward(x)
        label = max(range(len(probs)), key=probs.__getitem__)
        return Prediction(label=label, confidence=probs[label])

    def predict_proba(self, x: list[float]) -> list[float]:
        return self._forward(x)[2]

    def train_one(self, x: list[float], label: int, learning_rate: float, weight: float = 1.0) -> float:
        z1, h, probs = self._forward(x)
        loss = -math.log(max(probs[label], 1e-9)) * weight

        dz2 = probs[:]
        dz2[label] -= 1.0
        dz2 = [v * weight for v in dz2]

        old_w2 = [row[:] for row in self.w2]
        for i in range(self.output_size):
            for j in range(self.hidden_size):
                self.w2[i][j] -= learning_rate * dz2[i] * h[j]
            self.b2[i] -= learning_rate * dz2[i]

        dh = [0.0 for _ in range(self.hidden_size)]
        for j in range(self.hidden_size):
            dh[j] = sum(old_w2[i][j] * dz2[i] for i in range(self.output_size))

        dz1 = [dh[j] * (1.0 - math.tanh(z1[j]) ** 2) for j in range(self.hidden_size)]
        for j in range(self.hidden_size):
            for k in range(self.input_size):
                self.w1[j][k] -= learning_rate * dz1[j] * x[k]
            self.b1[j] -= learning_rate * dz1[j]

        return loss

    def save(self, path: str, labels: list[str]) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "labels": labels,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> tuple["SimpleMLP", list[str]]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            input_size=payload["input_size"],
            hidden_size=payload["hidden_size"],
            output_size=payload["output_size"],
        )
        model.w1 = payload["w1"]
        model.b1 = payload["b1"]
        model.w2 = payload["w2"]
        model.b2 = payload["b2"]
        return model, payload["labels"]

    def _forward(self, x: list[float]) -> tuple[list[float], list[float], list[float]]:
        z1 = [_dot(row, x) + bias for row, bias in zip(self.w1, self.b1)]
        h = [math.tanh(v) for v in z1]
        logits = [_dot(row, h) + bias for row, bias in zip(self.w2, self.b2)]
        probs = _softmax(logits)
        return z1, h, probs


def _matrix(rows: int, cols: int, rng: random.Random, scale: float) -> list[list[float]]:
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    exp_values = [math.exp(v - max_value) for v in values]
    total = sum(exp_values)
    return [v / total for v in exp_values]
