from __future__ import annotations

import math
import random
from pathlib import Path

from game_ai.config import config
from game_ai.data.dataset import JsonlDataset, is_informative_sample, reward_to_sample_weight
from game_ai.learning.action_codec import action_to_label
from game_ai.learning.features import FEATURE_SIZE, dict_state_to_vector
from game_ai.learning.simple_mlp import SimpleMLP
from game_ai.train.utils import ClassificationMetrics, split_items


def main() -> None:
    samples = _load_samples(config.data.record_path)
    if not samples:
        print(f"no training data: {config.data.record_path}")
        return

    train_samples, validation_samples = split_items(
        samples,
        config.model.validation_ratio,
        config.model.validation_seed,
        label_fn=lambda sample: sample["label"],
    )
    labels = sorted({sample["label"] for sample in samples})
    label_to_id = {label: index for index, label in enumerate(labels)}

    model = SimpleMLP(
        input_size=FEATURE_SIZE,
        hidden_size=config.model.hidden_size,
        output_size=len(labels),
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    stalled_epochs = 0

    for epoch in range(config.model.epochs):
        random.shuffle(train_samples)
        for sample in train_samples:
            model.train_one(
                sample["features"],
                label_to_id[sample["label"]],
                config.model.learning_rate,
                sample["weight"],
            )

        train_metrics = _evaluate_samples(model, train_samples, label_to_id)
        if validation_samples:
            validation_metrics = _evaluate_samples(model, validation_samples, label_to_id)
            print(
                f"epoch={epoch + 1} "
                f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.3f} "
                f"val_loss={validation_metrics.loss:.4f} val_acc={validation_metrics.accuracy:.3f}"
            )
            if validation_metrics.loss + 1e-6 < best_validation_loss:
                best_validation_loss = validation_metrics.loss
                best_epoch = epoch + 1
                stalled_epochs = 0
                model.save(config.model.model_path, labels)
                print(
                    f"best_model epoch={best_epoch} "
                    f"val_loss={best_validation_loss:.4f} saved={config.model.model_path}"
                )
            else:
                stalled_epochs += 1
                if (
                    config.model.early_stopping_patience > 0
                    and stalled_epochs >= config.model.early_stopping_patience
                ):
                    print(
                        f"early_stop epoch={epoch + 1} "
                        f"best_epoch={best_epoch} best_val_loss={best_validation_loss:.4f}"
                    )
                    break
        else:
            print(
                f"epoch={epoch + 1} "
                f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.3f}"
            )

    if not validation_samples:
        model.save(config.model.model_path, labels)

    print(f"saved model: {config.model.model_path}")
    print(f"labels: {labels}")


def _load_samples(path: str) -> list[dict]:
    data_path = Path(path)
    if not data_path.exists():
        return []

    samples: list[dict] = []
    for row in JsonlDataset(path):
        if not is_informative_sample(row):
            continue

        state = row.get("state", {})
        action = row.get("decision", {}).get("action", {})
        reward = row.get("reward", {}).get("step", 0.0)

        samples.append(
            {
                "features": dict_state_to_vector(state),
                "label": action_to_label(action),
                "weight": reward_to_sample_weight(reward),
            }
        )

    return samples


def _evaluate_samples(
    model: SimpleMLP,
    samples: list[dict],
    label_to_id: dict[str, int],
) -> ClassificationMetrics:
    if not samples:
        return ClassificationMetrics(loss=0.0, accuracy=0.0, samples=0)

    total_loss = 0.0
    total_weight = 0.0
    correct = 0
    for sample in samples:
        probs = model.predict_proba(sample["features"])
        label_index = label_to_id[sample["label"]]
        total_loss += -math.log(max(probs[label_index], 1e-9)) * sample["weight"]
        prediction = max(range(len(probs)), key=probs.__getitem__)
        if prediction == label_index:
            correct += 1
        total_weight += sample["weight"]

    loss = total_loss / total_weight if total_weight > 0 else 0.0
    accuracy = correct / len(samples)
    return ClassificationMetrics(loss=loss, accuracy=accuracy, samples=len(samples))


if __name__ == "__main__":
    main()
