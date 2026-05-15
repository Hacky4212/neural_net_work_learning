from __future__ import annotations

from pathlib import Path

from game_ai.action.action_space import DiscreteActionSpace
from game_ai.config import config
from game_ai.nn.torch_utils import require_torch
from game_ai.train.utils import ClassificationMetrics, split_items


def main() -> None:
    try:
        torch, _nn, F = require_torch()
    except RuntimeError as exc:
        print(exc)
        return

    from game_ai.nn.models import ActorCriticNet
    from game_ai.nn.tensor_data import BehaviorDataset

    action_space = DiscreteActionSpace.load(config.model.action_space_path)
    dataset = BehaviorDataset(
        config.data.record_path,
        action_space,
        image_size=config.model.image_size,
        use_images=config.model.use_images,
    )
    if len(dataset) == 0:
        print(f"no usable training data: {config.data.record_path}")
        return

    train_indices, validation_indices = split_items(
        list(range(len(dataset))),
        config.model.validation_ratio,
        config.model.validation_seed,
        label_fn=lambda index: dataset.samples[index]["action"],
    )
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    validation_dataset = torch.utils.data.Subset(dataset, validation_indices)

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.model.batch_size,
        shuffle=True,
    )
    model = ActorCriticNet(
        vector_size=config.model.vector_size,
        action_count=len(action_space),
        image_size=config.model.image_size,
        use_images=config.model.use_images,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate)
    output = Path(config.model.torch_model_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    best_validation_loss = float("inf")
    best_epoch = 0
    stalled_epochs = 0

    for epoch in range(config.model.epochs):
        model.train()
        for batch in loader:
            logits, _value = model(batch["vector"], batch["image"])
            loss_each = F.cross_entropy(logits, batch["action"], reduction="none")
            weights = batch["weight"]
            loss = (loss_each * weights).sum() / torch.clamp(weights.sum(), min=1e-8)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_metrics = _evaluate_loader(torch, F, model, train_dataset, config.model.batch_size)
        validation_metrics = _evaluate_loader(torch, F, model, validation_dataset, config.model.batch_size)
        if validation_metrics.samples > 0:
            print(
                f"epoch={epoch + 1} "
                f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.3f} "
                f"val_loss={validation_metrics.loss:.4f} val_acc={validation_metrics.accuracy:.3f}"
            )
            if validation_metrics.loss + 1e-6 < best_validation_loss:
                best_validation_loss = validation_metrics.loss
                best_epoch = epoch + 1
                stalled_epochs = 0
                torch.save(model.state_dict(), output)
                print(
                    f"best_model epoch={best_epoch} "
                    f"val_loss={best_validation_loss:.4f} saved={output}"
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

    if len(validation_dataset) == 0:
        torch.save(model.state_dict(), output)
    print(f"saved model: {output}")


def _evaluate_loader(torch, F, model, dataset, batch_size: int) -> ClassificationMetrics:
    if len(dataset) == 0:
        return ClassificationMetrics(loss=0.0, accuracy=0.0, samples=0)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_weight = 0.0
    correct = 0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits, _value = model(batch["vector"], batch["image"])
            loss_each = F.cross_entropy(logits, batch["action"], reduction="none")
            weights = batch["weight"]
            total_loss += float((loss_each * weights).sum().item())
            total_weight += float(weights.sum().item())
            predictions = logits.argmax(dim=1)
            correct += int((predictions == batch["action"]).sum().item())
            total_samples += int(batch["action"].shape[0])

    loss = total_loss / total_weight if total_weight > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    return ClassificationMetrics(loss=loss, accuracy=accuracy, samples=total_samples)


if __name__ == "__main__":
    main()
