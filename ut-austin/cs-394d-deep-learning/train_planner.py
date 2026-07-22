"""
Usage examples:
    python3 -m homework.train_planner --model mlp_planner --epochs 20 --batch_size 128 --lr 0.001
    python3 -m homework.train_planner --model transformer_planner --epochs 30 --batch_size 128 --lr 0.0005
    python3 -m homework.train_planner --model cnn_planner --epochs 30 --batch_size 128 --lr 0.001
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .datasets.road_dataset import load_data
from .metrics import PlannerMetric
from .models import MODEL_FACTORY, save_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def masked_planner_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    loss = loss * mask[..., None]
    return loss.sum() / mask.sum().clamp_min(1)


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader, device: torch.device, model_name: str) -> dict[str, float]:
    metric = PlannerMetric()
    model.eval()

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if model_name == "cnn_planner":
            pred = model(batch["image"])
        else:
            pred = model(batch["track_left"], batch["track_right"])

        metric.add(pred, batch["waypoints"], batch["waypoints_mask"])

    return metric.compute()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_FACTORY), required=True)
    parser.add_argument("--train_path", default="drive_data/train")
    parser.add_argument("--val_path", default="drive_data/val")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = get_device()
    torch.manual_seed(2024)

    transform_pipeline = "default" if args.model == "cnn_planner" else "state_only"
    train_loader = load_data(
        args.train_path,
        transform_pipeline=transform_pipeline,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = load_data(
        args.val_path,
        transform_pipeline=transform_pipeline,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = MODEL_FACTORY[args.model]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = float("inf")
    best_path = None

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        total = 0

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()

            if args.model == "cnn_planner":
                pred = model(batch["image"])
            else:
                pred = model(batch["track_left"], batch["track_right"])

            loss = masked_planner_loss(pred, batch["waypoints"], batch["waypoints_mask"])
            loss.backward()
            optimizer.step()

            bs = batch["waypoints"].shape[0]
            running_loss += loss.item() * bs
            total += bs

        metrics = evaluate(model, val_loader, device, args.model)
        score = metrics["longitudinal_error"] + metrics["lateral_error"]

        print(
            f"epoch {epoch + 1:02d} | "
            f"train_loss {running_loss / total:.4f} | "
            f"val_lon {metrics['longitudinal_error']:.4f} | "
            f"val_lat {metrics['lateral_error']:.4f} | "
            f"val_l1 {metrics['l1_error']:.4f}"
        )

        if score < best_score:
            best_score = score
            best_path = save_model(model)
            print(f"saved best model to {best_path}")

    print(f"best combined val error: {best_score:.4f}")
    if best_path is not None:
        print(f"best weights: {Path(best_path).name}")


if __name__ == "__main__":
    main()
