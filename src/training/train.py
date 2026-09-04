"""
src/training/train.py
------------------------
Config-driven training entrypoint.

    python3 train.py --config configs/baseline.yaml

Loads data (patient-level split), builds the model + loss from config,
trains with checkpointing on best validation macro ROC-AUC, logs every
run to MLflow (params, per-epoch losses, final metrics, checkpoint
artifact), and writes a final metrics JSON next to the checkpoint.
"""
import argparse
import json
import time
from pathlib import Path

import mlflow
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import ChestXrayDataset
from src.data.split import patient_level_split
from src.data.synthetic_generator import LABELS
from src.evaluation.metrics import evaluate_predictions, measure_inference_latency
from src.models.factory import DEFAULT_IMAGE_SIZE, build_model
from src.preprocessing.transforms import AugmentationConfig, build_transforms
from src.training.losses import build_loss

ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: dict, image_size: int):
    df = pd.read_csv(ROOT / cfg["data"]["labels_csv"])
    train_df, val_df, test_df = patient_level_split(
        df, train_frac=cfg["data"]["train_frac"], val_frac=cfg["data"]["val_frac"], seed=cfg["data"]["seed"],
    )

    aug = AugmentationConfig(**cfg["augmentation"])
    train_tf = build_transforms(image_size, train=True, aug=aug)
    eval_tf = build_transforms(image_size, train=False)

    images_dir = ROOT / cfg["data"]["images_dir"]
    train_ds = ChestXrayDataset(train_df, images_dir, transform=train_tf)
    val_ds = ChestXrayDataset(val_df, images_dir, transform=eval_tf)
    test_ds = ChestXrayDataset(test_df, images_dir, transform=eval_tf)

    bs = cfg["training"]["batch_size"]
    nw = cfg["training"]["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)

    return train_loader, val_loader, test_loader, train_df


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for images, labels, _ in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    import numpy as np
    return np.concatenate(all_labels), np.concatenate(all_probs)


def train_one_config(cfg: dict, quiet: bool = False) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = cfg["model"]["name"]
    image_size = DEFAULT_IMAGE_SIZE[model_name]

    train_loader, val_loader, test_loader, train_df = build_dataloaders(cfg, image_size)

    model = build_model(model_name, num_labels=len(LABELS), pretrained=cfg["model"]["pretrained"],
                         dropout=cfg["model"]["dropout"]).to(device)

    label_matrix = torch.tensor(train_df[LABELS].values.astype("float32"))
    loss_fn = build_loss(
        cfg["training"]["loss"], label_matrix=label_matrix,
        alpha=cfg["training"].get("focal_alpha", 0.25), gamma=cfg["training"].get("focal_gamma", 2.0),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"],
    )

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    checkpoint_dir = ROOT / cfg["output"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_name = cfg["output"]["run_name"]
    ckpt_path = checkpoint_dir / f"{run_name}.pt"

    best_val_auc = -1.0
    epochs_without_improvement = 0
    history = []

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": model_name, "pretrained": cfg["model"]["pretrained"],
            "batch_size": cfg["training"]["batch_size"], "lr": cfg["training"]["learning_rate"],
            "epochs": cfg["training"]["epochs"], "loss": cfg["training"]["loss"],
            "image_size": image_size, "train_images": len(train_loader.dataset),
            "val_images": len(val_loader.dataset), "test_images": len(test_loader.dataset),
        })

        start_time = time.time()
        for epoch in range(cfg["training"]["epochs"]):
            model.train()
            train_losses = []
            for images, labels, _ in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = sum(train_losses) / len(train_losses)

            model.eval()
            val_losses = []
            with torch.no_grad():
                for images, labels, _ in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    val_losses.append(loss_fn(logits, labels).item())
            val_loss = sum(val_losses) / len(val_losses)

            y_true, y_prob = run_inference(model, val_loader, device)
            val_eval = evaluate_predictions(y_true, y_prob, LABELS)

            mlflow.log_metrics({
                "train_loss": train_loss, "val_loss": val_loss,
                "val_macro_roc_auc": val_eval.macro_roc_auc, "val_macro_pr_auc": val_eval.macro_pr_auc,
                "val_macro_f1": val_eval.macro_f1,
            }, step=epoch)

            history.append({
                "epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
                "val_macro_roc_auc": round(val_eval.macro_roc_auc, 4),
            })
            if not quiet:
                print(f"[{run_name}] epoch {epoch+1}/{cfg['training']['epochs']} "
                      f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                      f"val_ROC-AUC={val_eval.macro_roc_auc:.4f}")

            if val_eval.macro_roc_auc > best_val_auc:
                best_val_auc = val_eval.macro_roc_auc
                epochs_without_improvement = 0
                torch.save({
                    "model_state_dict": model.state_dict(), "model_name": model_name,
                    "num_labels": len(LABELS), "label_names": LABELS, "image_size": image_size,
                    "val_macro_roc_auc": best_val_auc,
                }, ckpt_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg["training"]["early_stopping_patience"]:
                    if not quiet:
                        print(f"[{run_name}] early stopping at epoch {epoch+1}")
                    break

        training_time = time.time() - start_time

        # --- final test-set evaluation using the BEST checkpoint ---
        best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        y_true, y_prob = run_inference(model, test_loader, device)
        test_eval = evaluate_predictions(y_true, y_prob, LABELS)

        sample_input = next(iter(test_loader))[0][:1]
        latency_ms = measure_inference_latency(model, sample_input, device)
        test_eval.mean_inference_latency_ms = latency_ms

        mlflow.log_metrics({
            "test_macro_roc_auc": test_eval.macro_roc_auc, "test_macro_pr_auc": test_eval.macro_pr_auc,
            "test_macro_f1": test_eval.macro_f1, "test_macro_sensitivity": test_eval.macro_sensitivity,
            "test_macro_specificity": test_eval.macro_specificity, "inference_latency_ms": latency_ms,
            "training_time_sec": training_time,
        })
        mlflow.log_artifact(str(ckpt_path))

        result = {
            "run_name": run_name, "model": model_name, "loss": cfg["training"]["loss"],
            "training_time_sec": round(training_time, 1), "history": history,
            "test_metrics": test_eval.to_dict(), "checkpoint": str(ckpt_path),
        }

        report_path = checkpoint_dir / f"{run_name}_metrics.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        if not quiet:
            print(f"\n[{run_name}] TEST macro ROC-AUC={test_eval.macro_roc_auc:.4f} "
                  f"PR-AUC={test_eval.macro_pr_auc:.4f} F1={test_eval.macro_f1:.4f} "
                  f"latency={latency_ms:.1f}ms")
            print(f"[{run_name}] metrics written to {report_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Train a MedVision model")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_one_config(cfg)


if __name__ == "__main__":
    main()
