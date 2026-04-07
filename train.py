from __future__ import annotations

import argparse
import os
from typing import List, Tuple

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import models, transforms
    from datasets_core import ClassificationDataset, FeatureDataset
    _IMPORT_ERROR = None
except ModuleNotFoundError as import_error:
    torch = None
    nn = None
    DataLoader = None
    models = None
    transforms = None
    ClassificationDataset = None
    FeatureDataset = None
    _IMPORT_ERROR = import_error


def multiclass_accuracy_and_macro_f1(preds, labels, num_classes: int) -> Tuple[float, float]:
    correct = (preds == labels).sum().item()
    total = labels.numel()
    accuracy = correct / max(1, total)

    f1_scores: List[float] = []
    for class_id in range(num_classes):
        pred_pos = preds == class_id
        true_pos = labels == class_id
        tp = (pred_pos & true_pos).sum().item()
        fp = (pred_pos & (~true_pos)).sum().item()
        fn = ((~pred_pos) & true_pos).sum().item()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / max(1, len(f1_scores))
    return accuracy, macro_f1


def multilabel_accuracy_and_macro_f1(preds, labels, num_labels: int) -> Tuple[float, float]:
    element_acc = (preds == labels).float().mean().item()

    f1_scores: List[float] = []
    for label_id in range(num_labels):
        pred_pos = preds[:, label_id] == 1
        true_pos = labels[:, label_id] == 1
        tp = (pred_pos & true_pos).sum().item()
        fp = (pred_pos & (~true_pos)).sum().item()
        fn = ((~pred_pos) & true_pos).sum().item()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / max(1, len(f1_scores))
    return element_acc, macro_f1


def build_cls_model(pretrained: bool):
    weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_v2_s(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 6)
    return model


def build_feat_model(pretrained: bool):
    weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b4(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 4)
    return model


def evaluate_cls(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    return multiclass_accuracy_and_macro_f1(preds, labels, num_classes=6)


def evaluate_feat(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).int()
            all_preds.append(preds.cpu())
            all_labels.append(labels.int().cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    return multilabel_accuracy_and_macro_f1(preds, labels, num_labels=4)


def make_dataloaders(args):
    train_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    if args.task == "cls":
        train_ds = ClassificationDataset(args.train_dir, transform=train_tf)
        val_ds = ClassificationDataset(args.val_dir, transform=val_tf)
    else:
        train_ds = FeatureDataset(args.train_dir, transform=train_tf)
        val_ds = FeatureDataset(args.val_dir, transform=val_tf)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("数据集为空，请检查 --train-dir 和 --val-dir 路径。")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["cls", "feat"], required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--save-dir", default="outputs")
    args = parser.parse_args()

    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "运行训练需要先安装依赖，请执行: pip install -r requirements.txt"
        ) from _IMPORT_ERROR

    save_dir = os.path.join(args.save_dir, args.task)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_dataloaders(args)

    if args.task == "cls":
        model = build_cls_model(pretrained=args.pretrained).to(device)
        criterion = nn.CrossEntropyLoss()
        best_name = "best_cls_model.pth"
    else:
        model = build_feat_model(pretrained=args.pretrained).to(device)
        criterion = nn.BCEWithLogitsLoss()
        best_name = "best_feat_model.pth"

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = -1.0
    best_path = os.path.join(save_dir, best_name)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels if args.task == "feat" else labels.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if args.task == "cls":
            val_acc, val_f1 = evaluate_cls(model, val_loader, device)
        else:
            val_acc, val_f1 = evaluate_feat(model, val_loader, device)

        score = 0.5 * val_acc + 0.5 * val_f1
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"task={args.task} epoch={epoch} loss={avg_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)

    print(f"best_score={best_score:.4f}")
    print(f"saved={best_path}")


if __name__ == "__main__":
    main()
