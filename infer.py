import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from datasets_core import ClassificationDataset, FeatureDataset


def load_cls_model(weights_path: str, device):
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 6)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_feat_model(weights_path: str, device):
    model = models.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 4)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def write_lines(path: str, lines):
    with open(path, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls-data", default="data/乳腺分类训练数据集/test")
    parser.add_argument("--feat-data", default="data/乳腺特征训练数据集/test")
    parser.add_argument("--cls-weights", required=True)
    parser.add_argument("--feat-weights", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", default="outputs/predict")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    cls_loader = DataLoader(
        ClassificationDataset(args.cls_data, transform=tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    feat_loader = DataLoader(
        FeatureDataset(args.feat_data, transform=tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    cls_model = load_cls_model(args.cls_weights, device)
    feat_model = load_feat_model(args.feat_weights, device)

    cls_preds = []
    feat_preds = []

    with torch.no_grad():
        for images, _ in cls_loader:
            images = images.to(device)
            logits = cls_model(images)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            cls_preds.extend(preds)

        for images, _ in feat_loader:
            images = images.to(device)
            logits = feat_model(images)
            preds = (torch.sigmoid(logits) > 0.5).int().cpu().tolist()
            feat_preds.extend(preds)

    write_lines(os.path.join(args.out_dir, "cls_predictions.txt"), [str(v) for v in cls_preds])
    write_lines(
        os.path.join(args.out_dir, "feat_predictions.txt"),
        [" ".join(str(x) for x in row) for row in feat_preds],
    )

    print(f"saved={args.out_dir}")


if __name__ == "__main__":
    main()
