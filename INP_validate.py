import argparse
import csv
import math
import os
import re
import time
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

from dataset import get_data_transforms
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from INP_Former_Single_Class import (
    FolderLabelDataset,
    build_test_dataset,
    ensure_csv_headers,
    evaluate_image_level,
    fmt_float,
    fmt_int_or_nan,
    fmt_pct,
    resolve_item_list,
)
from utils import get_logger, setup_seed


class DirectLabelFolderDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.samples = []

        for subdir, label in (("good", 0), ("bad", 1), ("ng", 1), ("anomaly", 1)):
            folder = os.path.join(root, subdir)
            if not os.path.isdir(folder):
                continue
            for current_root, _, files in os.walk(folder):
                for filename in files:
                    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                        self.samples.append((os.path.join(current_root, filename), label))

        self.samples.sort()
        if not self.samples:
            raise RuntimeError(f"Khong tim thay anh test trong {root}. Can co thu muc good va bad|ng|anomaly.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long), image_path


def build_model(args, device):
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(args.encoder)
    if "small" in args.encoder:
        embed_dim, num_heads = 384, 6
    elif "base" in args.encoder:
        embed_dim, num_heads = 768, 12
    elif "large" in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError("Architecture not in small, base, large.")

    bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.0)])
    inp = nn.ParameterList([nn.Parameter(torch.randn(args.INP_num, embed_dim)) for _ in range(1)])
    inp_extractor = nn.ModuleList([
        Aggregation_Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
        )
        for _ in range(1)
    ])
    inp_guided_decoder = nn.ModuleList([
        Prototype_Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
        )
        for _ in range(8)
    ])

    return INP_Former(
        encoder=encoder,
        bottleneck=bottleneck,
        aggregation=inp_extractor,
        decoder=inp_guided_decoder,
        target_layers=target_layers,
        remove_class_token=True,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        prototype_token=inp,
    ).to(device)


def build_validate_output_paths(output_root):
    validation_log_path = os.path.join(output_root, "validation_log.csv")
    prediction_log_path = os.path.join(output_root, "prediction_log.csv")
    validation_headers = [
        "checkpoint_name",
        "epoch",
        "eval_time_sec",
        "image_auroc",
        "image_ap",
        "image_f1",
        "separation",
        "min_decision_margin",
        "max_ok_score",
        "min_ng_score",
        "threshold_f1_opt",
        "false_reject_at_f1_opt",
        "escape_at_f1_opt",
        "precision_at_f1_opt",
        "recall_at_f1_opt",
        "threshold_no_escape",
        "false_reject_at_no_escape",
        "escape_at_no_escape",
        "threshold_no_false_reject",
        "false_reject_at_no_false_reject",
        "escape_at_no_false_reject",
        "perfect_threshold_exists",
        "perfect_threshold_low",
        "perfect_threshold_high",
        "perfect_threshold_mid",
        "checkpoint_path",
    ]
    prediction_headers = [
        "checkpoint_name",
        "epoch",
        "image_index",
        "image_path",
        "label",
        "score",
    ]
    ensure_csv_headers(validation_log_path, validation_headers)
    ensure_csv_headers(prediction_log_path, prediction_headers)
    return validation_log_path, prediction_log_path, validation_headers, prediction_headers


def append_prediction_log(prediction_log_path, prediction_headers, checkpoint_name, epoch_value, image_paths, labels, scores):
    with open(prediction_log_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=prediction_headers)
        for index, (image_path, label, score) in enumerate(zip(image_paths, labels, scores)):
            writer.writerow({
                "checkpoint_name": checkpoint_name,
                "epoch": epoch_value,
                "image_index": index,
                "image_path": image_path,
                "label": int(label),
                "score": float(score),
            })


def append_validation_log(validation_log_path, validation_headers, checkpoint_name, epoch_value, eval_time_sec, metrics, checkpoint_path):
    with open(validation_log_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=validation_headers)
        writer.writerow({
            "checkpoint_name": checkpoint_name,
            "epoch": epoch_value,
            "eval_time_sec": eval_time_sec,
            "image_auroc": metrics["image_auroc"],
            "image_ap": metrics["image_ap"],
            "image_f1": metrics["image_f1"],
            "separation": metrics["separation"],
            "min_decision_margin": metrics["min_decision_margin"],
            "max_ok_score": metrics["max_ok_score"],
            "min_ng_score": metrics["min_ng_score"],
            "threshold_f1_opt": metrics["threshold_f1_opt"],
            "false_reject_at_f1_opt": metrics["false_reject_at_f1_opt"],
            "escape_at_f1_opt": metrics["escape_at_f1_opt"],
            "precision_at_f1_opt": metrics["precision_at_f1_opt"],
            "recall_at_f1_opt": metrics["recall_at_f1_opt"],
            "threshold_no_escape": metrics["threshold_no_escape"],
            "false_reject_at_no_escape": metrics["false_reject_at_no_escape"],
            "escape_at_no_escape": metrics["escape_at_no_escape"],
            "threshold_no_false_reject": metrics["threshold_no_false_reject"],
            "false_reject_at_no_false_reject": metrics["false_reject_at_no_false_reject"],
            "escape_at_no_false_reject": metrics["escape_at_no_false_reject"],
            "perfect_threshold_exists": int(bool(metrics["perfect_threshold_exists"])),
            "perfect_threshold_low": metrics["perfect_threshold_low"],
            "perfect_threshold_high": metrics["perfect_threshold_high"],
            "perfect_threshold_mid": metrics["perfect_threshold_mid"],
            "checkpoint_path": checkpoint_path,
        })


def append_summary_log(summary_path, item, checkpoint_name, epoch_value, metrics, checkpoint_path):
    summary_headers = [
        "item",
        "checkpoint_name",
        "epoch",
        "image_auroc",
        "image_ap",
        "image_f1",
        "separation",
        "min_decision_margin",
        "max_ok_score",
        "min_ng_score",
        "checkpoint_path",
    ]
    ensure_csv_headers(summary_path, summary_headers)
    with open(summary_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary_headers)
        writer.writerow({
            "item": item,
            "checkpoint_name": checkpoint_name,
            "epoch": epoch_value,
            "image_auroc": metrics["image_auroc"],
            "image_ap": metrics["image_ap"],
            "image_f1": metrics["image_f1"],
            "separation": metrics["separation"],
            "min_decision_margin": metrics["min_decision_margin"],
            "max_ok_score": metrics["max_ok_score"],
            "min_ng_score": metrics["min_ng_score"],
            "checkpoint_path": checkpoint_path,
        })


def parse_epoch_from_name(checkpoint_name):
    match = re.fullmatch(r"ckpt_epoch_(\d+)\.pth", checkpoint_name)
    if match:
        return int(match.group(1))
    return checkpoint_name.replace(".pth", "")


def list_epoch_checkpoints(checkpoint_dir):
    checkpoint_items = []
    for name in os.listdir(checkpoint_dir):
        match = re.fullmatch(r"ckpt_epoch_(\d+)\.pth", name)
        if match:
            checkpoint_items.append((int(match.group(1)), name))
    checkpoint_items.sort(key=lambda item: item[0])
    return checkpoint_items


def select_checkpoints(args, checkpoint_dir):
    if args.all_epochs:
        epoch_items = list_epoch_checkpoints(checkpoint_dir)
        if not epoch_items:
            raise FileNotFoundError(f"Khong tim thay checkpoint epoch nao trong {checkpoint_dir}")
        return [(epoch_value, name, os.path.join(checkpoint_dir, name)) for epoch_value, name in epoch_items]

    if args.epoch is not None:
        checkpoint_name = f"ckpt_epoch_{args.epoch:04d}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Khong tim thay checkpoint {checkpoint_path}")
        return [(args.epoch, checkpoint_name, checkpoint_path)]

    checkpoint_name = args.checkpoint_name
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Khong tim thay checkpoint {checkpoint_path}")
    return [(parse_epoch_from_name(checkpoint_name), checkpoint_name, checkpoint_path)]


def validate_one_item(args, device, summary_path, validation_root):
    setup_seed(1)
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)
    if args.test_dir:
        test_data = DirectLabelFolderDataset(args.test_dir, data_transform)
    else:
        test_data = build_test_dataset(args, data_transform, gt_transform)
        if not isinstance(test_data, FolderLabelDataset):
            raise RuntimeError(
                "Script validate rieng nay dang ho tro bo test kieu folder label-only. "
                "Hay dung --test_dir tro thang vao thu muc chua good va bad|ng|anomaly."
            )

    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    experiment_root = os.path.join(args.save_dir, args.save_name)
    checkpoint_dir = args.checkpoint_dir or os.path.join(experiment_root, args.item)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Khong tim thay thu muc checkpoint: {checkpoint_dir}")

    item_name_for_output = args.output_item_name or args.item
    if not item_name_for_output:
        item_name_for_output = os.path.basename(os.path.normpath(checkpoint_dir))
    item_output_dir = os.path.join(validation_root, item_name_for_output)
    os.makedirs(item_output_dir, exist_ok=True)
    validation_log_path, prediction_log_path, validation_headers, prediction_headers = build_validate_output_paths(item_output_dir)

    logger = get_logger(f"{args.save_name}_{item_name_for_output}_{args.validation_name}", item_output_dir)
    print_fn = logger.info
    print_fn(f"item={item_name_for_output}")
    print_fn(f"checkpoint_dir={checkpoint_dir}")
    if args.test_dir:
        print_fn(f"test_dir={args.test_dir}")
    print_fn(f"validation_output={item_output_dir}")
    print_fn(f"test_samples={len(test_data)}")

    checkpoint_specs = select_checkpoints(args, checkpoint_dir)
    model = build_model(args, device)

    results = []
    for epoch_value, checkpoint_name, checkpoint_path in checkpoint_specs:
        eval_start = time.time()
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        metrics, scores, labels, image_paths = evaluate_image_level(model, test_dataloader, device, max_ratio=0.01)
        eval_duration = time.time() - eval_start

        append_validation_log(
            validation_log_path,
            validation_headers,
            checkpoint_name,
            epoch_value,
            eval_duration,
            metrics,
            checkpoint_path,
        )
        append_prediction_log(
            prediction_log_path,
            prediction_headers,
            checkpoint_name,
            epoch_value,
            image_paths,
            labels,
            scores,
        )
        append_summary_log(summary_path, item_name_for_output, checkpoint_name, epoch_value, metrics, checkpoint_path)

        print_fn(
            f"ckpt:{checkpoint_name}"
            f" eval:{eval_duration:.2f}s"
            f" IAUC:{fmt_pct(metrics['image_auroc'])}"
            f" IAP:{fmt_pct(metrics['image_ap'])}"
            f" IF1:{fmt_pct(metrics['image_f1'])}"
            f" SEP:{fmt_float(metrics['separation'])}"
            f" MDM:{fmt_float(metrics['min_decision_margin'])}"
            f" maxOK:{fmt_float(metrics['max_ok_score'])}"
            f" minNG:{fmt_float(metrics['min_ng_score'])}"
            f" TH_F1:{fmt_float(metrics['threshold_f1_opt'])}"
            f" FR@F1:{fmt_int_or_nan(metrics['false_reject_at_f1_opt'])}"
            f" ESC@F1:{fmt_int_or_nan(metrics['escape_at_f1_opt'])}"
            f" TH_noESC:{fmt_float(metrics['threshold_no_escape'])}"
            f" FR@noESC:{fmt_int_or_nan(metrics['false_reject_at_no_escape'])}"
            f" TH_noFR:{fmt_float(metrics['threshold_no_false_reject'])}"
            f" ESC@noFR:{fmt_int_or_nan(metrics['escape_at_no_false_reject'])}"
            f" Perfect:{int(bool(metrics['perfect_threshold_exists']))}"
        )
        results.append((checkpoint_name, epoch_value, metrics))

    return results


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    parser = argparse.ArgumentParser(description="Validate lai checkpoint cua INP-Former-Single-Class")
    parser.add_argument("--dataset", type=str, default="MVTec-AD")
    parser.add_argument("--data_path", type=str, default="./mvtec_anomaly_detection")
    parser.add_argument("--save_dir", type=str, default="./saved_results")
    parser.add_argument("--save_name", type=str, default="INP-Former-Single-Class")
    parser.add_argument("--encoder", type=str, default="dinov2reg_vit_small_14")
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--crop_size", type=int, default=448)
    parser.add_argument("--INP_num", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--item_list", type=str, default="")
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--test_dir", type=str, default="")
    parser.add_argument("--output_item_name", type=str, default="")
    parser.add_argument("--checkpoint_name", type=str, default="ckpt_best.pth")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--all_epochs", action="store_true")
    parser.add_argument("--validation_name", type=str, default="")

    args = parser.parse_args()
    args.save_name = (
        args.save_name
        + f"_dataset={args.dataset}_Encoder={args.encoder}_Resize={args.input_size}_Crop={args.crop_size}_INP_num={args.INP_num}"
    )
    if args.checkpoint_dir and args.test_dir:
        args.item_list = [args.output_item_name.strip()] if args.output_item_name.strip() else [""]
    else:
        args.item_list = resolve_item_list(args)
    if args.validation_name.strip():
        args.validation_name = args.validation_name.strip()
    else:
        args.validation_name = datetime.now().strftime("reval_%Y%m%d_%H%M%S")

    if args.all_epochs and args.epoch is not None:
        raise ValueError("Chi chon mot trong hai: --all_epochs hoac --epoch")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    experiment_root = os.path.join(args.save_dir, args.save_name)
    validation_root = os.path.join(experiment_root, "revalidation", args.validation_name)
    os.makedirs(validation_root, exist_ok=True)
    summary_path = os.path.join(validation_root, "summary.csv")

    result_list = []
    for item in args.item_list:
        args.item = item
        item_results = validate_one_item(args, device, summary_path, validation_root)
        if item_results:
            last_checkpoint_name, last_epoch, last_metrics = item_results[-1]
            result_list.append([
                item,
                last_checkpoint_name,
                last_epoch,
                last_metrics.get("image_auroc", math.nan),
                last_metrics.get("image_ap", math.nan),
                last_metrics.get("image_f1", math.nan),
                last_metrics.get("separation", math.nan),
                last_metrics.get("min_decision_margin", math.nan),
            ])

    print(result_list)
    if result_list:
        mean_auroc_sp = np.mean([result[3] for result in result_list])
        mean_ap_sp = np.mean([result[4] for result in result_list])
        mean_f1_sp = np.mean([result[5] for result in result_list])
        root_logger = get_logger(f"{args.save_name}_{args.validation_name}_summary", validation_root)
        root_logger.info(result_list)
        root_logger.info(f"Mean: I-Auroc:{mean_auroc_sp:.4f}, I-AP:{mean_ap_sp:.4f}, I-F1:{mean_f1_sp:.4f}")
