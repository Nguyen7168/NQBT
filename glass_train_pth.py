#!/usr/bin/env python3
"""Self-contained GLASS training script for directory-based datasets."""

import argparse
import csv
import glob
import json
import logging
import math
import os
import random
import shutil
import time
import warnings
from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import measure
from sklearn import metrics as sklearn_metrics
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms

IMAGE_EXTENSIONS: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
IMAGENET_MEAN: Sequence[float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Sequence[float] = (0.229, 0.224, 0.225)


def list_images(root: Optional[str]) -> List[str]:
    if not root:
        return []
    paths: List[str] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.iglob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    return sorted(paths)


_BACKBONES: Dict[str, Callable[[], torch.nn.Module]] = {
    "alexnet": lambda: models.alexnet(pretrained=True),
    "resnet18": lambda: models.resnet18(pretrained=True),
    "resnet50": lambda: models.resnet50(pretrained=True),
    "resnet101": lambda: models.resnet101(pretrained=True),
    "resnext101": lambda: models.resnext101_32x8d(pretrained=True),
    "resnet200": lambda: timm.create_model("resnet200", pretrained=True),
    "resnest50": lambda: timm.create_model("resnest50d_4s2x40d", pretrained=True),
    "resnetv2_50_bit": lambda: timm.create_model("resnetv2_50x3_bitm", pretrained=True),
    "resnetv2_50_21k": lambda: timm.create_model("resnetv2_50x3_bitm_in21k", pretrained=True),
    "resnetv2_101_bit": lambda: timm.create_model("resnetv2_101x3_bitm", pretrained=True),
    "resnetv2_101_21k": lambda: timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=True),
    "resnetv2_152_bit": lambda: timm.create_model("resnetv2_152x4_bitm", pretrained=True),
    "resnetv2_152_21k": lambda: timm.create_model("resnetv2_152x4_bitm_in21k", pretrained=True),
    "resnetv2_152_384": lambda: timm.create_model("resnetv2_152x2_bit_teacher_384", pretrained=True),
    "resnetv2_101": lambda: timm.create_model("resnetv2_101", pretrained=True),
    "vgg11": lambda: models.vgg11(pretrained=True),
    "vgg19": lambda: models.vgg19(pretrained=True),
    "vgg19_bn": lambda: models.vgg19_bn(pretrained=True),
    "wideresnet50": lambda: models.wide_resnet50_2(pretrained=True),
    "wideresnet101": lambda: models.wide_resnet101_2(pretrained=True),
    "mnasnet_100": lambda: timm.create_model("mnasnet_100", pretrained=True),
    "mnasnet_a1": lambda: timm.create_model("mnasnet_a1", pretrained=True),
    "mnasnet_b1": lambda: timm.create_model("mnasnet_b1", pretrained=True),
    "densenet121": lambda: timm.create_model("densenet121", pretrained=True),
    "densenet201": lambda: timm.create_model("densenet201", pretrained=True),
    "inception_v4": lambda: timm.create_model("inception_v4", pretrained=True),
    "vit_small": lambda: timm.create_model("vit_small_patch16_224", pretrained=True),
    "vit_base": lambda: timm.create_model("vit_base_patch16_224", pretrained=True),
    "vit_large": lambda: timm.create_model("vit_large_patch16_224", pretrained=True),
    "vit_r50": lambda: timm.create_model("vit_large_r50_s32_224", pretrained=True),
    "vit_deit_base": lambda: timm.create_model("deit_base_patch16_224", pretrained=True),
    "vit_deit_distilled": lambda: timm.create_model("deit_base_distilled_patch16_224", pretrained=True),
    "vit_swin_base": lambda: timm.create_model("swin_base_patch4_window7_224", pretrained=True),
    "vit_swin_large": lambda: timm.create_model("swin_large_patch4_window7_224", pretrained=True),
    "efficientnet_b7": lambda: timm.create_model("tf_efficientnet_b7", pretrained=True),
    "efficientnet_b5": lambda: timm.create_model("tf_efficientnet_b5", pretrained=True),
    "efficientnet_b3": lambda: timm.create_model("tf_efficientnet_b3", pretrained=True),
    "efficientnet_b1": lambda: timm.create_model("tf_efficientnet_b1", pretrained=True),
    "efficientnetv2_m": lambda: timm.create_model("tf_efficientnetv2_m", pretrained=True),
    "efficientnetv2_l": lambda: timm.create_model("tf_efficientnetv2_l", pretrained=True),
    "efficientnet_b3a": lambda: timm.create_model("efficientnet_b3a", pretrained=True),
}


def load_backbone(name: str) -> torch.nn.Module:
    if name not in _BACKBONES:
        raise KeyError(f"Unknown backbone '{name}'. Available options: {sorted(_BACKBONES)}")
    return _BACKBONES[name]()


class FocalLoss(torch.nn.Module):
    """Focal loss with optional label smoothing."""

    def __init__(
        self,
        apply_nonlin=None,
        alpha=None,
        gamma: float = 2.0,
        balance_index: int = 0,
        smooth: float = 1e-5,
        size_average: bool = True,
    ) -> None:
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average
        if self.smooth < 0 or self.smooth > 1.0:
            raise ValueError("smooth value should be in [0,1]")

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        if target.dim() > 1:
            target = torch.squeeze(target, 1)
        target = target.view(-1, 1)

        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_class, 1, device=logit.device)
        elif isinstance(alpha, (list, np.ndarray)):
            alpha = torch.tensor(alpha, dtype=torch.float32, device=logit.device).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha_t = torch.ones(num_class, 1, device=logit.device)
            alpha_t = alpha_t * (1 - alpha)
            alpha_t[self.balance_index] = alpha
            alpha = alpha_t
        else:
            raise TypeError("Unsupported alpha type")

        idx = target.long().cpu()
        one_hot_key = torch.zeros(target.size(0), num_class)
        one_hot_key = one_hot_key.scatter_(1, idx, 1).to(logit.device)
        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()
        alpha = alpha[idx].to(logit.device).squeeze()
        loss = -1 * alpha * torch.pow((1 - pt), self.gamma) * logpt
        return loss.mean() if self.size_average else loss.sum()


def lerp_np(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (y - x) * w + x


def rand_perlin_2d_np(
    shape: Tuple[int, int],
    res: Tuple[int, int],
    fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3,
) -> np.ndarray:
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    def tile_grads(slice1, slice2):
        return np.repeat(
            np.repeat(gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]], d[0], axis=0),
            d[1],
            axis=1,
        )

    def dot(grad, shift):
        return (
            np.stack(
                (
                    grid[: shape[0], : shape[1], 0] + shift[0],
                    grid[: shape[0], : shape[1], 1] + shift[1],
                ),
                axis=-1,
            )
            * grad[: shape[0], : shape[1]]
        ).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])


def generate_thr(img_shape: Sequence[int], min_scale: int = 0, max_scale: int = 4) -> np.ndarray:
    perlin_scalex = 2 ** np.random.randint(min_scale, max_scale)
    perlin_scaley = 2 ** np.random.randint(min_scale, max_scale)
    perlin_noise_np = rand_perlin_2d_np((img_shape[1], img_shape[2]), (perlin_scalex, perlin_scaley))
    threshold = 0.5
    perlin_noise_np = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])(image=perlin_noise_np)
    perlin_thr = np.where(perlin_noise_np > threshold, np.ones_like(perlin_noise_np), np.zeros_like(perlin_noise_np))
    return perlin_thr


def perlin_mask(
    img_shape: Sequence[int],
    feat_size: int,
    min_scale: int,
    max_scale: int,
    mask_fg: torch.Tensor,
    flag: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((feat_size, feat_size))
    mask_l = np.zeros_like(mask)
    while np.max(mask) == 0:
        perlin_thr_1 = generate_thr(img_shape, min_scale, max_scale)
        perlin_thr_2 = generate_thr(img_shape, min_scale, max_scale)
        temp = torch.rand(1).item()
        if temp > 2 / 3:
            perlin_thr = np.where(perlin_thr_1 + perlin_thr_2 > 0, 1.0, 0.0)
        elif temp > 1 / 3:
            perlin_thr = perlin_thr_1 * perlin_thr_2
        else:
            perlin_thr = perlin_thr_1
        perlin_thr = torch.from_numpy(perlin_thr)
        perlin_thr_fg = perlin_thr * mask_fg
        down_ratio_y = int(img_shape[1] / feat_size)
        down_ratio_x = int(img_shape[2] / feat_size)
        mask_ = perlin_thr_fg
        mask = (
            torch.nn.functional.max_pool2d(
                perlin_thr_fg.unsqueeze(0).unsqueeze(0), (down_ratio_y, down_ratio_x)
            )
            .float()
            .numpy()[0, 0]
        )
        mask_l = mask_.numpy()
    if flag == 0:
        return mask, mask_l
    return mask, mask_l


def distribution_judge(img: np.ndarray, name: str) -> int:
    img_ = cv2.resize(img, (289, 289))
    img_gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.blur(img_gray, (39, 39))

    dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    magnitude[magnitude > 170] = 255
    magnitude[magnitude <= 170] = 0

    height, width = magnitude.shape
    center = (height // 2, width // 2)
    y_indices, x_indices = np.where(magnitude == 255)
    y_all, x_all = np.indices((2 * height, 2 * width))

    l1_dist_x = np.abs(x_indices - center[1])
    l1_dist_y = np.abs(y_indices - center[0])
    dist = np.sqrt((x_indices - center[1]) ** 2 + (y_indices - center[0]) ** 2)
    l2_dist_all = np.sqrt((x_all - center[1]) ** 2 + (y_all - center[0]) ** 2)

    side_x = np.max(l1_dist_x) if len(l1_dist_x) else 0
    side_y = np.max(l1_dist_y) if len(l1_dist_y) else 0
    radius = np.max(dist) if len(dist) else 0
    points_num = len(dist)

    l1_density = points_num / (4 * max(side_x, 1) * max(side_y, 1))
    l2_density = points_num / (np.sum(l2_dist_all <= radius) + 1e-10)
    flag = 1 if (l1_density > 0.21 or l2_density > 0.21) and radius > 12 and points_num > 60 else 0
    dist_type = "HyperSphere" if flag == 1 else "Manifold"
    print(f"Distribution: {flag} / {dist_type}.")

    output_path = os.path.join("./results/judge/fft", str(flag), f"{name}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    magnitude_rgb = np.repeat(magnitude[..., None], 3, axis=-1)
    img_up = np.hstack([img_, magnitude_rgb])
    cv2.imwrite(output_path, img_up)
    return flag


def del_remake_dir(path: str, del_flag: bool = True) -> None:
    if os.path.exists(path):
        if del_flag:
            shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)


def torch_format_2_numpy_img(img: np.ndarray) -> np.ndarray:
    if img.shape[0] == 3:
        img = img.transpose([1, 2, 0])
        img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        img = img[:, :, [2, 1, 0]]
        img = (img * 255).astype("uint8")
    else:
        img = img.transpose([1, 2, 0])
        img = np.repeat(img, 3, axis=-1)
        img = (img * 255).astype("uint8")
    return img


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights: Iterable[float],
    anomaly_ground_truth_labels: Iterable[int],
    path: str = "training",
) -> Dict[str, float]:
    labels = np.asarray(list(anomaly_ground_truth_labels))
    scores = np.asarray(list(anomaly_prediction_weights))
    if np.unique(labels).size <= 1:
        warnings.warn(
            "Only one class present in image-level ground truth labels; returning default retrieval metrics.",
            RuntimeWarning,
        )
        auroc = 0.0
        ap = 0.0
    else:
        auroc = sklearn_metrics.roc_auc_score(labels, scores)
        ap = 0.0 if path == "training" else sklearn_metrics.average_precision_score(labels, scores)
    return {"auroc": float(auroc), "ap": float(ap)}


def compute_pixelwise_retrieval_metrics(
    anomaly_segmentations: Iterable[np.ndarray],
    ground_truth_masks: Iterable[np.ndarray],
    path: str = "training",
) -> Dict[str, float]:
    anomaly_segmentations = np.stack(list(anomaly_segmentations))
    ground_truth_masks = np.stack(list(ground_truth_masks))
    flat_seg = anomaly_segmentations.ravel()
    flat_gt = ground_truth_masks.ravel().astype(int)
    if np.unique(flat_gt).size <= 1:
        warnings.warn(
            "Only one class present in pixel-level ground truth masks; returning default retrieval metrics.",
            RuntimeWarning,
        )
        auroc = 0.0
        ap = 0.0
    else:
        auroc = sklearn_metrics.roc_auc_score(flat_gt, flat_seg)
        ap = 0.0 if path == "training" else sklearn_metrics.average_precision_score(flat_gt, flat_seg)
    return {"auroc": float(auroc), "ap": float(ap)}


def compute_pro(masks: np.ndarray, amaps: np.ndarray, num_th: int = 200) -> float:
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / max(num_th, 1)
    if delta <= 0:
        return 0.0
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros: List[float] = []
        for binary_amap, mask in zip(binary_amaps, masks):
            binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / (inverse_masks.sum() + 1e-10)
        df = pd.concat([df, pd.DataFrame({"pro": np.mean(pros) if pros else 0.0, "fpr": fpr, "threshold": th}, index=[0])])

    if len(df) == 0:
        return 0.0
    df = df[df["fpr"] < 0.3]
    if len(df) == 0:
        return 0.0
    df["fpr"] = (df["fpr"] - df["fpr"].min()) / (df["fpr"].max() - df["fpr"].min() + 1e-10)
    pro_auc = sklearn_metrics.auc(df["fpr"], df["pro"])
    return float(pro_auc)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim: int) -> None:
        super().__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims: Sequence[int], output_dim: int) -> None:
        super().__init__()
        self.preprocessing_modules = torch.nn.ModuleList(
            [MeanMapper(output_dim) for _ in input_dims]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.stack([module(feature) for module, feature in zip(self.preprocessing_modules, features)], dim=1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim: int) -> None:
        super().__init__()
        self.target_dim = target_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class RescaleSegmentor:
    def __init__(self, device: torch.device, target_size: Sequence[int]) -> None:
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores: torch.Tensor) -> List[np.ndarray]:
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(_scores, size=self.target_size, mode="bilinear", align_corners=False)
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()
        return [ndimage.gaussian_filter(patch_score, sigma=self.smoothing) for patch_score in patch_scores]


class LastLayerToExtractReachedException(Exception):
    pass


class ForwardHook:
    def __init__(self, hook_dict: Dict[str, torch.Tensor], layer_name: str, last_layer_to_extract: str) -> None:
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = layer_name == last_layer_to_extract

    def __call__(self, module, inp, output):  # type: ignore[override]
        self.hook_dict[self.layer_name] = output
        return None


class NetworkFeatureAggregator(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        layers_to_extract_from: Sequence[str],
        device: torch.device,
        train_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.layers_to_extract_from = list(layers_to_extract_from)
        self.backbone = backbone
        self.device = device
        self.train_backbone = train_backbone
        if not hasattr(backbone, "hook_handles"):
            backbone.hook_handles = []  # type: ignore[attr-defined]
        for handle in backbone.hook_handles:  # type: ignore[attr-defined]
            handle.remove()
        self.outputs: Dict[str, torch.Tensor] = {}
        for extract_layer in self.layers_to_extract_from:
            self.register_hook(extract_layer)
        self.to(self.device)

    def forward(self, images: torch.Tensor, eval: bool = True) -> Dict[str, torch.Tensor]:
        self.outputs.clear()
        if self.train_backbone and not eval:
            self.backbone(images)
        else:
            with torch.no_grad():
                try:
                    _ = self.backbone(images)
                except LastLayerToExtractReachedException:
                    pass
        return self.outputs

    def feature_dimensions(self, input_shape: Sequence[int]) -> List[int]:
        dummy = torch.ones([1] + list(input_shape)).to(self.device)
        outputs = self(dummy)
        return [outputs[layer].shape[1] for layer in self.layers_to_extract_from]

    def register_hook(self, layer_name: str) -> None:
        module = self.find_module(self.backbone, layer_name)
        if module is None:
            raise ValueError(f"Module {layer_name} not found in the model")
        forward_hook = ForwardHook(self.outputs, layer_name, self.layers_to_extract_from[-1])
        if isinstance(module, torch.nn.Sequential):
            hook = module[-1].register_forward_hook(forward_hook)
        else:
            hook = module.register_forward_hook(forward_hook)
        self.backbone.hook_handles.append(hook)  # type: ignore[attr-defined]

    def find_module(self, model: torch.nn.Module, module_name: str) -> Optional[torch.nn.Module]:
        for name, module in model.named_modules():
            if name == module_name:
                return module
            if "." in module_name:
                father, child = module_name.split(".", 1)
                if name == father:
                    return self.find_module(module, child)
        return None


def init_weight(m: torch.nn.Module) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    if isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)


class Discriminator(torch.nn.Module):
    def __init__(self, in_planes: int, n_layers: int = 2, hidden: Optional[int] = None) -> None:
        super().__init__()
        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers - 1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module(
                f"block{i + 1}",
                torch.nn.Sequential(
                    torch.nn.Linear(_in, _hidden),
                    torch.nn.BatchNorm1d(_hidden),
                    torch.nn.LeakyReLU(0.2),
                ),
            )
        self.tail = torch.nn.Sequential(
            torch.nn.Linear(_hidden, 1, bias=False),
            torch.nn.Sigmoid(),
        )
        self.apply(init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        x = self.tail(x)
        return x


class Projection(torch.nn.Module):
    def __init__(self, in_planes: int, out_planes: Optional[int] = None, n_layers: int = 1, layer_type: int = 0) -> None:
        super().__init__()
        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc", torch.nn.Linear(_in, _out))
            if i < n_layers - 1 and layer_type > 1:
                self.layers.add_module(f"{i}relu", torch.nn.LeakyReLU(0.2))
        self.apply(init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PatchMaker:
    def __init__(self, patchsize: int, top_k: int = 0, stride: Optional[int] = None) -> None:
        self.patchsize = patchsize
        self.stride = stride or 1
        self.top_k = top_k

    def patchify(self, features: torch.Tensor, return_spatial_info: bool = False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1)
        unfolded_features = unfolder(features)
        number_of_total_patches: List[int] = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(*features.shape[:2], self.patchsize, self.patchsize, -1)
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x: torch.Tensor, batchsize: int) -> torch.Tensor:
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :, 0]
        x = torch.max(x, dim=1).values
        return x


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("glass_train")
LOGGER.setLevel(logging.INFO)


class TBWrapper:
    def __init__(self, log_dir: str) -> None:
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self) -> None:
        self.g_iter += 1


class GLASS(torch.nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.log_csv_path: str = ""
        self.log_csv_headers: List[str] = []

    def load(
        self,
        backbone: torch.nn.Module,
        layers_to_extract_from: Sequence[str],
        device: torch.device,
        input_shape: Sequence[int],
        pretrain_embed_dimension: int,
        target_embed_dimension: int,
        patchsize: int = 3,
        patchstride: int = 1,
        meta_epochs: int = 640,
        eval_epochs: int = 1,
        dsc_layers: int = 2,
        dsc_hidden: int = 1024,
        dsc_margin: float = 0.5,
        train_backbone: bool = False,
        pre_proj: int = 1,
        mining: int = 1,
        noise: float = 0.015,
        radius: float = 0.75,
        p: float = 0.5,
        lr: float = 0.0001,
        svd: int = 0,
        step: int = 20,
        limit: int = 392,
        **kwargs,
    ) -> None:
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device
        self.forward_modules = torch.nn.ModuleDict({})
        feature_aggregator = NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = Preprocessing(feature_dimensions, pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing
        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = Aggregator(target_dim=target_embed_dimension)
        preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.meta_epochs = meta_epochs
        self.lr = lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(
                self.forward_modules["feature_aggregator"].backbone.parameters(), lr
            )

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.Adam(self.pre_projection.parameters(), lr, weight_decay=1e-5)

        self.eval_epochs = eval_epochs
        self.dsc_layers = dsc_layers
        self.dsc_hidden = dsc_hidden
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.AdamW(self.discriminator.parameters(), lr=lr * 2)
        self.dsc_margin = dsc_margin

        self.c = torch.tensor(0.0)
        self.c_ = torch.tensor(0.0)
        self.p = p
        self.radius = radius
        self.mining = mining
        self.noise = noise
        self.svd = svd
        self.step = step
        self.limit = limit
        self.distribution = 0
        self.focal_loss = FocalLoss()

        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.anomaly_segmentor = RescaleSegmentor(device=self.device, target_size=input_shape[-2:])
        self.model_dir = ""
        self.dataset_name = ""
        self.logger: Optional[TBWrapper] = None

    def set_model_dir(self, model_dir: str, dataset_name: str) -> None:
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir)
        self.log_csv_path = os.path.join(self.ckpt_dir, "training_log.csv")
        self.log_csv_headers = [
            "epoch",
            "center_time_sec",
            "train_time_sec",
            "eval_time_sec",
            "total_time_sec",
            "train_loss",
            "p_true",
            "p_fake",
            "image_auroc",
            "image_ap",
            "pixel_auroc",
            "pixel_ap",
            "pixel_pro",
            "separation",
            "min_decision_margin",
            "is_best",
        ]
        with open(self.log_csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.log_csv_headers)
            writer.writeheader()

        self.predictions_dir = os.path.join(self.ckpt_dir, "epoch_predictions")
        if os.path.exists(self.predictions_dir):
            shutil.rmtree(self.predictions_dir)
        os.makedirs(self.predictions_dir, exist_ok=True)
        self.predictions_csv_path = os.path.join(self.ckpt_dir, "prediction_log.csv")
        self.predictions_csv_headers = [
            "epoch",
            "image_index",
            "image_path",
            "label",
            "score",
            "prediction_mask_path",
            "gt_mask_path",
        ]
        with open(self.predictions_csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.predictions_csv_headers)
            writer.writeheader()

    def _embed(
        self,
        images: torch.Tensor,
        detach: bool = True,
        provide_patch_shapes: bool = False,
        evaluation: bool = False,
    ):
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)
        features = [features[layer] for layer in self.layers_to_extract_from]
        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)
        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        patch_features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]
        for i in range(1, len(patch_features)):
            _features = patch_features[i]
            patch_dims = patch_shapes[i]
            _features = _features.reshape(_features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:])
            _features = _features.permute(0, 3, 4, 5, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(*perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1])
            _features = _features.permute(0, 4, 5, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            patch_features[i] = _features
        patch_features = [x.reshape(-1, *x.shape[-3:]) for x in patch_features]
        patch_features = self.forward_modules["preprocessing"](patch_features)
        patch_features = self.forward_modules["preadapt_aggregator"](patch_features)
        return patch_features, patch_shapes

        
    def trainer(
        self, training_data: DataLoader, val_data: DataLoader, name: str
    ) -> Optional[Dict[str, float]]:
        if self.logger is None:
            raise RuntimeError("Logger not initialized. Call set_model_dir first.")
        state_dict: Dict[str, OrderedDict] = {}
        ckpt_path = glob.glob(os.path.join(self.ckpt_dir, "ckpt_best*"))
        ckpt_path_save = os.path.join(self.ckpt_dir, "ckpt.pth")
        if len(ckpt_path) != 0:
            LOGGER.info("Start testing, ckpt file found! Skip training.")
            return None

        def update_state_dict() -> None:
            state_dict["discriminator"] = OrderedDict(
                {k: v.detach().cpu() for k, v in self.discriminator.state_dict().items()}
            )
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict(
                    {k: v.detach().cpu() for k, v in self.pre_projection.state_dict().items()}
                )

        self.distribution = getattr(training_data.dataset, "distribution", 0)
        xlsx_path = os.path.join("./datasets/excel", f"{name.split('_')[0]}_distribution.xlsx")
        if os.path.exists(xlsx_path):
            try:
                if self.distribution == 1:
                    self.svd = 1
                elif self.distribution == 2:
                    self.distribution = 0
                    self.svd = 0
                elif self.distribution == 3:
                    self.distribution = 0
                    self.svd = 1
                elif self.distribution == 4:
                    self.distribution = 0
                    df = pd.read_excel(xlsx_path)
                    self.svd = 1 - df.loc[df["Class"] == name, "Distribution"].values[0]
                else:
                    df = pd.read_excel(xlsx_path)
                    self.svd = df.loc[df["Class"] == name, "Distribution"].values[0]
            except Exception:
                self.distribution = 0
                self.svd = 0
        else:
            self.distribution = 0
            self.svd = 0

        if self.distribution == 1:
            self.forward_modules.eval()
            with torch.no_grad():
                weighted_sum: Optional[torch.Tensor] = None
                total_count = 0
                for data in training_data:
                    img = data["image"].to(torch.float).to(self.device)
                    batch_size = img.shape[0]
                    batch_sum = torch.sum(img, dim=0)
                    weighted_sum = batch_sum if weighted_sum is None else (weighted_sum + batch_sum)
                    total_count += batch_size
                if total_count <= 0 or weighted_sum is None:
                    raise RuntimeError("Training dataloader is empty when estimating distribution center.")
                self.c = weighted_sum / total_count
            avg_img = torch_format_2_numpy_img(self.c.detach().cpu().numpy())
            self.svd = distribution_judge(avg_img, name)
            os.makedirs(f"./results/judge/avg/{self.svd}", exist_ok=True)
            cv2.imwrite(f"./results/judge/avg/{self.svd}/{name}.png", avg_img)
            LOGGER.info("Distribution-only mode finished (svd=%s). Skip discriminator training.", self.svd)
            return None

        def is_better(candidate: Dict[str, float], current_best: Optional[Dict[str, float]]) -> bool:
            if current_best is None:
                return True
            tol = 1e-12
            cand_sep = candidate.get("separation", math.nan)
            best_sep = current_best.get("separation", math.nan)
            cand_sep_nan = math.isnan(cand_sep)
            best_sep_nan = math.isnan(best_sep)
            if cand_sep_nan and not best_sep_nan:
                return False
            if not cand_sep_nan and best_sep_nan:
                return True
            if not cand_sep_nan and not best_sep_nan:
                if cand_sep > best_sep + tol:
                    return True
                if cand_sep < best_sep - tol:
                    return False

            cand_margin = candidate.get("min_decision_margin", math.nan)
            best_margin = current_best.get("min_decision_margin", math.nan)
            cand_margin_nan = math.isnan(cand_margin)
            best_margin_nan = math.isnan(best_margin)
            if cand_margin_nan and not best_margin_nan:
                return False
            if not cand_margin_nan and best_margin_nan:
                return True
            if not cand_margin_nan and not best_margin_nan:
                if cand_margin > best_margin + tol:
                    return True
                if cand_margin < best_margin - tol:
                    return False

            if candidate["image_auroc"] > current_best["image_auroc"] + tol:
                return True
            if candidate["image_auroc"] < current_best["image_auroc"] - tol:
                return False

            if candidate["train_loss"] < current_best["train_loss"] - tol:
                return True
            if candidate["train_loss"] > current_best["train_loss"] + tol:
                return False

            return candidate["epoch"] < current_best["epoch"]

        best_record: Optional[Dict[str, float]] = None
        pbar_str1 = ""
        for i_epoch in range(self.meta_epochs):
            epoch_start = time.time()
            center_start = time.time()
            self.forward_modules.eval()
            with torch.no_grad():
                weighted_sum: Optional[torch.Tensor] = None
                total_count = 0
                for data in training_data:
                    img = data["image"].to(torch.float).to(self.device)
                    if self.pre_proj > 0:
                        outputs = self.pre_projection(self._embed(img, evaluation=False)[0])
                        outputs = outputs[0] if len(outputs) == 2 else outputs
                    else:
                        outputs = self._embed(img, evaluation=False)[0]
                        outputs = outputs[0] if len(outputs) == 2 else outputs
                    outputs = outputs.reshape(img.shape[0], -1, outputs.shape[-1])
                    batch_size = img.shape[0]
                    batch_sum = torch.sum(outputs, dim=0)
                    weighted_sum = batch_sum if weighted_sum is None else (weighted_sum + batch_sum)
                    total_count += batch_size
                if total_count <= 0 or weighted_sum is None:
                    raise RuntimeError("Training dataloader is empty when computing embedding center.")
                self.c = weighted_sum / total_count

            center_duration = time.time() - center_start

            train_start = time.time()
            pbar_str, pt, pf, avg_loss = self._train_discriminator(training_data, i_epoch, None, pbar_str1)
            train_duration = time.time() - train_start
            update_state_dict()

            if self.logger is not None and not math.isnan(avg_loss):
                self.logger.logger.add_scalar("train_loss_epoch", avg_loss, i_epoch)
            if self.logger is not None and not math.isnan(pt):
                self.logger.logger.add_scalar("p_true_epoch", pt, i_epoch)
            if self.logger is not None and not math.isnan(pf):
                self.logger.logger.add_scalar("p_fake_epoch", pf, i_epoch)
            if self.logger is not None:
                self.logger.logger.add_scalar("time/center", center_duration, i_epoch)
                self.logger.logger.add_scalar("time/train", train_duration, i_epoch)

            eval_duration = 0.0
            image_auroc = math.nan
            image_ap = math.nan
            pixel_auroc = math.nan
            pixel_ap = math.nan
            pixel_pro = math.nan
            separation = math.nan
            min_decision_margin = math.nan
            is_new_best = False
            if (i_epoch + 1) % self.eval_epochs == 0:
                eval_start = time.time()
                (
                    images,
                    scores,
                    segmentations,
                    labels_gt,
                    masks_gt,
                    image_paths,
                ) = self.predict(val_data)
                image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(
                    images, scores, segmentations, labels_gt, masks_gt, name
                )
                self.logger.logger.add_scalar("i-auroc", image_auroc, i_epoch)
                self.logger.logger.add_scalar("i-ap", image_ap, i_epoch)
                self.logger.logger.add_scalar("p-auroc", pixel_auroc, i_epoch)
                self.logger.logger.add_scalar("p-ap", pixel_ap, i_epoch)
                self.logger.logger.add_scalar("p-pro", pixel_pro, i_epoch)

                scores_arr = np.atleast_1d(np.squeeze(np.array(scores, dtype=np.float32)))
                labels_arr = np.atleast_1d(np.array(labels_gt, dtype=np.int64))
                ok_mask = labels_arr == 0
                ng_mask = labels_arr == 1
                if ok_mask.any() and ng_mask.any():
                    separation = float(scores_arr[ng_mask].mean() - scores_arr[ok_mask].mean())
                    min_decision_margin = float(np.min(scores_arr[ng_mask]) - np.max(scores_arr[ok_mask]))
                    if self.logger is not None:
                        self.logger.logger.add_scalar("separation", separation, i_epoch)
                        self.logger.logger.add_scalar("min_decision_margin", min_decision_margin, i_epoch)

                eval_duration = time.time() - eval_start
                if self.logger is not None:
                    self.logger.logger.add_scalar("time/eval", eval_duration, i_epoch)

                metrics_payload = {
                    "image_auroc": float(image_auroc),
                    "image_ap": float(image_ap),
                    "pixel_auroc": float(pixel_auroc),
                    "pixel_ap": float(pixel_ap),
                    "pixel_pro": float(pixel_pro),
                    "separation": separation,
                    "min_decision_margin": min_decision_margin,
                    "train_loss": float(avg_loss),
                }
                self._log_epoch_predictions(
                    epoch=i_epoch,
                    image_paths=image_paths,
                    scores=scores,
                    segmentations=segmentations,
                    labels_gt=labels_gt,
                    masks_gt=masks_gt,
                    metrics=metrics_payload,
                )

                eval_path = os.path.join("./results/eval", name)
                train_path = os.path.join("./results/training", name)
                os.makedirs(train_path, exist_ok=True)

                current_record = {
                    "image_auroc": float(image_auroc),
                    "image_ap": float(image_ap),
                    "pixel_auroc": float(pixel_auroc),
                    "pixel_ap": float(pixel_ap),
                    "pixel_pro": float(pixel_pro),
                    "epoch": float(i_epoch),
                    "separation": separation,
                    "min_decision_margin": min_decision_margin,
                    "train_loss": float(avg_loss),
                }

                is_new_best = is_better(current_record, best_record)
                if best_record is None or is_new_best:
                    best_record = current_record
                    shutil.rmtree(eval_path, ignore_errors=True)
                    ckpt_old = glob.glob(os.path.join(self.ckpt_dir, "ckpt_best_*.pth"))
                    for ckpt in ckpt_old:
                        os.remove(ckpt)
                    ckpt_path_best = os.path.join(self.ckpt_dir, f"ckpt_best_{i_epoch}.pth")
                    torch.save(state_dict, ckpt_path_best)
                    if os.path.exists(train_path):
                        shutil.copytree(train_path, eval_path)

                best_for_display = best_record if best_record is not None else current_record

                def _fmt_pct(value: float) -> str:
                    return "nan" if math.isnan(value) else f"{value * 100:.2f}"

                def _fmt_float(value: float, precision: int = 4) -> str:
                    return "nan" if math.isnan(value) else f"{value:.{precision}f}"

                best_loss = best_for_display.get("train_loss", math.nan)
                best_margin = best_for_display.get("min_decision_margin", math.nan)
                pbar_str1 = (
                    f" IAUC:{_fmt_pct(image_auroc)}({_fmt_pct(best_for_display['image_auroc'])})"
                    f" IAP:{_fmt_pct(image_ap)}({_fmt_pct(best_for_display['image_ap'])})"
                    f" PAUC:{_fmt_pct(pixel_auroc)}({_fmt_pct(best_for_display['pixel_auroc'])})"
                    f" PAP:{_fmt_pct(pixel_ap)}({_fmt_pct(best_for_display['pixel_ap'])})"
                    f" PRO:{_fmt_pct(pixel_pro)}({_fmt_pct(best_for_display['pixel_pro'])})"
                    f" SEP:{_fmt_float(separation)}({_fmt_float(best_for_display.get('separation', math.nan))})"
                    f" MDM:{_fmt_float(min_decision_margin)}({_fmt_float(best_margin)})"
                    f" LOSS:{'nan' if math.isnan(avg_loss) else f'{avg_loss:.2e}'}"
                    f"({'nan' if math.isnan(best_loss) else f'{best_loss:.2e}'})"
                    f" E:{i_epoch}({int(best_for_display['epoch'])})"
                )

            total_duration = time.time() - epoch_start
            time_info = (
                f" center:{center_duration:.2f}s"
                f" train:{train_duration:.2f}s"
                f" eval:{eval_duration:.2f}s"
                f" total:{total_duration:.2f}s"
            )

            LOGGER.info(
                "epoch:%d%s %s %s",
                i_epoch,
                time_info,
                (pbar_str or "").strip(),
                pbar_str1,
            )

            csv_row = {
                "epoch": i_epoch,
                "center_time_sec": center_duration,
                "train_time_sec": train_duration,
                "eval_time_sec": eval_duration,
                "total_time_sec": total_duration,
                "train_loss": avg_loss,
                "p_true": pt,
                "p_fake": pf,
                "image_auroc": image_auroc,
                "image_ap": image_ap,
                "pixel_auroc": pixel_auroc,
                "pixel_ap": pixel_ap,
                "pixel_pro": pixel_pro,
                "separation": separation,
                "min_decision_margin": min_decision_margin,
                "is_best": int(bool(is_new_best)),
            }
            with open(self.log_csv_path, "a", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.log_csv_headers)
                writer.writerow(csv_row)

            ckpt_epoch_path = os.path.join(self.ckpt_dir, f"ckpt_epoch_{i_epoch:04d}.pth")
            torch.save(state_dict, ckpt_epoch_path)
            torch.save(state_dict, ckpt_path_save)
        return best_record

    def _train_discriminator(self, input_data: DataLoader, cur_epoch: int, pbar, pbar_str1: str):
        self.forward_modules.eval()
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()

        all_loss: List[float] = []
        all_p_true: List[float] = []
        all_p_fake: List[float] = []
        all_r_t: List[float] = []
        all_r_g: List[float] = []
        all_r_f: List[float] = []
        sample_num = 0
        pbar_str2 = ""

        for data_item in input_data:
            self.dsc_opt.zero_grad()
            if self.pre_proj > 0:
                self.proj_opt.zero_grad()

            aug = data_item["aug"].to(torch.float).to(self.device)
            img = data_item["image"].to(torch.float).to(self.device)
            if self.pre_proj > 0:
                fake_feats = self.pre_projection(self._embed(aug, evaluation=False)[0])
                fake_feats = fake_feats[0] if len(fake_feats) == 2 else fake_feats
                true_feats = self.pre_projection(self._embed(img, evaluation=False)[0])
                true_feats = true_feats[0] if len(true_feats) == 2 else true_feats
            else:
                fake_feats = self._embed(aug, evaluation=False)[0]
                fake_feats.requires_grad = True
                true_feats = self._embed(img, evaluation=False)[0]
                true_feats.requires_grad = True

            mask_s_gt = data_item["mask_s"].reshape(-1, 1).to(self.device)
            noise = torch.normal(0, self.noise, true_feats.shape).to(self.device)
            gaus_feats = true_feats + noise

            center = self.c.repeat(img.shape[0], 1, 1)
            center = center.reshape(-1, center.shape[-1])
            true_points = torch.concat([fake_feats[mask_s_gt[:, 0] == 0], true_feats], dim=0)
            c_t_points = torch.concat([center[mask_s_gt[:, 0] == 0], center], dim=0)
            dist_t = torch.norm(true_points - c_t_points, dim=1)
            r_t = torch.tensor([torch.quantile(dist_t, q=self.radius)]).to(self.device)

            for step in range(self.step + 1):
                scores = self.discriminator(torch.cat([true_feats, gaus_feats]))
                true_scores = scores[: len(true_feats)]
                gaus_scores = scores[len(true_feats) :]
                true_loss = torch.nn.BCELoss()(true_scores, torch.zeros_like(true_scores))
                gaus_loss = torch.nn.BCELoss()(gaus_scores, torch.ones_like(gaus_scores))
                bce_loss = true_loss + gaus_loss

                if step == self.step:
                    break
                elif self.mining == 0:
                    dist_g = torch.norm(gaus_feats - center, dim=1)
                    r_g = torch.tensor([torch.quantile(dist_g, q=self.radius)]).to(self.device)
                    break

                grad = torch.autograd.grad(gaus_loss, [gaus_feats])[0]
                grad_norm = torch.norm(grad, dim=1).view(-1, 1)
                grad_normalized = grad / (grad_norm + 1e-10)

                with torch.no_grad():
                    gaus_feats.add_(0.001 * grad_normalized)

                if (step + 1) % 5 == 0:
                    dist_g = torch.norm(gaus_feats - center, dim=1)
                    r_g = torch.tensor([torch.quantile(dist_g, q=self.radius)]).to(self.device)
                    proj_feats = center if self.svd == 1 else true_feats
                    r = r_t if self.svd == 1 else 0.5

                    h = gaus_feats - proj_feats
                    h_norm = dist_g if self.svd == 1 else torch.norm(h, dim=1)
                    alpha = torch.clamp(h_norm, r, 2 * r)
                    proj = (alpha / (h_norm + 1e-10)).view(-1, 1)
                    h = proj * h
                    gaus_feats = proj_feats + h

            fake_points = fake_feats[mask_s_gt[:, 0] == 1]
            true_points = true_feats[mask_s_gt[:, 0] == 1]
            c_f_points = center[mask_s_gt[:, 0] == 1]
            dist_f = torch.norm(fake_points - c_f_points, dim=1)
            r_f = torch.tensor([torch.quantile(dist_f, q=self.radius)]).to(self.device)
            proj_feats = c_f_points if self.svd == 1 else true_points
            r = r_t if self.svd == 1 else 1

            if self.svd == 1:
                h = fake_points - proj_feats
                h_norm = dist_f if self.svd == 1 else torch.norm(h, dim=1)
                alpha = torch.clamp(h_norm, 2 * r, 4 * r)
                proj = (alpha / (h_norm + 1e-10)).view(-1, 1)
                h = proj * h
                fake_points = proj_feats + h
                fake_feats[mask_s_gt[:, 0] == 1] = fake_points

            fake_scores = self.discriminator(fake_feats)
            if self.p > 0:
                fake_dist = (fake_scores - mask_s_gt) ** 2
                d_hard = torch.quantile(fake_dist, q=self.p)
                fake_scores_ = fake_scores[fake_dist >= d_hard].unsqueeze(1)
                mask_ = mask_s_gt[fake_dist >= d_hard].unsqueeze(1)
            else:
                fake_scores_ = fake_scores
                mask_ = mask_s_gt
            output = torch.cat([1 - fake_scores_, fake_scores_], dim=1)
            focal_loss = self.focal_loss(output, mask_)

            loss = bce_loss + focal_loss
            loss.backward()
            if self.pre_proj > 0:
                self.proj_opt.step()
            if self.train_backbone:
                self.backbone_opt.step()
            self.dsc_opt.step()

            pix_true = torch.concat([fake_scores.detach() * (1 - mask_s_gt), true_scores.detach()])
            pix_fake = torch.concat([fake_scores.detach() * mask_s_gt, gaus_scores.detach()])
            p_true = (
                (pix_true < self.dsc_margin).sum() - (pix_true == 0).sum()
            ) / ((mask_s_gt == 0).sum() + true_scores.shape[0])
            p_fake = (pix_fake >= self.dsc_margin).sum() / ((mask_s_gt == 1).sum() + gaus_scores.shape[0])

            assert self.logger is not None
            self.logger.logger.add_scalar("p_true", p_true, self.logger.g_iter)
            self.logger.logger.add_scalar("p_fake", p_fake, self.logger.g_iter)
            self.logger.logger.add_scalar("r_t", r_t, self.logger.g_iter)
            self.logger.logger.add_scalar("r_g", r_g, self.logger.g_iter)
            self.logger.logger.add_scalar("r_f", r_f, self.logger.g_iter)
            self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
            self.logger.step()

            all_loss.append(loss.detach().cpu().item())
            all_p_true.append(p_true.cpu().item())
            all_p_fake.append(p_fake.cpu().item())
            all_r_t.append(r_t.cpu().item())
            all_r_g.append(r_g.cpu().item())
            all_r_f.append(r_f.cpu().item())

            all_loss_ = np.mean(all_loss)
            all_p_true_ = np.mean(all_p_true)
            all_p_fake_ = np.mean(all_p_fake)
            all_r_t_ = np.mean(all_r_t)
            all_r_g_ = np.mean(all_r_g)
            all_r_f_ = np.mean(all_r_f)
            sample_num += img.shape[0]

            pbar_str = (
                f"epoch:{cur_epoch} loss:{all_loss_:.2e}"
                f" pt:{all_p_true_ * 100:.2f}"
                f" pf:{all_p_fake_ * 100:.2f}"
                f" rt:{all_r_t_:.2f}"
                f" rg:{all_r_g_:.2f}"
                f" rf:{all_r_f_:.2f}"
                f" svd:{self.svd}"
                f" sample:{sample_num}"
            )
            pbar_str2 = pbar_str
            pbar_str += pbar_str1
            LOGGER.debug(pbar_str)

            if sample_num > self.limit:
                break

        all_loss_ = float(np.mean(all_loss)) if all_loss else math.nan
        all_p_true_ = float(np.mean(all_p_true)) if all_p_true else math.nan
        all_p_fake_ = float(np.mean(all_p_fake)) if all_p_fake else math.nan

        return pbar_str2, all_p_true_, all_p_fake_, all_loss_

    def tester(self, test_data: DataLoader, name: str):
        ckpt_path = glob.glob(os.path.join(self.ckpt_dir, "ckpt_best*"))
        if len(ckpt_path) != 0:
            state_dict = torch.load(ckpt_path[0], map_location=self.device)
            if "discriminator" in state_dict:
                self.discriminator.load_state_dict(state_dict["discriminator"])
                if "pre_projection" in state_dict and self.pre_proj > 0:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                self.load_state_dict(state_dict, strict=False)

            (
                images,
                scores,
                segmentations,
                labels_gt,
                masks_gt,
                image_paths,
            ) = self.predict(test_data)
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(
                images, scores, segmentations, labels_gt, masks_gt, name, path="eval"
            )
            epoch = int(ckpt_path[0].split("_")[-1].split(".")[0])
            scores_arr = np.atleast_1d(np.squeeze(np.array(scores, dtype=np.float32)))
            labels_arr = np.atleast_1d(np.array(labels_gt, dtype=np.int64))
            ok_mask = labels_arr == 0
            ng_mask = labels_arr == 1
            separation = math.nan
            min_decision_margin = math.nan
            if ok_mask.any() and ng_mask.any():
                separation = float(scores_arr[ng_mask].mean() - scores_arr[ok_mask].mean())
                min_decision_margin = float(np.min(scores_arr[ng_mask]) - np.max(scores_arr[ok_mask]))
            self._log_epoch_predictions(
                epoch=epoch,
                image_paths=image_paths,
                scores=scores,
                segmentations=segmentations,
                labels_gt=labels_gt,
                masks_gt=masks_gt,
                metrics={
                    "image_auroc": float(image_auroc),
                    "image_ap": float(image_ap),
                    "pixel_auroc": float(pixel_auroc),
                    "pixel_ap": float(pixel_ap),
                    "pixel_pro": float(pixel_pro),
                    "separation": separation,
                    "min_decision_margin": min_decision_margin,
                },
            )
        else:
            image_auroc = image_ap = pixel_auroc = pixel_ap = pixel_pro = 0.0
            epoch = -1
            LOGGER.info("No ckpt file found!")

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch

    def _evaluate(
        self,
        images: List[np.ndarray],
        scores: List[float],
        segmentations: List[np.ndarray],
        labels_gt: List[int],
        masks_gt: List[np.ndarray],
        name: str,
        path: str = "training",
    ):
        scores_arr = np.squeeze(np.array(scores))
        image_scores = compute_imagewise_retrieval_metrics(scores_arr, labels_gt, path)
        image_auroc = image_scores["auroc"]
        image_ap = image_scores["ap"]

        if len(masks_gt) > 0:
            segmentations_arr = np.array(segmentations)
            pixel_scores = compute_pixelwise_retrieval_metrics(segmentations_arr, masks_gt, path)
            pixel_auroc = pixel_scores["auroc"]
            pixel_ap = pixel_scores["ap"]
            if path == "eval":
                try:
                    pixel_pro = compute_pro(np.squeeze(np.array(masks_gt)), segmentations_arr)
                except Exception:
                    pixel_pro = 0.0
            else:
                pixel_pro = 0.0
        else:
            pixel_auroc = -1.0
            pixel_ap = -1.0
            pixel_pro = -1.0
            return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

        defects = np.array(images)
        targets = np.array(masks_gt)
        for i in range(len(defects)):
            defect = torch_format_2_numpy_img(defects[i])
            target = torch_format_2_numpy_img(targets[i])

            mask = cv2.cvtColor(
                cv2.resize(segmentations_arr[i], (defect.shape[1], defect.shape[0])),
                cv2.COLOR_GRAY2BGR,
            )
            mask = (mask * 255).astype("uint8")
            mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

            img_up = np.hstack([defect, target, mask])
            img_up = cv2.resize(img_up, (256 * 3, 256))
            full_path = os.path.join("./results", path, name)
            del_remake_dir(full_path, del_flag=False)
            cv2.imwrite(os.path.join(full_path, f"{i + 1:03d}.png"), img_up)

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

    def _log_epoch_predictions(
        self,
        epoch: int,
        image_paths: Sequence[str],
        scores: Sequence[float],
        segmentations: Sequence[np.ndarray],
        labels_gt: Sequence[int],
        masks_gt: Sequence[np.ndarray],
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        if not hasattr(self, "predictions_dir"):
            return

        epoch_dir = os.path.join(self.predictions_dir, f"epoch_{epoch:04d}")
        os.makedirs(epoch_dir, exist_ok=True)

        mask_dir = os.path.join(epoch_dir, "pred_masks")
        gt_dir = os.path.join(epoch_dir, "gt_masks")
        has_masks = bool(segmentations)
        has_gt_masks = bool(masks_gt)
        if has_masks:
            os.makedirs(mask_dir, exist_ok=True)
        if has_gt_masks:
            os.makedirs(gt_dir, exist_ok=True)

        if metrics is not None:
            metrics_path = os.path.join(epoch_dir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

        with open(self.predictions_csv_path, "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.predictions_csv_headers)
            for index, (img_path, score) in enumerate(zip(image_paths, scores)):
                pred_mask_path = ""
                if has_masks and index < len(segmentations):
                    mask_array = np.asarray(segmentations[index])
                    mask_img = np.squeeze(mask_array)
                    mask_img = np.clip(mask_img, 0.0, 1.0)
                    mask_img = (mask_img * 255).astype(np.uint8)
                    mask_file = os.path.join(mask_dir, f"pred_{index:05d}.png")
                    cv2.imwrite(mask_file, mask_img)
                    pred_mask_path = os.path.relpath(mask_file, self.ckpt_dir)

                gt_mask_path = ""
                if has_gt_masks and index < len(masks_gt):
                    gt_array = np.asarray(masks_gt[index])
                    gt_img = np.squeeze(gt_array)
                    gt_img = np.clip(gt_img, 0.0, 1.0)
                    gt_img = (gt_img * 255).astype(np.uint8)
                    gt_file = os.path.join(gt_dir, f"gt_{index:05d}.png")
                    cv2.imwrite(gt_file, gt_img)
                    gt_mask_path = os.path.relpath(gt_file, self.ckpt_dir)

                row = {
                    "epoch": epoch,
                    "image_index": index,
                    "image_path": img_path,
                    "label": int(labels_gt[index]) if index < len(labels_gt) else "",
                    "score": float(score),
                    "prediction_mask_path": pred_mask_path,
                    "gt_mask_path": gt_mask_path,
                }
                writer.writerow(row)

    def predict(self, test_dataloader: DataLoader):
        self.forward_modules.eval()

        images: List[np.ndarray] = []
        scores: List[float] = []
        masks: List[np.ndarray] = []
        labels_gt: List[int] = []
        masks_gt: List[np.ndarray] = []
        img_paths: List[str] = []

        with torch.no_grad():
            for data in test_dataloader:
                labels_gt.extend(data["is_anomaly"].numpy().tolist())
                if data.get("mask_gt", None) is not None:
                    masks_gt.extend(data["mask_gt"].numpy().tolist())
                image = data["image"]
                images.extend(image.numpy().tolist())
                img_paths.extend(data["image_path"])
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)

        return images, scores, masks, labels_gt, masks_gt, img_paths

    def _predict(self, img: torch.Tensor):
        img = img.to(torch.float).to(self.device)
        self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()

        with torch.no_grad():
            patch_features, patch_shapes = self._embed(img, provide_patch_shapes=True, evaluation=True)
            if self.pre_proj > 0:
                patch_features = self.pre_projection(patch_features)
                patch_features = patch_features[0] if len(patch_features) == 2 else patch_features

            patch_scores = image_scores = self.discriminator(patch_features)
            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=img.shape[0])
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(img.shape[0], scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=img.shape[0])
            image_scores = self.patch_maker.score(image_scores)
            if isinstance(image_scores, torch.Tensor):
                image_scores = image_scores.cpu().numpy()

        return list(image_scores), list(masks)


class DirectoryTrainDataset(Dataset):
    def __init__(
        self,
        ok_dir: str,
        anomaly_source_dir: Optional[str] = None,
        use_dtd: bool = False,
        resize: int = 288,
        imagesize: int = 288,
        downsampling: int = 8,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        gray: float = 0.0,
        hflip: float = 0.0,
        vflip: float = 0.0,
        rotate_degrees: int = 0,
        translate: float = 0.0,
        scale: float = 0.0,
        mean: float = 0.5,
        std: float = 0.1,
        rand_aug: int = 1,
    ) -> None:
        super().__init__()
        self.image_paths = list_images(ok_dir)
        if not self.image_paths:
            raise RuntimeError(f"No training images found in '{ok_dir}'.")

        self.use_dtd = use_dtd
        self.anomaly_paths = list_images(anomaly_source_dir) if use_dtd else []
        if self.use_dtd and not self.anomaly_paths:
            raise RuntimeError(
                "No anomaly source images found. Provide --dtd_dir pointing to the DTD dataset and enable --use_dtd."
            )

        self.resize = resize
        self.imagesize = imagesize
        self.downsampling = downsampling
        self.mean = mean
        self.std = std
        self.rand_aug = rand_aug
        self.distribution = 0

        self.transform_img = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.ColorJitter(brightness, contrast, saturation),
                transforms.RandomHorizontalFlip(hflip),
                transforms.RandomVerticalFlip(vflip),
                transforms.RandomGrayscale(gray),
                transforms.RandomAffine(
                    degrees=rotate_degrees,
                    translate=(translate, translate),
                    scale=(1.0 - scale, 1.0 + scale),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def rand_augmenter(self) -> transforms.Compose:
        augmentations = [
            transforms.ColorJitter(contrast=(0.8, 1.2)),
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            transforms.ColorJitter(saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.RandomGrayscale(p=1.0),
            transforms.RandomAutocontrast(p=1.0),
            transforms.RandomEqualize(p=1.0),
            transforms.RandomAffine(degrees=(-45, 45)),
        ]
        idx = np.random.choice(np.arange(len(augmentations)), 3, replace=False)
        return transforms.Compose(
            [
                transforms.Resize(self.resize),
                augmentations[idx[0]],
                augmentations[idx[1]],
                augmentations[idx[2]],
                transforms.CenterCrop(self.imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _sample_anomaly_image(self, base_image: Image.Image) -> torch.Tensor:
        if not self.use_dtd:
            return self.transform_img(base_image)
        aug_path = random.choice(self.anomaly_paths)
        aug_image = Image.open(aug_path).convert("RGB")
        transformer = self.rand_augmenter() if self.rand_aug else self.transform_img
        return transformer(aug_image)

    def __getitem__(self, index: int) -> Dict[str, object]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform_img(image)

        anomaly_image = self._sample_anomaly_image(image)

        mask_fg = torch.ones((self.imagesize, self.imagesize), dtype=torch.float32)
        mask_s, mask_l = perlin_mask(
            image_tensor.shape,
            self.imagesize // self.downsampling,
            0,
            6,
            mask_fg,
            1,
        )
        mask_s_tensor = torch.from_numpy(mask_s).float()
        mask_l_tensor = torch.from_numpy(mask_l).float()

        beta = np.random.normal(loc=self.mean, scale=self.std)
        beta = np.clip(beta, 0.2, 0.8)
        aug_image = (
            image_tensor * (1.0 - mask_l_tensor)
            + (1.0 - beta) * anomaly_image * mask_l_tensor
            + beta * image_tensor * mask_l_tensor
        )

        mask_gt = torch.zeros((1, self.imagesize, self.imagesize), dtype=torch.float32)

        return {
            "image": image_tensor,
            "aug": aug_image,
            "mask_s": mask_s_tensor,
            "mask_gt": mask_gt,
            "is_anomaly": torch.tensor(0, dtype=torch.long),
            "image_path": image_path,
        }


class DirectoryTestDataset(Dataset):
    def __init__(self, ok_dir: str, ng_dir: str, resize: int = 288, imagesize: int = 288) -> None:
        super().__init__()
        ok_paths = list_images(ok_dir)
        ng_paths = list_images(ng_dir)
        if not ok_paths:
            raise RuntimeError(f"No test OK images found in '{ok_dir}'.")
        if not ng_paths:
            raise RuntimeError(f"No test NG images found in '{ng_dir}'.")
        self.samples = [(path, 0) for path in ok_paths] + [(path, 1) for path in ng_paths]
        self.imagesize = imagesize
        self.transform_img = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform_img(image)
        mask_gt = torch.zeros((1, self.imagesize, self.imagesize), dtype=torch.float32)
        return {
            "image": image,
            "mask_gt": mask_gt,
            "is_anomaly": torch.tensor(label, dtype=torch.long),
            "image_path": image_path,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GLASS with directory-based datasets.")
    parser.add_argument("--train_ok_dir", type=str, required=True, help="Directory with training OK images.")
    parser.add_argument("--test_ok_dir", type=str, required=True, help="Directory with test OK images.")
    parser.add_argument("--test_ng_dir", type=str, required=True, help="Directory with test anomaly images.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to store logs and checkpoints.")
    parser.add_argument("--use_dtd", action="store_true", help="Enable DTD-based LAS synthesis for training.")
    parser.add_argument(
        "--dtd_dir",
        type=str,
        default="./datasets/dtd/images",
        help="Directory with DTD images used for LAS synthesis.",
    )
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=288)
    parser.add_argument("--backbone", type=str, default="wideresnet50")
    parser.add_argument("--run_name", type=str, default="custom")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to store result visualizations.")
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    setup_seed(args.seed)

    device = resolve_device(args.device)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    train_dataset = DirectoryTrainDataset(
        ok_dir=args.train_ok_dir,
        anomaly_source_dir=args.dtd_dir,
        use_dtd=args.use_dtd,
        imagesize=args.image_size,
    )
    test_dataset = DirectoryTestDataset(
        ok_dir=args.test_ok_dir,
        ng_dir=args.test_ng_dir,
        imagesize=args.image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    backbone = load_backbone(args.backbone)
    glass_model = GLASS(device)
    glass_model.load(
        backbone=backbone,
        layers_to_extract_from=("layer2", "layer3"),
        device=device,
        input_shape=(3, args.image_size, args.image_size),
        pretrain_embed_dimension=1536,
        target_embed_dimension=1536,
        patchsize=3,
        meta_epochs=args.epochs,
        eval_epochs=1,
        dsc_layers=2,
        dsc_hidden=1024,
        dsc_margin=0.5,
        train_backbone=False,
        pre_proj=1,
        mining=1,
        noise=0.015,
        radius=0.75,
        p=0.5,
        lr=args.lr,
        svd=0,
        step=20,
        limit=len(train_dataset),
    )
    glass_model.set_model_dir(args.save_dir, dataset_name=args.run_name)

    print("Starting training...")
    best_record = glass_model.trainer(train_loader, test_loader, name=args.run_name)
    if best_record is not None:
        sep_val = best_record.get("separation", math.nan)
        sep_str = f"{sep_val:.4f}" if not math.isnan(sep_val) else "nan"
        mdm_val = best_record.get("min_decision_margin", math.nan)
        mdm_str = f"{mdm_val:.4f}" if not math.isnan(mdm_val) else "nan"
        train_loss = best_record.get("train_loss", math.nan)
        loss_str = f"{train_loss:.4e}" if not math.isnan(train_loss) else "nan"
        print(
            "Best validation metrics - "
            f"Epoch {int(best_record['epoch'])}: IAUC={best_record['image_auroc']:.4f}, "
            f"IAP={best_record['image_ap']:.4f}, PAUC={best_record['pixel_auroc']:.4f}, "
            f"PAP={best_record['pixel_ap']:.4f}, PRO={best_record['pixel_pro']:.4f}, "
            f"SEP={sep_str}, MDM={mdm_str}, TrainLoss={loss_str}"
        )

    print("Evaluating best checkpoint...")
    image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch = glass_model.tester(
        test_loader, name=args.run_name
    )
    print(
        "Test metrics - "
        f"Epoch {epoch}: IAUC={image_auroc:.4f}, IAP={image_ap:.4f}, "
        f"PAUC={pixel_auroc:.4f}, PAP={pixel_ap:.4f}, PRO={pixel_pro:.4f}"
    )


if __name__ == "__main__":
    main()
