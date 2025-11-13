"""PyTorch implementation of the GLASS anomaly detector.

Derived from infer_pth.py so that both the CLI and the Qt application can
share the same model code.
"""
from __future__ import annotations

import copy
from typing import Iterable, List, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _init_weight(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)


class Discriminator(nn.Module):
    def __init__(self, in_planes: int, n_layers: int = 2, hidden: int | None = None):
        super().__init__()
        _hidden = in_planes if hidden is None else hidden
        layers = nn.Sequential()
        for i in range(n_layers - 1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            block = nn.Sequential(
                nn.Linear(_in, _hidden),
                nn.BatchNorm1d(_hidden),
                nn.LeakyReLU(0.2),
            )
            layers.add_module(f"block{i + 1}", block)
        self.body = layers
        self.tail = nn.Sequential(nn.Linear(_hidden, 1, bias=False), nn.Sigmoid())
        self.apply(_init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.body(x)
        x = self.tail(x)
        return x


class Projection(nn.Module):
    def __init__(self, in_planes: int, out_planes: int | None = None, n_layers: int = 1, layer_type: int = 0):
        super().__init__()
        if out_planes is None:
            out_planes = in_planes
        layers = nn.Sequential()
        for i in range(n_layers):
            in_dim = in_planes if i == 0 else out_planes
            layers.add_module(f"{i}fc", nn.Linear(in_dim, out_planes))
            if i < n_layers - 1 and layer_type > 1:
                layers.add_module(f"{i}relu", nn.LeakyReLU(0.2))
        self.layers = layers
        self.apply(_init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.layers(x)


class PatchMaker:
    def __init__(self, patchsize: int, top_k: int = 0, stride: int | None = None):
        self.patchsize = patchsize
        self.stride = stride or 1
        self.top_k = top_k

    def patchify(self, features: torch.Tensor, return_spatial_info: bool = False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = nn.Unfold(kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1)
        unfolded = unfolder(features)
        num_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            num_patches.append(int(n_patches))
        unfolded = unfolded.reshape(*features.shape[:2], self.patchsize, self.patchsize, -1)
        unfolded = unfolded.permute(0, 4, 1, 2, 3)
        if return_spatial_info:
            return unfolded, num_patches
        return unfolded

    @staticmethod
    def unpatch_scores(x: torch.Tensor, batchsize: int) -> torch.Tensor:
        return x.reshape(batchsize, -1, *x.shape[1:])

    @staticmethod
    def score(x: torch.Tensor) -> torch.Tensor:
        x = x[:, :, 0]
        return torch.max(x, dim=1).values


class Preprocessing(nn.Module):
    def __init__(self, input_dims: Sequence[int], output_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([MeanMapper(output_dim) for _ in input_dims])

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:  # type: ignore[override]
        mapped = []
        for module, feat in zip(self.layers, features):
            mapped.append(module(feat))
        return torch.stack(mapped, dim=1)


class MeanMapper(nn.Module):
    def __init__(self, preprocessing_dim: int):
        super().__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(nn.Module):
    def __init__(self, target_dim: int):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class RescaleSegmentor:
    def __init__(self, device: torch.device, target_size: tuple[int, int] = (288, 288), smoothing_kernel: int = 9):
        self.device = device
        self.target_size = target_size
        self.smoothing_kernel = smoothing_kernel

    def convert_to_segmentation(self, patch_scores: torch.Tensor) -> list[np.ndarray]:
        with torch.no_grad():
            scores = patch_scores.to(self.device).unsqueeze(1)
            scores = F.interpolate(scores, size=self.target_size, mode="bilinear", align_corners=False)
            scores = scores.squeeze(1).cpu().numpy()

        masks: list[np.ndarray] = []
        for arr in scores:
            s_min, s_max = arr.min(), arr.max()
            norm = (arr - s_min) / (s_max - s_min + 1e-8)
            uint8 = np.uint8(norm * 255)
            blur = cv2.GaussianBlur(uint8, (self.smoothing_kernel, self.smoothing_kernel), 0)
            masks.append(blur.astype(np.float32) / 255.0)
        return masks


class LastLayerToExtractReachedException(Exception):
    pass


class ForwardHook:
    def __init__(self, hook_dict: dict[str, torch.Tensor], layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(layer_name == last_layer_to_extract)

    def __call__(self, module, input, output):  # type: ignore[override]
        self.hook_dict[self.layer_name] = output
        return None


class NetworkFeatureAggregator(nn.Module):
    def __init__(self, backbone: nn.Module, layers_to_extract_from: Sequence[str], device: torch.device, train_backbone: bool = False):
        super().__init__()
        self.layers_to_extract_from = list(layers_to_extract_from)
        self.backbone = backbone
        self.device = device
        self.train_backbone = train_backbone
        if not hasattr(backbone, "hook_handles"):
            backbone.hook_handles = []  # type: ignore[attr-defined]
        for handle in getattr(backbone, "hook_handles", []):
            handle.remove()
        self.outputs: dict[str, torch.Tensor] = {}
        for extract_layer in layers_to_extract_from:
            self._register_hook(extract_layer)
        self.to(self.device)

    def forward(self, images: torch.Tensor, eval: bool = True) -> dict[str, torch.Tensor]:  # type: ignore[override]
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

    def feature_dimensions(self, input_shape: Sequence[int]) -> list[int]:
        dummy = torch.ones([1] + list(input_shape)).to(self.device)
        output = self(dummy)
        return [output[layer].shape[1] for layer in self.layers_to_extract_from]

    def _register_hook(self, layer_name: str) -> None:
        module = self._find_module(self.backbone, layer_name)
        if module is None:
            raise ValueError(f"Module {layer_name} not found in the model")
        forward_hook = ForwardHook(self.outputs, layer_name, self.layers_to_extract_from[-1])
        if isinstance(module, nn.Sequential):
            hook = module[-1].register_forward_hook(forward_hook)
        else:
            hook = module.register_forward_hook(forward_hook)
        self.backbone.hook_handles.append(hook)  # type: ignore[attr-defined]

    def _find_module(self, model: nn.Module, module_name: str) -> nn.Module | None:
        for name, module in model.named_modules():
            if name == module_name:
                return module
            if "." in module_name:
                father, child = module_name.split(".", 1)
                if name == father:
                    return self._find_module(module, child)
        return None


def load_backbone(name: str) -> nn.Module:
    name = name.lower()
    if name == "wideresnet50":
        try:
            backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        except Exception:
            backbone = models.wide_resnet50_2(pretrained=True)
    else:
        raise ValueError(f"Unsupported backbone '{name}'")
    return backbone


class GLASSInfer(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        backbone_name: str = "wideresnet50",
        layers_to_extract_from: Sequence[str] = ("layer2", "layer3"),
        input_shape: Sequence[int] = (3, 288, 288),
        pretrain_embed_dimension: int = 1536,
        target_embed_dimension: int = 1536,
        patchsize: int = 3,
        patchstride: int = 1,
        dsc_layers: int = 2,
        dsc_hidden: int = 1024,
        pre_proj: int = 1,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.layers_to_extract_from = list(layers_to_extract_from)
        self.input_shape = tuple(input_shape)
        self.target_embed_dimension = target_embed_dimension
        self.pre_proj = pre_proj

        backbone = load_backbone(backbone_name).to(self.device)
        feature_aggregator = NetworkFeatureAggregator(backbone, self.layers_to_extract_from, self.device, train_backbone=False)
        feature_dimensions = feature_aggregator.feature_dimensions(self.input_shape)

        self.forward_modules = nn.ModuleDict()
        self.forward_modules["feature_aggregator"] = feature_aggregator
        self.forward_modules["preprocessing"] = Preprocessing(feature_dimensions, pretrain_embed_dimension)
        preadapt_aggregator = Aggregator(target_dim=target_embed_dimension).to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, n_layers=pre_proj).to(self.device)
        else:
            self.pre_projection = None

        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden).to(self.device)
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.anomaly_segmentor = RescaleSegmentor(device=self.device, target_size=self.input_shape[-2:], smoothing_kernel=9)
        self.eval()

    def load_checkpoint(self, ckpt_path: str) -> None:
        state = torch.load(ckpt_path, map_location=self.device)
        if "discriminator" in state:
            self.discriminator.load_state_dict(state["discriminator"])
            if self.pre_projection is not None and "pre_projection" in state:
                self.pre_projection.load_state_dict(state["pre_projection"])
        else:
            self.load_state_dict(state, strict=False)
        self.eval()

    def _embed(self, images: torch.Tensor, provide_patch_shapes: bool = False):
        self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)
        features = [features[layer] for layer in self.layers_to_extract_from]
        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(L**0.5), int(L**0.5), C).permute(0, 3, 1, 2)

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        patch_features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(patch_features)):
            feat = patch_features[i]
            patch_dims = patch_shapes[i]
            feat = feat.reshape(feat.shape[0], patch_dims[0], patch_dims[1], *feat.shape[2:])
            feat = feat.permute(0, 3, 4, 5, 1, 2)
            base_shape = feat.shape
            feat = feat.reshape(-1, *feat.shape[-2:])
            feat = F.interpolate(feat.unsqueeze(1), size=(ref_num_patches[0], ref_num_patches[1]), mode="bilinear", align_corners=False)
            feat = feat.squeeze(1)
            feat = feat.reshape(*base_shape[:-2], ref_num_patches[0], ref_num_patches[1])
            feat = feat.permute(0, 4, 5, 1, 2, 3)
            feat = feat.reshape(len(feat), -1, *feat.shape[-3:])
            patch_features[i] = feat

        patch_features = [x.reshape(-1, *x.shape[-3:]) for x in patch_features]
        patch_features = self.forward_modules["preprocessing"](patch_features)
        patch_features = self.forward_modules["preadapt_aggregator"](patch_features)
        if provide_patch_shapes:
            return patch_features, patch_shapes
        return patch_features

    def _predict(self, imgs: torch.Tensor):
        imgs = imgs.to(torch.float).to(self.device)
        self.forward_modules.eval()
        if self.pre_projection is not None:
            self.pre_projection.eval()
        self.discriminator.eval()
        with torch.no_grad():
            patch_features, patch_shapes = self._embed(imgs, provide_patch_shapes=True)
            if self.pre_projection is not None:
                patch_features = self.pre_projection(patch_features)
                if isinstance(patch_features, (list, tuple)):
                    patch_features = patch_features[0]

            patch_scores = image_scores = self.discriminator(patch_features)
            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=imgs.shape[0])
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(imgs.shape[0], scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=imgs.shape[0])
            image_scores = self.patch_maker.score(image_scores)
            if isinstance(image_scores, torch.Tensor):
                image_scores = image_scores.cpu().numpy()
        return list(image_scores), masks

    def infer_batch(self, imgs: torch.Tensor):
        import time

        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        imgs = imgs.to(self.device, non_blocking=True)
        start = time.perf_counter()
        scores, masks = self._predict(imgs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return [float(s) for s in scores], masks, float(elapsed_ms)


def preprocess_patch(patch_bgr: np.ndarray, size: int) -> torch.Tensor:
    rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    )
    tensor = transform(rgb)
    return tensor


__all__ = [
    "GLASSInfer",
    "preprocess_patch",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
