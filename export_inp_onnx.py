"""Convert INP-Former .pth checkpoint to ONNX with dynamic batch support.

Example:
  python export_inp_onnx.py \
    --pretrained_model_path "D:/DataMea14/NGUYEN/INP/ckpt_epoch_0214.pth" \
    --encoder dinov2reg_vit_small_14 \
    --height 448 --width 448 --batch 1 --dynamic-batch
"""
from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path

import onnx
import onnxsim
import torch
import torch.nn as nn

from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Aggregation_Block, Mlp, Prototype_Block
from utils import setup_seed


def set_model(args: argparse.Namespace) -> tuple[nn.Module, torch.device]:
    setup_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = vit_encoder.load(args.encoder)
    if "small" in args.encoder:
        embed_dim, num_heads = 384, 6
    elif "base" in args.encoder:
        embed_dim, num_heads = 768, 12
    elif "large" in args.encoder:
        embed_dim, num_heads = 1024, 16
    else:
        raise RuntimeError("Architecture must contain one of: small/base/large.")

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.0)])
    inp_tokens = nn.ParameterList([nn.Parameter(torch.randn(args.inp_num, embed_dim)) for _ in range(1)])
    inp_extractor = nn.ModuleList(
        [
            Aggregation_Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
            )
            for _ in range(1)
        ]
    )
    inp_guided_decoder = nn.ModuleList(
        [
            Prototype_Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
            )
            for _ in range(8)
        ]
    )

    model = INP_Former(
        encoder=encoder,
        bottleneck=bottleneck,
        aggregation=inp_extractor,
        decoder=inp_guided_decoder,
        target_layers=target_layers,
        remove_class_token=True,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        prototype_token=inp_tokens,
    ).to(device)

    state_dict = torch.load(args.pretrained_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def export_onnx(args: argparse.Namespace, torch_model: nn.Module, device: torch.device) -> Path:
    ckpt_path = Path(args.pretrained_model_path)
    export_onnx_path = ckpt_path.with_suffix(".onnx")
    if args.output_onnx:
        export_onnx_path = Path(args.output_onnx)

    dummy_input = torch.randn(args.batch, 3, args.height, args.width, device=device)
    output_names = ["fs0", "fs1", "ft0", "ft1"]
    dynamic_axes = {"input": {0: "batch_size"}} if args.dynamic_batch else None
    if args.dynamic_batch:
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}

    torch.onnx.export(
        torch_model,
        dummy_input,
        str(export_onnx_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    simplified_model, ok = onnxsim.simplify(str(export_onnx_path))
    if not ok:
        raise RuntimeError("onnxsim.simplify failed validation")
    onnx.save(simplified_model, str(export_onnx_path))
    return export_onnx_path


def inspect_onnx_batch(path: Path) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    shape = sess.get_inputs()[0].shape
    print(f"[INFO] Exported ONNX: {path}")
    print(f"[INFO] Input shape metadata: {shape}")
    if shape and len(shape) > 0:
        bdim = shape[0]
        if isinstance(bdim, int):
            print(f"[WARN] Batch dimension is static: {bdim}")
        else:
            print(f"[OK] Batch dimension is dynamic: {bdim}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert INP-Former .pth to ONNX with optional dynamic batch.")
    parser.add_argument("--encoder", type=str, default="dinov2reg_vit_small_14")
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--inp-num", type=int, default=16)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--output-onnx", type=str, default="")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    model, device = set_model(args)
    onnx_path = export_onnx(args, model, device)
    inspect_onnx_batch(onnx_path)
    print(f"[DONE] ONNX saved at: {onnx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
