import argparse
import os
from typing import List

import cv2
import numpy as np
import torch

from app.models.glass_torch import GLASSInfer, IMAGENET_MEAN, IMAGENET_STD
from torchvision import transforms


def preprocess_image(path: str, image_size: int = 288):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    tensor = transform(img_rgb).unsqueeze(0)
    return img_bgr, tensor


def save_heatmap(mask: np.ndarray, img_bgr: np.ndarray, out_path: str) -> None:
    h, w = img_bgr.shape[:2]
    m_min, m_max = mask.min(), mask.max()
    m_norm = (mask - m_min) / (m_max - m_min + 1e-8)
    m_uint8 = np.uint8(m_norm * 255)
    m_color = cv2.applyColorMap(m_uint8, cv2.COLORMAP_JET)
    m_color = cv2.resize(m_color, (w, h))
    overlay = cv2.addWeighted(img_bgr, 0.6, m_color, 0.4, 0)
    concat = np.hstack([img_bgr, m_color, overlay])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, concat)


def gather_images(single_path: str | None, image_dir: str | None) -> List[str]:
    images: List[str] = []
    if single_path and os.path.isfile(single_path):
        images.append(single_path)
    if image_dir and os.path.isdir(image_dir):
        for fname in os.listdir(image_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                images.append(os.path.join(image_dir, fname))
    if not images:
        raise ValueError("Không tìm thấy ảnh hợp lệ (tham số --image hoặc --image_dir)")
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="GLASS anomaly inference (.pth)")
    parser.add_argument("--ckpt", required=True, help="Đường dẫn ckpt .pth")
    parser.add_argument("--image", default=None, help="Đường dẫn ảnh duy nhất")
    parser.add_argument("--image_dir", default=None, help="Thư mục chứa nhiều ảnh")
    parser.add_argument("--device", default="cuda:0", help="Thiết bị (ví dụ cuda:0 hoặc cpu)")
    parser.add_argument("--out_dir", default="glass_results", help="Thư mục lưu kết quả")
    parser.add_argument("--img_size", type=int, default=288, help="Kích thước resize")
    parser.add_argument("--backbone", default="wideresnet50", help="Backbone, mặc định wideresnet50")
    args = parser.parse_args()

    paths = gather_images(args.image, args.image_dir)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA không khả dụng, chuyển sang CPU")
        device = "cpu"
    else:
        device = args.device

    model = GLASSInfer(
        device=device,
        backbone_name=args.backbone,
        layers_to_extract_from=("layer2", "layer3"),
        input_shape=(3, args.img_size, args.img_size),
        pretrain_embed_dimension=1536,
        target_embed_dimension=1536,
        patchsize=3,
        patchstride=1,
        dsc_layers=2,
        dsc_hidden=1024,
        pre_proj=1,
    )
    model.load_checkpoint(args.ckpt)
    os.makedirs(args.out_dir, exist_ok=True)

    print("file\tanomaly_score\tinfer_ms")
    for img_path in paths:
        img_bgr, img_tensor = preprocess_image(img_path, image_size=args.img_size)
        img_tensor = img_tensor.to(device)
        scores, masks, elapsed_ms = model.infer_batch(img_tensor)
        score = float(scores[0])
        mask = masks[0]
        base = os.path.splitext(os.path.basename(img_path))[0]
        save_heatmap(mask, img_bgr, os.path.join(args.out_dir, base + "_glass.png"))
        print(f"{base}\t{score:.6f}\t{elapsed_ms:.2f}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()

