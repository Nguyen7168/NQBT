import argparse
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


WINDOW_NAME = "Kiem guong - dieu chinh tham so"


@dataclass
class CircleResult:
    center_x: int
    center_y: int
    radius: int

    @property
    def diameter(self) -> int:
        return self.radius * 2


def _odd(v: int) -> int:
    return v if v % 2 == 1 else v + 1


def _create_trackbars(image_shape: Tuple[int, int]) -> None:
    h, w = image_shape
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("crop_left_px", WINDOW_NAME, 0, w // 2, lambda _: None)
    cv2.createTrackbar("crop_right_px", WINDOW_NAME, 0, w // 2, lambda _: None)
    cv2.createTrackbar("crop_top_px", WINDOW_NAME, 0, h // 2, lambda _: None)
    cv2.createTrackbar("crop_bottom_px", WINDOW_NAME, 0, h // 2, lambda _: None)

    cv2.createTrackbar("blur_ksize", WINDOW_NAME, 7, 31, lambda _: None)
    cv2.createTrackbar("hough_dp_x10", WINDOW_NAME, 12, 30, lambda _: None)
    cv2.createTrackbar("minDist", WINDOW_NAME, 14, 200, lambda _: None)
    cv2.createTrackbar("canny_param1", WINDOW_NAME, 120, 300, lambda _: None)
    cv2.createTrackbar("acc_param2", WINDOW_NAME, 22, 120, lambda _: None)
    cv2.createTrackbar("minRadius", WINDOW_NAME, 15, min(h, w) // 2, lambda _: None)
    cv2.createTrackbar("maxRadius", WINDOW_NAME, min(h, w) // 4, min(h, w) // 2, lambda _: None)

    cv2.createTrackbar("center_tol", WINDOW_NAME, 10, 80, lambda _: None)
    cv2.createTrackbar("min_radius_gap", WINDOW_NAME, 4, 30, lambda _: None)


def _read_params(image_shape: Tuple[int, int]) -> dict:
    h, w = image_shape
    left = cv2.getTrackbarPos("crop_left_px", WINDOW_NAME)
    right = cv2.getTrackbarPos("crop_right_px", WINDOW_NAME)
    top = cv2.getTrackbarPos("crop_top_px", WINDOW_NAME)
    bottom = cv2.getTrackbarPos("crop_bottom_px", WINDOW_NAME)

    x0 = min(left, w - 2)
    x1 = max(w - right, x0 + 2)
    y0 = min(top, h - 2)
    y1 = max(h - bottom, y0 + 2)

    blur = _odd(max(1, cv2.getTrackbarPos("blur_ksize", WINDOW_NAME)))
    dp = max(1.0, cv2.getTrackbarPos("hough_dp_x10", WINDOW_NAME) / 10.0)
    min_dist = max(1, cv2.getTrackbarPos("minDist", WINDOW_NAME))
    param1 = max(1, cv2.getTrackbarPos("canny_param1", WINDOW_NAME))
    param2 = max(1, cv2.getTrackbarPos("acc_param2", WINDOW_NAME))

    min_r = cv2.getTrackbarPos("minRadius", WINDOW_NAME)
    max_r = cv2.getTrackbarPos("maxRadius", WINDOW_NAME)
    if max_r <= min_r:
        max_r = min_r + 1

    center_tol = max(1, cv2.getTrackbarPos("center_tol", WINDOW_NAME))
    min_gap = max(1, cv2.getTrackbarPos("min_radius_gap", WINDOW_NAME))

    return {
        "crop": (x0, y0, x1, y1),
        "blur": blur,
        "dp": dp,
        "min_dist": min_dist,
        "param1": param1,
        "param2": param2,
        "min_r": min_r,
        "max_r": max_r,
        "center_tol": center_tol,
        "min_gap": min_gap,
    }


def _filter_concentric(circles: np.ndarray, center_tol: int, min_radius_gap: int) -> List[CircleResult]:
    if circles.size == 0:
        return []

    circles = np.round(circles[0]).astype(int)
    med_x = int(np.median(circles[:, 0]))
    med_y = int(np.median(circles[:, 1]))

    candidates = []
    for x, y, r in circles:
        if np.hypot(x - med_x, y - med_y) <= center_tol:
            candidates.append((x, y, r))

    if not candidates:
        return []

    candidates.sort(key=lambda c: c[2])
    selected: List[CircleResult] = []
    for x, y, r in candidates:
        if not selected or abs(r - selected[-1].radius) >= min_radius_gap:
            selected.append(CircleResult(x, y, r))

    return selected


def detect_circles(image_bgr: np.ndarray, params: dict) -> Tuple[np.ndarray, List[CircleResult], np.ndarray]:
    x0, y0, x1, y1 = params["crop"]
    roi = image_bgr[y0:y1, x0:x1]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (params["blur"], params["blur"]), 0)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=params["dp"],
        minDist=params["min_dist"],
        param1=params["param1"],
        param2=params["param2"],
        minRadius=params["min_r"],
        maxRadius=params["max_r"],
    )

    filtered = _filter_concentric(
        circles if circles is not None else np.empty((1, 0, 3)),
        params["center_tol"],
        params["min_gap"],
    )

    annotated = image_bgr.copy()
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 200, 0), 2)

    for idx, c in enumerate(filtered, start=1):
        abs_center = (c.center_x + x0, c.center_y + y0)
        cv2.circle(annotated, abs_center, c.radius, (0, 0, 255), 2)
        cv2.circle(annotated, abs_center, 2, (0, 255, 255), 3)
        cv2.putText(
            annotated,
            f"{idx}: D={c.diameter}px",
            (abs_center[0] + 6, abs_center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    info = f"So duong tron: {len(filtered)}"
    cv2.putText(annotated, info, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 255), 2, cv2.LINE_AA)

    if filtered:
        diameters = ", ".join(str(c.diameter) for c in filtered)
        cv2.putText(
            annotated,
            f"Duong kinh (px): {diameters}",
            (12, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated, filtered, gray


def run_app(image_path: str) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Khong mo duoc anh: {image_path}")

    _create_trackbars(image.shape[:2])

    while True:
        params = _read_params(image.shape[:2])
        annotated, circles, gray = detect_circles(image, params)

        cv2.imshow(WINDOW_NAME, annotated)
        cv2.imshow("ROI gray", gray)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            out_path = "ket_qua_kiem_guong.png"
            cv2.imwrite(out_path, annotated)
            print(f"Da luu ket qua: {out_path}, so duong tron={len(circles)}")

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="App tim duong kinh cac vong tron dong tam trong guong mau trang"
    )
    parser.add_argument("--image", required=True, help="Duong dan anh dau vao")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_app(args.image)
