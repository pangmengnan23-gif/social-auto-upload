# -*- coding: utf-8 -*-
import base64
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np
import segno


def build_login_qrcode_path(account_file: str, suffix: str = "login_qrcode") -> Path:
    account_path = Path(account_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return account_path.with_name(f"{account_path.stem}_{suffix}_{timestamp}.png")


def save_data_url_image(data_url: str, output_path: Path) -> Path:
    if not data_url.startswith("data:image/"):
        raise ValueError("二维码地址不是 data:image 格式")

    header, encoded = data_url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("二维码图片不是 base64 编码")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(encoded))
    return output_path


def remove_qrcode_file(qrcode_path: Path | None) -> bool:
    if qrcode_path and qrcode_path.exists():
        qrcode_path.unlink()
        return True
    return False


def decode_qrcode_from_path(qrcode_path: Path) -> str | None:
    image = cv2.imread(str(qrcode_path))
    if image is None:
        return None

    detector = cv2.QRCodeDetector()
    qrcode_content, _, _ = detector.detectAndDecode(image)
    return qrcode_content or None


def _crop_binary_qrcode(binary_image: np.ndarray) -> np.ndarray | None:
    dark_rows = np.where(np.any(binary_image, axis=1))[0]
    dark_cols = np.where(np.any(binary_image, axis=0))[0]
    if dark_rows.size == 0 or dark_cols.size == 0:
        return None

    return binary_image[
           dark_rows[0]: dark_rows[-1] + 1,
           dark_cols[0]: dark_cols[-1] + 1,
           ]


def _extract_qrcode_matrix_from_image(qrcode_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(qrcode_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    detector = cv2.QRCodeDetector()
    try:
        _, _, straight_qrcode = detector.detectAndDecode(image)
    except cv2.error:
        straight_qrcode = None

    if straight_qrcode is not None and getattr(straight_qrcode, "size", 0):
        binary_image = straight_qrcode < 128
        cropped = _crop_binary_qrcode(binary_image)
        if cropped is not None:
            return cropped

    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    binary_image = thresholded > 0
    return _crop_binary_qrcode(binary_image)


def _print_matrix_qrcode(matrix: np.ndarray, border: int = 1) -> bool:
    if matrix.size == 0:
        return False

    padded = np.pad(matrix.astype(bool), border, constant_values=False)
    white = "  "
    black = "██"
    empty_line = white * padded.shape[1]

    print(empty_line)
    for row in padded:
        print("".join(black if cell else white for cell in row))
    print(empty_line)
    return True


def _print_qrcode_from_image(qrcode_path: Path, border: int = 1) -> bool:
    matrix = _extract_qrcode_matrix_from_image(qrcode_path)
    if matrix is None:
        return False
    return _print_matrix_qrcode(matrix, border=border)


def _print_ascii_qrcode(qrcode, border: int = 1) -> None:
    rows = np.array(list(qrcode.matrix), dtype=bool)
    _print_matrix_qrcode(rows, border=border)


def _print_matrix_qrcode_compact(matrix: np.ndarray, border: int = 1) -> bool:
    """
    使用半高字符渲染二维码，显著缩小高度。
    一个终端字符行可以表示二维码的两行数据。
    """
    if matrix.size == 0:
        return False

    # 添加边距
    padded = np.pad(matrix.astype(bool), border, constant_values=False)
    h, w = padded.shape

    # 遍历行，步长为 2
    for r in range(0, h, 2):
        line = ""
        for c in range(w):
            upper = padded[r, c]
            # 如果是奇数行，最后一行可能没有配对的下一行，假设下一行为白色（False）
            lower = padded[r + 1, c] if r + 1 < h else False

            if upper and lower:
                line += "█"  # 全黑
            elif upper and not lower:
                line += "▀"  # 上黑
            elif not upper and lower:
                line += "▄"  # 下黑
            else:
                line += " "  # 全白
        print(line)
    return True


def print_terminal_qrcode(
        qrcode_content: str | None,
        qrcode_path: Path,
        app_name: str,
        border: int = 4,
) -> None:
    print(f"\n请使用{app_name}扫描下方二维码登录：")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # 1. 优先从内容生成矩阵（保证最清晰且最小化）
    # 2. 如果没内容，则从图片提取
    matrix = None
    if qrcode_content:
        # 使用 segno 生成最小版本的矩阵
        qr = segno.make(qrcode_content, error="L", boost_error=False)
        matrix = np.array(list(qr.matrix), dtype=bool)
    else:
        matrix = _extract_qrcode_matrix_from_image(qrcode_path)

    if matrix is not None:
        try:
            # 使用紧凑模式打印
            _print_matrix_qrcode_compact(matrix, border=border)
        except (UnicodeEncodeError, OSError):
            print("当前终端不支持二维码块字符，请直接打开本地图片扫码。")
    else:
        print("二维码矩阵提取失败，请查看本地图片。")

    print(f"提示：如果上方显示异常，请打开图片扫码: {qrcode_path}\n")
