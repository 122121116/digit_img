import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_gray_float(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float64)


def pad_to_size(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    out = np.zeros((target_h, target_w), dtype=img.dtype)
    out[:h, :w] = img
    return out


def fft_cross_correlation(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    F_img = np.fft.fft2(image)
    F_tpl = np.fft.fft2(template)
    corr = np.fft.ifft2(F_img * np.conj(F_tpl))
    return corr


def locate_max_correlation(image_path: str, template_path: str) -> tuple[int, int, float]:

    img = load_gray_float(image_path)
    tpl = load_gray_float(template_path)

    target_h, target_w = 298, 298
    img_pad = pad_to_size(img, target_h, target_w)
    tpl_pad = pad_to_size(tpl, target_h, target_w)

    corr = fft_cross_correlation(img_pad, tpl_pad)
    corr_abs = np.abs(corr)
    flat_idx = int(np.argmax(corr_abs))
    r, c = np.unravel_index(flat_idx, corr_abs.shape)

    vmin, vmax = float(corr_abs.min()), float(corr_abs.max())
    if vmax <= vmin + 1e-12:
        corr_norm = np.zeros_like(corr_abs)
    else:
        corr_norm = (corr_abs - vmin) / (vmax - vmin)
    
    plt.imshow(corr_norm, cmap="gray")
    plt.scatter([c], [r], c="r", s=20)
    plt.title(f"corr_abs (max=({r},{c}))")
    plt.show()

    return r, c, float(corr_abs[r, c])


if __name__ == "__main__":
    #取3为模板
    row, col, val = locate_max_correlation("2.jpg", "3.png")
    print(f"max corr at (row={row}, col={col}), value={val}")

    #取2为模板
    row, col, val = locate_max_correlation("3.png", "2.jpg")
    print(f"max corr at (row={row}, col={col}), value={val}")

