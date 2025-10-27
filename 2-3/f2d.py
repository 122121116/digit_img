import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def gaussian_lowpass_transfer(shape: tuple[int, int], cutoff: float) -> np.ndarray:
    H, W = shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    # 经典频域高斯低通：exp(-D^2 / (2*D0^2))
    eps = 1e-12
    D0 = max(float(cutoff), eps)
    Hc = np.exp(-dist2 / (2.0 * (D0 ** 2)))
    
    return np.fft.ifftshift(Hc)

def apply_gaussian_lowpass(spectrum: np.ndarray, cutoff: float) -> np.ndarray:
    H = gaussian_lowpass_transfer(spectrum.shape, cutoff)
    return spectrum * H

def apply_gaussian_highpass(spectrum: np.ndarray, cutoff: float) -> np.ndarray:
    H = gaussian_lowpass_transfer(spectrum.shape, cutoff)
    return spectrum * (1-H)

def show(input_path: str, cutoff: float) -> None:
    orig = Image.open(input_path).convert("L")
    img_np = np.asarray(orig, dtype=np.float64)

    # 频域
    S = np.fft.fft2(img_np)
    # 低通
    Sf_lp = apply_gaussian_lowpass(S, cutoff)
    img_lp = np.fft.ifft2(Sf_lp).real
    # 高通
    Sf_hp = apply_gaussian_highpass(S, cutoff)
    img_hp = np.fft.ifft2(Sf_hp).real
    # 原图减低通
    img_sub = img_np - img_lp

    plt.figure(figsize=(18, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(img_np, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(img_lp, cmap="gray")
    plt.title(f"Gaussian LP (cutoff={cutoff})")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(img_hp, cmap="gray")
    plt.title(f"Gaussian HP (cutoff={cutoff})")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(img_sub, cmap="gray")
    plt.title("Gaussian HP(origin-low) (cutoff={cutoff})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show("1.jpeg", 10)