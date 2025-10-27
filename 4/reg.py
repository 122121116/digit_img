import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_skin_mask(image_path):
    # 读取图片并转换为RGB
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 归一化到0-1范围
    normalized = rgb / 255.0

    r = normalized[:, :, 0]
    g = normalized[:, :, 1]
    b = normalized[:, :, 2]

    cond1 = (r >= g) & (g >= b)  # R > G > B
    cond2 = (r >= 0.2) & (r <= 0.9)  # 亮度范围
    cond3 = (g / r >= 0.5) & (g / r <= 0.85)  # G与R的比例
    cond4 = (b / r >= 0.2) & (b / r <= 0.65)  # B与R的比例

    skin_mask = cond1 & cond2 & cond3 & cond4
    return rgb, skin_mask.astype(np.bool_)


def plot_histograms(original_rgb, skin_mask):
    # 提取肤色区域的像素
    skin_pixels = original_rgb[skin_mask]

    # 创建画布
    plt.figure(figsize=(15, 10))

    # 绘制原图的RGB直方图
    plt.subplot(2, 1, 1)
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        # 计算通道直方图（使用未归一化的原始像素值0-255）
        hist, bins = np.histogram(original_rgb[:, :, i].ravel(), bins=256, range=[0, 256])
        plt.plot(bins[:-1], hist, color=color, alpha=0.7, label=f'{color} channel')
    plt.title('RGB (original)')
    plt.xlabel('pixel value (0-255)')
    plt.ylabel('num of pixels')
    plt.legend()
    plt.xlim(0, 255)

    # 绘制肤色区域的RGB直方图
    plt.subplot(2, 1, 2)
    if len(skin_pixels) > 0:  # 避免无肤色像素时出错
        for i, color in enumerate(colors):
            hist, bins = np.histogram(skin_pixels[:, i].ravel(), bins=256, range=[0, 256])
            plt.plot(bins[:-1], hist, color=color, alpha=0.7, label=f'{color} channel (skin)')
    plt.title('RGB (skin)')
    plt.xlabel('pixel value (0-255)')
    plt.ylabel('pixes num')
    plt.legend()
    plt.xlim(0, 255)

    plt.tight_layout()
    plt.show()

def draw_compared_images(original_rgb, skin_mask):
    # 显示原图和蒙版
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(original_rgb)
    plt.title("ori")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(skin_mask, cmap="gray")
    plt.title("mask")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 绘制直方图
    plot_histograms(original_rgb, skin_mask)

if __name__ == "__main__":
    image_path = "img_2.png"
    original_rgb, skin_mask = extract_skin_mask(image_path)
    draw_compared_images(original_rgb, skin_mask)

    image_path = "img_1.png"
    original_rgb, skin_mask = extract_skin_mask(image_path)
    draw_compared_images(original_rgb, skin_mask)
