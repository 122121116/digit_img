import cv2
import numpy as np
import matplotlib.pyplot as plt

def keep_largest_connected_component(mask):
    # 确保输入为uint8类型
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # 查找所有连通组件
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # 无有效连通区域时直接返回
    if num_labels <= 1:
        return np.zeros_like(mask)

    # 找到最大连通组件
    max_area = -1
    max_label = 1
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            max_label = i

    # 生成只包含最大连通区域的蒙版
    largest_mask = np.zeros_like(mask)
    largest_mask[labels == max_label] = 255

    return largest_mask

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

def apply_morphological_operations(mask):
    # 将布尔蒙版转换为uint8 (0和255)
    mask_uint8 = mask.astype(np.uint8) * 255

    # 应用闭操作
    kernel_size = (12, 12)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    closed_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    closed_mask = keep_largest_connected_component(closed_mask)
    # 应用开操作
    kernel_size = (15, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)

    return keep_largest_connected_component(opened_mask)

def apply_mask_to_image(original_rgb, mask):
    # 确保蒙版是二值的 (0和255)
    if mask.dtype == np.bool_:
        mask_uint8 = mask.astype(np.uint8) * 255
    else:
        mask_uint8 = mask
    
    # 将蒙版转换为3通道以便与原图进行运算
    mask_3ch = np.stack([mask_uint8] * 3, axis=-1)
    
    # 应用蒙版
    masked_image = np.where(mask_3ch == 255, original_rgb, 0)
    
    return masked_image.astype(np.uint8)

def draw_compared_images(original_rgb, skin_mask, closed_mask, masked_image):
    # 创建画布
    plt.figure(figsize=(16, 12))
    
    # 原图
    plt.subplot(2, 3, 1)
    plt.imshow(original_rgb)
    plt.title("Original Image")
    plt.axis("off")
    
    # 原始蒙版
    plt.subplot(2, 3, 2)
    plt.imshow(skin_mask, cmap="gray")
    plt.title("Original Mask")
    plt.axis("off")
    
    # 闭操作后的蒙版
    plt.subplot(2, 3, 3)
    plt.imshow(closed_mask, cmap="gray")
    plt.title("After Closing and Opening Operation")
    plt.axis("off")
    
    # 应用蒙版后的图像
    plt.subplot(2, 3, 4)
    plt.imshow(masked_image)
    plt.title("Masked Image")
    plt.axis("off")
    
    # 蒙版对比 (原始 vs 闭操作后)
    plt.subplot(2, 3, 5)
    # 用不同颜色显示差异
    comparison = np.zeros((*skin_mask.shape, 3), dtype=np.uint8)
    # 原始蒙版区域显示为红色
    comparison[skin_mask] = [255, 0, 0]
    # 闭操作新增区域显示为绿色
    closed_bool = closed_mask > 0
    new_areas = closed_bool & ~skin_mask
    comparison[new_areas] = [0, 255, 0]
    plt.imshow(comparison)
    plt.title("Mask Comparison\nRed: Original, Green: Added by Closing")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def process_image_pipeline(image_path):
    # 1. 提取肤色蒙版
    original_rgb, skin_mask = extract_skin_mask(image_path)
    
    # 2. 对蒙版进行闭操作
    closed_mask = apply_morphological_operations(skin_mask)
    
    # 3. 将处理后的蒙版应用到图像
    masked_image = apply_mask_to_image(original_rgb, closed_mask)
    
    # 4. 显示结果
    draw_compared_images(original_rgb, skin_mask, closed_mask, masked_image)
    
    return original_rgb, skin_mask, closed_mask, masked_image

if __name__ == "__main__":

    image_path1 = "img_1.png"
    image_path2 = "img_2.png"
    image_path3 = "img_3.png"

    process_image_pipeline(image_path1)
    process_image_pipeline(image_path2)
    process_image_pipeline(image_path3)


