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

def apply_morphological_operations(mask):
    """
    对蒙版进行形态学闭操作处理
    
    参数:
        mask: 输入的布尔蒙版
    
    返回:
        closed_mask: 经过闭操作处理后的蒙版 (numpy二维array, uint8类型)
    """
    # 将布尔蒙版转换为uint8 (0和255)
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # 定义闭操作的核
    # 可以调整核的大小来适应不同的图像
    kernel_size = (15, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    
    # 应用闭操作 (先膨胀后腐蚀，用于填充小孔洞和连接断裂区域)
    closed_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    
    return closed_mask

def apply_mask_to_image(original_rgb, mask):
    """
    将蒙版应用到原图
    
    参数:
        original_rgb: 原始RGB图像
        mask: 蒙版 (numpy二维array)
    
    返回:
        masked_image: 应用蒙版后的图像
    """
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

def draw_compared_images(original_rgb, skin_mask, closed_mask, masked_image):
    """
    显示原图、原始蒙版、闭操作后蒙版和应用蒙版后的图像
    """
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
    plt.title("After Closing Operation")
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
    
    # 叠加显示
    plt.subplot(2, 3, 6)
    overlay = original_rgb.copy()
    overlay[closed_bool] = overlay[closed_bool] // 2 + np.array([128, 0, 0])  # 红色叠加
    plt.imshow(overlay)
    plt.title("Overlay on Original")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def process_image_pipeline(image_path):
    """
    完整的图像处理流程
    """
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
    # 处理多张图片
    image_paths = ["img_2.png", "img_1.png"]
    
    for image_path in image_paths:
        try:
            print(f"处理图片: {image_path}")
            original_rgb, skin_mask, closed_mask, masked_image = process_image_pipeline(image_path)
            
            # 打印蒙版信息
            print(f"原始蒙版非零像素数: {np.sum(skin_mask)}")
            print(f"闭操作后蒙版非零像素数: {np.sum(closed_mask > 0)}")
            print(f"蒙版数据类型: {closed_mask.dtype}")
            print(f"蒙版形状: {closed_mask.shape}")
            print("-" * 50)
            
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {e}")
