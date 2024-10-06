import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import cv2


def convert_rgb_to_hsv(rgb_colors):
    """将 RGB 颜色转换为 HSV 颜色空间"""
    rgb_colors = rgb_colors.reshape((-1, 1, 3)).astype(np.uint8)  # 确保数据类型正确
    hsv_colors = cv2.cvtColor(rgb_colors, cv2.COLOR_RGB2HSV)
    return hsv_colors.reshape((-1, 3))


def sort_colors_by_hue(colors):
    """将颜色按 HSV 色相排序"""
    hsv_colors = convert_rgb_to_hsv(colors)
    # 根据 HSV 中的色相（H值）排序，色相范围通常是0-180或0-360
    # 按照色相排序则为0， 饱和度为1， 明度为2
    sorted_indices = np.argsort(hsv_colors[:, 0])
    return colors[sorted_indices]


def extract_dominant_colors(image, k=20):
    """使用K-Means聚类提取图像的主要颜色"""
    pixels = np.float32(image.reshape(-1, 3))
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(pixels)
    palette = kmeans.cluster_centers_
    return palette


def recluster_colors(colors, k):
    """使用K-Means聚类将颜色数量减少到k"""
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(colors)
    new_palette = kmeans.cluster_centers_
    return new_palette


def analyze_folder(folder_path, brightness_threshold=80):
    """分析文件夹中的所有图像并提取主要颜色，排除黑色像素点。
    
    参数:
    folder_path -- 要分析的文件夹路径。
    brightness_threshold -- 亮度阈值，用于排除黑色像素点。
    """
    all_dominant_colors = []
    for img_file in Path(folder_path).glob('*.jpg'):
        img = cv2.imread(str(img_file))
        img_r_h, img_r_w = img.shape[0] // 5, img.shape[1] // 5
        img = cv2.resize(img, (img_r_w, img_r_h))  # 修正了这里的参数顺序，应为宽度在前，高度在后

        # 将图像从BGR转换为RGB，因为OpenCV默认使用BGR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 将RGB图像转换为灰度图像，然后检查亮度
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # 找到所有亮度高于阈值的像素点
        mask = img_gray > brightness_threshold
        bright_pixels = img_rgb[mask]
        
        # 如果图像中有亮度足够的像素点，则进行聚类
        if bright_pixels.size > 0:
            dominant_colors = extract_dominant_colors(bright_pixels)
            all_dominant_colors.append(dominant_colors)
        else:
            # 如果没有足够亮的像素点，则跳过这张图片
            continue

    return np.array(all_dominant_colors)


def plot_color_spectrum(colors, title, street):
    """绘制单行颜色谱图"""
    # 确保颜色数组形状正确并且归一化
    colors = colors.reshape((1, -1, 3)) / 255
    plt.figure(figsize=(12, 2))  # 图片宽度为12英寸，高度为2英寸
    plt.imshow(colors, aspect='auto')  # aspect='auto'表示不保持纵横比
    plt.axis('off')  # 关闭坐标轴
    plt.title(title)
    plt.tight_layout()  # 调整布局
    output = os.path.join(street, title+'.png')
    plt.savefig(output, dpi=300, transparent=True)  # 以300 DPI的高分辨率保存图像


# 获取所有街区的颜色数据
folder_paths = [os.path.join('C:\\Users\\yyt70\\Desktop\\Three Styles Color', sub) for sub in os.listdir('C:\\Users\\yyt70\\Desktop\\Three Styles Color')]
street_color_data = {}

for folder in folder_paths:
    file_basename = os.path.basename(folder)
    colors = analyze_folder(folder)
    street_color_data[file_basename] = colors

# 确定所有街区的最小颜色集合大小
min_colors = 20

# 为了进行比较，我们需要重新聚类每个街区的颜色
for street in street_color_data:
    # 将所有颜色聚合在一起，以便重新聚类
    aggregated_colors = np.vstack(street_color_data[street])
    recolored = recluster_colors(aggregated_colors, min_colors)
    # Save the dominant colors to a text file
    # np.savetxt('dominant_colors.txt', aggregated_colors, fmt='%d')
    sorted_colors = sort_colors_by_hue(recolored)
    street_color_data[street] = sorted_colors


# 为每个街区绘制颜色谱带
for street, colors in street_color_data.items():
    os.makedirs(str(street), exist_ok=True)
    plot_color_spectrum(colors, f'{street}_recluster New Color Spectrum sorted by RGB', street)
    print(f'-----------------------successful plotting {street}-----------------------')

