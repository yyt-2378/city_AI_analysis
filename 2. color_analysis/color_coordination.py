from sklearn.cluster import KMeans
import numpy as np
import cv2
from skimage.color import rgb2lab, deltaE_cie76
from skimage import color
import os
import matplotlib.pyplot as plt
import pandas as pd
import copy


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

# todo: need to fix
# def calculate_color_coordination(image_path):
#     img = cv2.imread(image_path)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_rgb_filter = copy.deepcopy(img_rgb)
#
#     # 获取颜色A和颜色B的像素值
#     # 进行聚类
#     kmeans = KMeans(n_clusters=20)
#     kmeans.fit(img_rgb_filter)
#
#     # 获取每个像素的类别标签
#     labels = kmeans.labels_
#
#     # 创建一个字典，将类别标签作为键，像素集合作为值
#     pixel_clusters = {}
#     for i in range(len(labels)):
#         label = labels[i]
#         pixel = img_rgb_filter[i]
#
#         if label not in pixel_clusters:
#             pixel_clusters[label] = []
#
#         pixel_clusters[label].append(pixel)
#
#     pixels_A = img_rgb[np.all(img_rgb == color_A, axis=2)]
#     pixels_B = img_rgb[np.all(img_rgb == color_B, axis=2)]
#
#     # 计算颜色A和颜色B的亮度和饱和度值
#     brightness_A = pixels_A[:, 0, 0]
#     brightness_B = pixels_B[:, 0, 0]
#     chroma_A = pixels_A[:, 0, 1]
#     chroma_B = pixels_B[:, 0, 1]
#
#     # 计算颜色A和颜色B的区域面积
#     area_A = len(pixels_A)
#     area_B = len(pixels_B)
#
#     # 计算颜色协调值
#     coordination_value = (brightness_A * brightness_B) / (chroma_A * chroma_B) * (area_A * area_B)
#
#     return coordination_value


def extract_dominant_colors(image_path, n_clusters):
    img = cv2.imread(image_path)
    img_r_h, img_r_w = img.shape[0] // 5, img.shape[1] // 5
    img = cv2.resize(img, (img_r_h, img_r_w))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 计算像素的亮度
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    brightness = img_hsv[:, :, 2]

    threshold = 20

    # 去除低于阈值的像素
    filtered_pixels = img_rgb[brightness >= threshold].reshape(-1, 3)
    unique_filtered_pixels = np.unique(filtered_pixels, axis=0)
    if unique_filtered_pixels.shape[0] < n_clusters:
        n_clusters = unique_filtered_pixels.shape[0]

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(filtered_pixels)
    dominant_colors_rgb = np.uint8(kmeans.cluster_centers_)

    return dominant_colors_rgb


def analyze_color_harmony(dominant_colors_rgb):
    dominant_colors_lab = rgb2lab(dominant_colors_rgb.reshape(1, -1, 3))
    harmony_scores = []
    for i in range(dominant_colors_lab.shape[1]):
        for j in range(i + 1, dominant_colors_lab.shape[1]):
            delta_e = deltaE_cie76(dominant_colors_lab[0][i], dominant_colors_lab[0][j])
            harmony_scores.append(delta_e)

    average_harmony = np.mean(harmony_scores)
    return average_harmony


def plot_color_spectrum(colors, title, image_dir):
    """绘制单行颜色谱图"""
    # 确保颜色数组形状正确并且归一化
    colors = colors.reshape((1, -1, 3)) / 255
    plt.figure(figsize=(12, 2))  # 图片宽度为12英寸，高度为2英寸
    plt.imshow(colors, aspect='auto')  # aspect='auto'表示不保持纵横比
    plt.axis('off')  # 关闭坐标轴
    plt.title(title)
    plt.tight_layout()  # 调整布局
    img_path = os.path.join(image_dir, title)
    plt.savefig(f'{image_path}.png', dpi=300, transparent=True)  # 以300 DPI的高分辨率保存图像


if __name__ == '__main__':
    image_dirs = [os.path.join('/Selected/B&A', path_i) for path_i in
                  os.listdir('/Selected/B&A')]
    for image_dir in image_dirs:
        if os.path.isfile(image_dir):
            continue
        # 设定图片文件夹路径
        image_paths = [os.path.join(image_dir, img_name) for img_name in os.listdir(image_dir) if
                       img_name.endswith(('.png', '.jpg', '.jpeg'))]
        # 分析每张图片并计算颜色协调度得分
        harmony_scores = []
        image_names = []

        for image_path in image_paths:
            dominant_colors_rgb = extract_dominant_colors(image_path, n_clusters=20)
            color_harmony_score = analyze_color_harmony(np.array(dominant_colors_rgb))
            harmony_scores.append(color_harmony_score)
            image_names.append(os.path.basename(image_path))

        # Create DataFrame and save to CSV
        df = pd.DataFrame({
            'Image Name': image_names,
            'Harmony Score': harmony_scores
        })
        csv_file_path = os.path.join(image_dir, "harmony_scores_and_average.csv")
        df.to_csv(csv_file_path, index=False)

        plt.figure(figsize=(20, 16))
        indexes = np.arange(len(image_names))
        plt.bar(indexes, harmony_scores, color='skyblue')
        plt.xlabel('Images', fontsize=14)
        plt.ylabel('Harmony Scores', fontsize=14)
        plt.xticks(indexes, image_names, rotation='vertical', fontsize=8)
        plt.title('Harmony Scores Distribution', fontsize=17)
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, 'color_harmony_distribution.jpg'))

        # Calculate and print the average color harmony score
        average_harmony_score = np.mean(harmony_scores)
        print(f"\nAverage Color Harmony Score for {image_dir}: {average_harmony_score}")

        # Append average to CSV
        with open(csv_file_path, "a") as file:
            file.write(f"\nAverage Color Harmony Score,{average_harmony_score}")

        # Inform the user
        print(f"Harmony scores and their average have been saved to {csv_file_path}")

