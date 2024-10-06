import cv2
import csv
import numpy as np
from sklearn.cluster import DBSCAN
import os
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns


def extract_colors(image_path, brightness_threshold=50):
    """Extract and convert colors to CIELAB color space with brightness threshold."""
    img = cv2.imread(image_path)

    # 重新计算缩放尺寸，以保持原图的纵横比
    img_r_h, img_r_w = img.shape[0], img.shape[1]
    img = cv2.resize(img, (img_r_h, img_r_w))

    # 转换到LAB颜色空间
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # 应用亮度阈值
    mask = img_lab[:, :, 0] > brightness_threshold  # L通道大于亮度阈值
    filtered_lab = img_lab[mask]

    return filtered_lab.reshape(-1, 3)


def calculate_color_distribution(colors):
    """Calculate color distribution using entropy."""
    hist, _ = np.histogramdd(colors, bins=20, range=[[0, 255], [0, 255], [0, 255]])
    hist_norm = hist / hist.sum()
    return entropy(hist_norm.flatten())


def save_results_to_file(complexity_metrics, filename):
    """Save the complexity metrics to a file."""
    with open(filename, 'w') as file:
        for key, value in complexity_metrics.items():
            file.write(f'{key}: {value}\n')


def plot_color_distribution(colors, title, filename):
    """Plot color distribution as a histogram."""
    plt.figure(figsize=(10, 6))
    sns.histplot(colors[:, 0], bins=30, kde=True, color='blue', label='L Channel')
    sns.histplot(colors[:, 1], bins=30, kde=True, color='green', label='A Channel')
    sns.histplot(colors[:, 2], bins=30, kde=True, color='red', label='B Channel')
    plt.title(title)
    plt.xlabel('Color Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(filename)


def analyze_single_image_color_complexity(image_path):
    """Analyze color complexity for a single image."""
    colors = extract_colors(image_path)
    color_dist = calculate_color_distribution(colors)
    complexity = {
        'color_entropy': color_dist,
    }
    return complexity


def save_results_to_csv(results, filename):
    """Save the complexity metrics of images to a CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['image', 'color_entropy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def plot_complexity_distribution(results, title, filename):
    """Plot the distribution of color complexity."""
    complexities = [r['color_entropy'] for r in results]
    plt.figure(figsize=(10, 6))
    sns.histplot(complexities, bins=30, kde=True)
    plt.title(title)
    plt.xlabel('Color Complexity (Entropy)')
    plt.ylabel('Frequency')
    plt.savefig(filename)


if __name__ == '__main__':
    folder_path = r'F:\E_files\final_preprocessed_imgs\final_preprocessed_imgs'  # 替换为您的文件夹路径
    output_dir = r'F:\E_files\final_preprocessed_imgs\final_preprocessed_imgs'
    os.makedirs(output_dir, exist_ok=True)

    for img_file in os.listdir(folder_path):
        img_path_folder = os.path.join(folder_path, img_file)
        file_basename = img_path_folder.split('\\')[-1]
        results = []
        for im_i in os.listdir(img_path_folder):
            im_i_path = os.path.join(img_path_folder, im_i)
            complexity = analyze_single_image_color_complexity(im_i_path)
            complexity['image'] = im_i_path
            results.append(complexity)
            print(f"-----------------success doing complexity {im_i_path}--------------------------")
        
        # 保存结果到CSV文件
        csv_filename = os.path.join(output_dir, f'color_city_style_{file_basename}_complexity_results.csv')
        save_results_to_csv(results, csv_filename)
        
        # 绘制复杂度分布图
        plot_filename = os.path.join(output_dir, f'color_city_style_{file_basename}_complexity_distribution.png')
        plot_complexity_distribution(results, 'Color Complexity Distribution', plot_filename)