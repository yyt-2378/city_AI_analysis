import cv2
import numpy as np
from matplotlib import pyplot as plt
from imageio import mimsave
import os
from sklearn.cluster import KMeans


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

# def ensure_hsv_format(hsv_colors):
#     """确保 HSV 颜色数据类型正确，并转换格式以便处理"""
#     hsv_colors = hsv_colors.reshape((-1, 1, 3)).astype(np.uint8)
#     return hsv_colors.reshape((-1, 3))

# def sort_colors_by_hue(colors):
#     """将颜色按 HSV 色相排序"""
#     colors = np.squeeze(colors)
#     hsv_colors = ensure_hsv_format(colors)
#     sorted_indices = np.argsort(hsv_colors[:, 0])
#     return colors[sorted_indices]


def calculate_optimal_clusters(mask_image):
    non_zero_pixels = np.count_nonzero(mask_image)
    optimal_clusters = int(non_zero_pixels / 1000)
    optimal_clusters = max(1, optimal_clusters)
    return optimal_clusters


def extract_dominant_colors(mask_image):
    # k = calculate_optimal_clusters(mask_image)
    pixels = np.float32(mask_image.reshape(-1, 3))
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(pixels)
    palette = kmeans.cluster_centers_
    return palette


def analyze_single_img(img_rgb, brightness_threshold=7):
    img_value = img_rgb[:,:,0]  # R channel for brightness
    mask = img_value > brightness_threshold
    bright_pixels = img_rgb[mask]
    dominant_colors = extract_dominant_colors(bright_pixels)
    # Save the dominant colors to a text file
    np.savetxt('dominant_colors.txt', dominant_colors, fmt='%d')
    return np.array([dominant_colors])


def plot_color_spectrum(colors, title):
    hsv_colors = colors.astype(np.uint8).reshape(1, -1, 3)
    colors = cv2.cvtColor(hsv_colors, cv2.COLOR_HSV2RGB) / 255.0
    plt.figure(figsize=(12, 2))
    plt.imshow(colors, aspect='auto')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{title}.png', dpi=300, transparent=True)


def extract_features_and_mask(image_path, channel_idx):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    channel = img_rgb[:,:,channel_idx]
    edges = cv2.Canny(channel, 100, 200)
    mask = np.where(edges > 0, 1, 0)
    masked_img = np.zeros_like(img_rgb)
    masked_img[:,:,channel_idx] = img_rgb[:,:,channel_idx] * mask
    channel_img = np.zeros_like(img_rgb)
    channel_img[:,:,channel_idx] = img_rgb[:,:,channel_idx]
    return masked_img, channel_img


def extract_mask(image_path):
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masked_img = np.zeros_like(img_hsv)
    masked_img = img_hsv
    channel_img = np.zeros_like(img_hsv)
    channel_img = img_hsv
    return masked_img, channel_img


def create_fading_gif(images, output_path, steps=10):
    if len(images) < 2:
        raise ValueError("需要至少两张图片来创建渐变GIF。")
    gif_frames = []
    start_img, end_img = images[:2]
    
    # 确保输入图像是RGB格式
    start_img_rgb = cv2.cvtColor(start_img, cv2.COLOR_HSV2RGB)
    end_img_rgb = cv2.cvtColor(end_img, cv2.COLOR_HSV2RGB)
    
    gif_frames.append(start_img_rgb)
    for step in range(1, steps + 1):
        alpha = step / (steps + 1)
        transition_frame = cv2.addWeighted(start_img_rgb, 1 - alpha, end_img_rgb, alpha, 0)
        gif_frames.append(transition_frame)
    gif_frames.append(end_img_rgb)
    
    mimsave(output_path, gif_frames, 'GIF', duration=0.1)


if __name__ == '__main__':
    image_dir = '/root/autodl-tmp/before_after'
    output_dir = '/root/autodl-tmp/Three_style'
    image_paths = [os.path.join(image_dir, img_name) for img_name in os.listdir(image_dir)]
    mask_img_list = []
    gif_img = []
    street_color_data = {}
    for image_path in image_paths:
        file_basename_with_jpg = os.path.basename(image_path)
        file_basename = file_basename_with_jpg.split('.jpg')[0]
        masked_img, channel_img = extract_features_and_mask(image_path, 0)  # Assuming s channel for features
        mask_img_list.append(masked_img)
        gif_img.append(channel_img)
        colors = analyze_single_img(masked_img)
        sorted_colors = sort_colors_by_hue(colors[0])
        street_color_data[file_basename] = sorted_colors
        plot_color_spectrum(sorted_colors, os.path.join(output_dir, f'{file_basename}_sorted_by_hue Color Spectrum'))

    # Assume the first two images for GIF
    create_fading_gif([mask_img_list[0], mask_img_list[1]], os.path.join(output_dir, 'analyze.gif'))