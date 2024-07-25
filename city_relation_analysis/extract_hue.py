import cv2
import numpy as np
import os
import csv


def save_results_to_csv(results, filename):
    """Save the complexity metrics of images to a CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['image', 'hue_avg']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def calculate_hue_average(image_path, brightness_threshold=20):
    """计算给定图片Hue通道的平均值，排除过暗的像素"""
    img = cv2.imread(image_path)
    # 按原始比例缩小图像以减少计算量
    img_r_h, img_r_w = img.shape[0] // 5, img.shape[1] // 5
    img = cv2.resize(img, (img_r_w, img_r_h))  # 修正了这里的参数顺序，应为宽度在前，高度在后
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 从BGR转换到HSV

    # 提取亮度（Value）通道
    brightness = img_hsv[:, :, 2]

    # 提取Hue通道，但只考虑亮度大于阈值的像素
    hue_channel = img_hsv[:, :, 0][brightness >= brightness_threshold]

    # 计算过滤后的Hue值的平均值
    if hue_channel.size > 0:  # 确保数组非空
        return np.mean(hue_channel)
    else:
        return 0  # 如果过滤后没有剩余像素，返回0


def analyze_folder_for_hue_averages(folder_path, output):
    """分析文件夹中所有图片的Hue平均值，并保存到文本文件"""
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                   f.endswith('.jpg')]
    hue_averages = []
    hue_averages_dic = []

    csv_filename = os.path.join(output, f'hue_results.csv')
    for image_path in image_paths:
        hue_avg = calculate_hue_average(image_path)
        results = {'image': image_path, 'hue_avg': hue_avg}
        hue_averages_dic.append(results)
        hue_averages.append(hue_avg)

    save_results_to_csv(hue_averages_dic, csv_filename)

    hue_averages_total = sum(hue_averages) / len(hue_averages)

    # 保存Hue平均值到文本文件
    with open('hue_averages.txt', 'a+') as f:
        f.write(f"{output}: {hue_averages_total}\n")


if __name__ == '__main__':
    tot = [os.path.join('/Selected/B&A', path_i) for path_i in os.listdir('/Selected/B&A')]

    for folder_path in tot:
        if os.path.isfile(folder_path):
            continue
        analyze_folder_for_hue_averages(folder_path, output=folder_path)
