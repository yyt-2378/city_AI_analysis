import cv2
import numpy as np
import os
import csv


def save_results_to_csv(results, filename):
    """Save the complexity metrics of images to a CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['image', 'saturation_avg']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def calculate_saturation_average(image_path, brightness_threshold=20):
    """计算给定图片饱和度通道的平均值，排除过暗的像素"""
    img = cv2.imread(image_path)
    # 按原始比例缩小图像以减少计算量
    img_r_h, img_r_w = img.shape[0] // 5, img.shape[1] // 5
    img = cv2.resize(img, (img_r_w, img_r_h))  # 修正了这里的参数顺序，应为宽度在前，高度在后
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 从BGR转换到HSV
    # 提取亮度（Value）和饱和度（Saturation）通道
    brightness_channel = img_hsv[:, :, 2]
    saturation_channel = img_hsv[:, :, 1]

    # 根据亮度阈值过滤像素
    filtered_saturation = saturation_channel[brightness_channel > brightness_threshold]

    # 计算过滤后的饱和度值的平均值
    if filtered_saturation.size > 0:  # 确保数组非空
        return np.mean(filtered_saturation)
    else:
        return 0  # 如果过滤后没有剩余像素，返回0


def analyze_folder_for_saturation_averages(folder_path, output):
    """分析文件夹中所有图片的饱和度平均值，并保存到文本文件"""
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                   f.endswith('.jpg')]
    saturation_averages = []
    saturation_averages_dic = []

    csv_filename = os.path.join(output, f'saturation_results.csv')
    for image_path in image_paths:
        saturation_avg = calculate_saturation_average(image_path)
        results = {'image': image_path, 'saturation_avg': saturation_avg}
        saturation_averages_dic.append(results)
        saturation_averages.append(saturation_avg)

    save_results_to_csv(saturation_averages_dic, csv_filename)

    sat_averages_total = sum(saturation_averages) / len(saturation_averages)

    # 保存饱和度平均值到文本文件
    with open('Selected/saturation_averages.txt', 'a+') as f:
        f.write(f"{output}_saturation_averages: {sat_averages_total}\n\n")


if __name__ == '__main__':
    tot = [os.path.join('/Selected/B&A', path_i) for path_i in os.listdir('/Selected/B&A')]

    for folder_path in tot:
        if os.path.isfile(folder_path):
            continue
        analyze_folder_for_saturation_averages(folder_path, output=folder_path)
