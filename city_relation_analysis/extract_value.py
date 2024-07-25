import cv2
import numpy as np
import os
import csv


def save_results_to_csv(results, filename):
    """Save the complexity metrics of images to a CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['image', 'value_avg']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def calculate_value_average(image_path, value_threshold=20):
    """计算给定图片亮度（Value）通道的平均值，排除亮度低于阈值的像素"""
    # 读取图像并转换到HSV色彩空间
    img = cv2.imread(image_path)
    # 按原始比例缩小图像以减少计算量
    img_r_h, img_r_w = img.shape[0] // 5, img.shape[1] // 5
    img = cv2.resize(img, (img_r_w, img_r_h))  # 修正了这里的参数顺序，应为宽度在前，高度在后
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 提取Value通道
    value_channel = img_hsv[:, :, 2]

    # 计算亮度平均值，只考虑亮度高于阈值的像素
    mask = value_channel > value_threshold
    average_value = np.mean(value_channel[mask]) if np.any(mask) else 0

    return average_value


def analyze_folder_for_value_averages(folder_path, output):
    # 获取文件夹中所有jpg图片的路径
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                   f.endswith('.jpg')]
    # 存储每张图片的平均亮度值
    value_averages = []
    value_averages_dic = []

    csv_filename = os.path.join(output, f'value_results.csv')
    for image_path in image_paths:
        average_value = calculate_value_average(image_path)
        results = {'image': image_path, 'value_avg': average_value}
        value_averages.append(average_value)
        value_averages_dic.append(results)

    save_results_to_csv(value_averages_dic, csv_filename)

    value_averages_total = sum(value_averages) / len(value_averages)

    # 保存亮度平均值到文本文件
    with open('value_averages.txt', 'a+') as f:
        f.write(f"{output}: {value_averages_total}\n")


if __name__ == '__main__':
    tot = [os.path.join('/Selected/B&A', path_i)
           for path_i in os.listdir('/Selected/B&A')]

    for folder_path in tot:
        if os.path.isfile(folder_path):
            continue
        analyze_folder_for_value_averages(folder_path, output=folder_path)
