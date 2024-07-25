import csv
import shutil
import os


def filter_and_copy_images(csv_file, target_folder, low_threshold=3.0, high_threshold=4.5):
    """Filter images based on color entropy and copy them to a target folder."""
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            color_entropy = float(row['color_entropy'])
            if color_entropy < low_threshold or color_entropy > high_threshold:
                image_path = row['image']
                # 构建目标文件路径
                target_path = os.path.join(target_folder, os.path.basename(image_path))
                # 复制文件
                shutil.copy(image_path, target_path)

# 示例用法
csv_file = '/root/autodl-tmp/color_complexity_analysis/color_city_style_S3_new_complexity_results.csv'  # 替换为CSV文件的路径
target_folder = '/root/autodl-tmp/color_complexity_analysis/color_city_style_S3_complexity_results'  # 替换为目标文件夹的路径
os.makedirs(target_folder, exist_ok=True)
filter_and_copy_images(csv_file, target_folder)
