import cv2
import numpy as np
import os
import shutil


# 定义红色、紫色和绿色在HSV色彩空间中的范围
color_ranges = {
    'red': ((0, 50, 50), (10, 255, 255)),  # 红色的HSV范围
    'purple': ((140, 50, 50), (160, 255, 255)),  # 紫色的HSV范围
    'green': ((50, 50, 50), (70, 255, 255))  # 绿色的HSV范围
}

# 检查一个图像是否包含特定颜色且该颜色区域超过一定大小
def contains_color(image_path, color_range, min_area=700):
    # 读取图像
    image = cv2.imread(image_path)
    # 将图像从BGR转换到HSV色彩空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 应用颜色范围
    mask = cv2.inRange(hsv_image, color_range[0], color_range[1])
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历所有找到的轮廓
    for contour in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(contour)
        # 如果面积大于最小面积阈值，则返回True
        if area > min_area:
            return True
    # 如果没有找到足够大的颜色区域，则返回False
    return False

# 主函数
def find_colored_houses(folders):
    colored_houses = {'red': [], 'purple': [], 'green': []}
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            for color, color_range in color_ranges.items():
                if contains_color(file_path, color_range):
                    colored_houses[color].append(file_path)
                    break  # 假设一栋房子只能归类为一种颜色
    return colored_houses


# 主函数，现在包括复制文件的功能
def find_and_copy_colored_houses(folders, destination_folder):
    # 为每种颜色创建一个新的子文件夹
    colored_house_folders = {color: os.path.join(destination_folder, color) for color in color_ranges}
    for color_folder in colored_house_folders.values():
        if not os.path.exists(color_folder):
            os.makedirs(color_folder)
    
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            for color, color_range in color_ranges.items():
                if contains_color(file_path, color_range):
                    # 构建目标文件路径
                    dest_path = os.path.join(colored_house_folders[color], filename)
                    # 复制文件
                    shutil.copy(file_path, dest_path)
                    break  # 如果一栋房子只能归类为一种颜色，使用break

# 调用主函数进行色彩检索并复制文件
if __name__ == '__main__':
    # 定义要检索的文件夹
    folders_to_search = [os.path.join('/root/autodl-tmp/yolov5/yolov5/runs/detect/exp3/crops', im) for im in os.listdir('/root/autodl-tmp/yolov5/yolov5/runs/detect/exp3/crops')]
    destination_folder = 'Colored_Houses'  # 保存结果的目标文件夹
    find_and_copy_colored_houses(folders_to_search, destination_folder)

    print(f"Colored houses copied to {destination_folder}.")
