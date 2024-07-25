import cv2
import numpy as np
import os
import glob

# Your image folders
image_folder = '/root/autodl-tmp/Three_style/S3'
image_folder_save = '/root/autodl-tmp/Three_style/S3_new'

# Get all jpg images in the folder
image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))


# Function to apply histogram equalization to the L channel in LAB color space
def equalize_histogram_l_channel(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # Convert to LAB color space
    l, a, b = cv2.split(lab)                    # Split into channels
    l_hist = cv2.equalizeHist(l)                # Equalize histogram on the L channel
    lab_hist = cv2.merge((l_hist, a, b))        # Merge channels
    img_hist = cv2.cvtColor(lab_hist, cv2.COLOR_LAB2BGR)  # Convert back to BGR color space
    return img_hist


# Function to apply white balance using OpenCV xphoto module
def apply_white_balance(img):
    wb = cv2.xphoto.createSimpleWB()  # Create a white balance object
    balanced_img = wb.balanceWhite(img)  # Apply white balance
    return balanced_img


# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Read image using OpenCV
    # img = equalize_histogram_l_channel(img)  # Apply histogram equalization
    img_balanced = apply_white_balance(img)  # Apply white balance
    save_path = os.path.join(image_folder_save, 'preprocessed_' + os.path.basename(image_path))
    cv2.imwrite(save_path, img_balanced)  # Save the processed image


# Apply the preprocessing to all images
def batch_preprocess_images(image_paths):
    for path in image_paths:
        preprocess_image(path)


# Main entry point
if __name__ == '__main__':
    batch_preprocess_images(image_paths)