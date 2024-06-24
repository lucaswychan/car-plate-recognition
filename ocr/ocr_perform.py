import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ocr.resnet import ResNet


# Define the ResNet model
def get_model(num_classes=47):
    ocr_model = ResNet(num_classes=num_classes)
    ocr_model.load_state_dict(
        torch.load(
            "model/finetuned_ocr.pth",
            map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )
    )
    ocr_model.eval()

    return ocr_model


ocr_model = get_model()


def preprocess_image(image, border_size=3, border_color=255):
    # Add a border around the text region
    image = cv2.copyMakeBorder(
        image,
        border_size,
        border_size,
        border_size,
        border_size,
        cv2.BORDER_CONSTANT,
        value=border_color,
    )

    image = Image.fromarray(image)

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (
                    0.1307,
                    0.1307,
                    0.1307,
                ),
                (
                    0.3081,
                    0.3081,
                    0.3081,
                ),
            ),
        ]
    )
    return transform(image).unsqueeze(0)


def segment_license(image_path, saved_dir="text_regions"):
    def resize_image():
        image = cv2.imread(image_path)

        height, width = image.shape[0], image.shape[1]

        width_new = 600
        height_new = 140

        if width / height >= width_new / height_new:
            img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
        else:
            img_new = cv2.resize(image, (int(width * height_new / height), height_new))
        return img_new

    if os.path.exists(saved_dir):
        os.system(f"rm -rf {saved_dir}")
    os.makedirs(saved_dir)

    image = resize_image()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    segmented_license = []

    thresh1 = thresh.copy()
    high, width = thresh1.shape
    arr1 = np.zeros(high, dtype=int)
    for row in range(high):
        for column in range(width):
            if thresh1[row, column] == 0:
                arr1[row] += 1
                thresh1[row, column] = 255
        for column in range(arr1[row]):
            thresh1[row, column] = 0

    thresh2 = thresh.copy()
    high, width = thresh2.shape
    arr2 = np.zeros(width, dtype=int)
    for column in range(width):
        for row in range(high):
            if thresh2[row, column] == 0:
                arr2[column] += 1
                thresh2[row, column] = 255
    for column in range(width):
        for row in range(high - arr2[column], high):
            thresh2[row, column] = 0

    # Find text regions based on horizontal projection
    text_regions = []
    start_row = None
    for row in range(high):
        if arr1[row] > 0 and start_row is None:
            start_row = row
        elif arr1[row] == 0 and start_row is not None:
            end_row = row - 1
            text_regions.append((start_row, end_row))
            start_row = None

    # Split text regions vertically
    for i, region in enumerate(text_regions):
        start_row, end_row = region
        # Vertical histogram projection
        arr2 = np.zeros(width, dtype=int)
        for column in range(width):
            for row in range(start_row, end_row + 1):
                if thresh[row, column] == 0:
                    arr2[column] += 1

        # Find text lines based on vertical projection
        text_lines = []
        start_column = None
        for column in range(width):
            if arr2[column] > 0 and start_column is None:
                start_column = column
            elif arr2[column] == 0 and start_column is not None:
                end_column = column - 1
                text_lines.append((start_column, end_column))
                start_column = None

        # Split text lines horizontally
        for j, line in enumerate(text_lines):
            start_column, end_column = line
            # Extract text region
            text_region = thresh[start_row : end_row + 1, start_column : end_column + 1]

            # Check if the height or width of the text region is smaller than the threshold
            min_height = 20
            min_width = 20
            if text_region.shape[0] < min_height or text_region.shape[1] < min_width:
                continue

            # Save text region as an image file
            output_path = f"{saved_dir}/text_region_{i}_{j}.png"
            segmented_license.append(text_region)
            cv2.imwrite(str(output_path), text_region)

    return segmented_license


def ocr(image_path):
    segmented_license = segment_license(image_path)

    car_plate = ""
    confidences = []

    for image in segmented_license:
        image = preprocess_image(image)

        with torch.no_grad():
            prediction = ocr_model(image)
            prediction = torch.softmax(prediction, dim=1)

        confidence, class_label = torch.max(prediction, 1)
        class_label = class_label.item()

        if 0 <= class_label <= 9:
            output = str(class_label)
        elif 10 <= class_label <= 35:
            output = chr(class_label + 55)
        else:
            output = chr(class_label + 61)

        car_plate += output
        confidences.append(confidence.item())

    return confidences, car_plate
